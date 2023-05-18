#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mandelbrot.h"

#pragma warning(disable : 4996)

//image size
#define WIDTH 1024
#define HEIGHT 768

#define MAX_ITERATIONS 10000 //number of iterations

//C parameters (modify these to change the zoom and position of the mandelbrot)
#define ZOOM 1.0
#define X_DISPLACEMENT -0.5
#define Y_DISPLACEMENT 0.0


static int iterations[HEIGHT][WIDTH];					//store the escape time (iteration count) as an integer
static double iterations_d[HEIGHT][WIDTH];				//store for the escape time as a double (with fractional part) for NIC method only
static int histogram[MAX_ITERATIONS + 1];					//histogram if escape times
static rgb rgb_output[HEIGHT][WIDTH];					//output data

const TRANSFER_FUNCTION tf = HISTOGRAM_NORMALISED_ITERATION_COUNT;

int main(int argc, char *argv[])
{
	double begin, end;									//timers
	double elapsed;										//elapsed time
	FILE *f;											//output file handle


	//open the output file and write header info for PPM filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# COM4521 Lab 03 Exercise02\n");
	fprintf(f, "%d %d\n%d\n", WIDTH, HEIGHT, 255);

	//start timer
	begin = omp_get_wtime();

	//clear the histogram initial values
	memset(histogram, 0, sizeof(histogram));

	int** local_histograms = NULL;
	int num_threads;

	#pragma omp parallel
	{
		//STAGE 0) Initialize local histograms
		//#pragma omp single // 'single' has an implicit barrier at the end
		#pragma omp master
		{
			num_threads = omp_get_num_threads();
			local_histograms = malloc(sizeof(int*) * num_threads);
			for (int t = 0; t < num_threads; t++) {
				local_histograms[t] = malloc(sizeof(histogram));
				memset(local_histograms[t], 0, sizeof(histogram));
				//printf("Malloc-ed for thread %d\n", t);
				//printf("Malloc-ed %d elements\n", sizeof(histogram));
			}
		}
		#pragma omp barrier // 'for' has an implicit barrier at the end, not the start

		//STAGE 1) calculate the escape time for each pixel
		int y;
		//#pragma omp parallel for // for the basic case, but works with '#pragma omp parallel' too
		#pragma omp for schedule(dynamic)
		for (y = 0; y < HEIGHT; y++)
		{
			for (int x = 0; x < WIDTH; x++)
			{
				//zero complex number values
				double n_r = 0;
				double n_i = 0;
				double o_r = 0;
				double o_i = 0;

				//calculate the initial real and imaginary part of z defined by the complex polynomial z-> z^2 + c  where c is the initial parameter based on zoom and displacement
				double c_r = 1.5 * (x - WIDTH / 2) / (0.5 * ZOOM * WIDTH) + X_DISPLACEMENT;
				double c_i = (y - HEIGHT / 2) / (0.5 * ZOOM * HEIGHT) + Y_DISPLACEMENT;

				//iterate to find how many iterations before outsde the julia set
				int i;
				for (i = 0; (i < MAX_ITERATIONS) && ((n_r * n_r + n_i * n_i) < ESCAPE_RADIUS_SQ); i++)
				{
					//store current values
					o_r = n_r;
					o_i = n_i;

					//apply mandelbrot function
					n_r = o_r * o_r - o_i * o_i + c_r;
					n_i = 2.0 * o_r * o_i + c_i;
				}

				//escape time algorithm if using HISTOGRAM_NORMALISED_ITERATION_COUNT transfer function or RANDOM_NORMALISED_ITERATION_COUNT
				if ((tf == HISTOGRAM_NORMALISED_ITERATION_COUNT) && (i < MAX_ITERATIONS)) {
					double mu = (double)i - log(log(sqrt(n_r * n_r + n_i * n_i))) / log(2);
					iterations_d[y][x] = mu;
					i = (int)mu;
				}

				iterations[y][x] = i;	//record the escape velocity

				if ((tf == HISTOGRAM_ESCAPE_VELOCITY) || (tf == HISTOGRAM_NORMALISED_ITERATION_COUNT)) {
					//#pragma omp critical // noob
					//#pragma omp atomic // less noob
						//histogram[i]++;

					int t = omp_get_thread_num();
					(local_histograms[t][i])++;
				}

			}
		}
	}


	//STAGE 1.5) Accumulate and then free local histograms (in parallel)
	//printf("Accumulate... %d\n", num_threads);
	for (int t = 0; t < num_threads; t++) {
		//printf("Thread %d ..\n", t);
		int r;
		#pragma omp parallel for
		for (r = 0; r < MAX_ITERATIONS + 1; r++)
		{
			histogram[r] += local_histograms[t][r];
		}

		free(local_histograms[t]);
	}

	free(local_histograms);


	//STAGE 2) calculate the transfer (rgb output) for each pixel
	int y;
	#pragma omp parallel for
	for (y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			switch (tf) {
			case (ESCAPE_VELOCITY): {
				rgb_output[y][x] = ev_transfer(x, y);
				break;
			}
			case (HISTOGRAM_ESCAPE_VELOCITY): {
				rgb_output[y][x] = h_ev_tranfer(x, y);
				break;
			}
			case (HISTOGRAM_NORMALISED_ITERATION_COUNT): {
				rgb_output[y][x] = h_nic_transfer(x, y);
				break;
			}

			}
		}
	}

	//STAGE 3) output the madlebrot to a file
	fwrite(rgb_output, sizeof(char), sizeof(rgb_output), f);
	fclose(f);

	//stop timer
	end = omp_get_wtime();

	elapsed = end - begin;
	printf("Complete in %f seconds\n", elapsed);

	return 0;
}


rgb ev_transfer(int x, int y){
	rgb a;
	double hue;
	int its;

	its = iterations[y][x];
	if (its == MAX_ITERATIONS){
		a.r = a.g = a.b = 0;
	}
	else{
		hue = its / (double)MAX_ITERATIONS;
		a.r = a.g = 0;
		a.b = (char)(hue * 255.0); //clamp to range of 0-255
	}
	return a;
}

rgb h_ev_tranfer(int x, int y){
	rgb a;
	double hue;
	int its;
	int i;

	its = iterations[y][x];
	if (its == MAX_ITERATIONS){
		a.r = a.g = a.b = 0;
	}
	else{
		hue = 0;
		for (i = 0; i < its; i++)
			hue += (histogram[i] / (double)(1024 * 768));
		a.r = a.g = 0;
		a.b = (char)(hue * 255.0); //clamp to range of 0-255
	}
	return a;
}

rgb h_nic_transfer(int x, int y){
	rgb a;
	double hue, hue1, hue2, its_d, frac;
	int i, its;

	its_d = iterations_d[y][x];
	its = iterations[y][x];

	hue1 = hue2 = 0;
	for (i = 0; (i < its) && (its<MAX_ITERATIONS); i++)
		hue1 += (histogram[i] / (double)(1024 * 768));
	if (i <= MAX_ITERATIONS)
		hue2 = hue1 + (histogram[i] / (double)(1024 * 768));
	a.r = a.g = 0;
	frac = its_d - (int)its_d;
	hue = (1 - frac)*hue1 + frac*hue2;	//linear interpolation between hues
	a.b = (char)(hue * 255.0);			//clamp to range of 0-255
	return a;
}
