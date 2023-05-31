#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 65536
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

// Statically-defined global variables
__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];


__global__ void vectorAdd() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_c[i] = d_a[i] + d_b[i];
}


int main(void) {
	int *a, *b, *c;			// host copies of a, b, c
	unsigned int size = N * sizeof(int);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);

	// Time event
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Device properties
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	for (int dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		// Memory clock rate (kHz); Memory bus width (bits)
		float theoretical_BW = deviceProp.memoryClockRate * deviceProp.memoryBusWidth * 2 / 8 / 1e6;
		printf("Device: %s; Theoretical Memory Bandwidth (GB/s): %f\n", deviceProp.name, theoretical_BW);
	}

	// Copy inputs to device
	cudaMemcpyToSymbol(d_a, a, size);
	cudaMemcpyToSymbol(d_b, b, size);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	cudaEventRecord(start);
	vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >();
	cudaEventRecord(stop);
	checkCUDAError("CUDA kernel");

	// Elapsed time
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds,	start, stop);
	printf("Time elapsed: %f\n", milliseconds);

	// Copy result back to host
	cudaMemcpyFromSymbol(c, d_c, size);
	checkCUDAError("CUDA memcpy");

	// Measured Memory Bandwidth
	int readBytes =  N * sizeof(int);
	int writeBytes = 2 * readBytes;
	float measured_BW = ((readBytes + writeBytes)/1e9) / (milliseconds/1e3);
	printf("Measured Memory Bandwidth (GB/s): %f\n", measured_BW);

	// Cleanup
	free(a); free(b); free(c);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}
