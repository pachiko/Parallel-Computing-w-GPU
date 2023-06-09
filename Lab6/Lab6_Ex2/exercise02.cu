#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define A_WIDTH 1024
#define A_HEIGHT 1024
#define B_WIDTH 1024
#define B_HEIGHT 1024
#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

__constant__ int BLOCK_SIZE;
__constant__ int NUM_SUBS;

__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

void checkCUDAError(const char *msg);
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]);
int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);

__global__ void matrixMulCUDA()
{
    // Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = bx*BLOCK_SIZE + tx;
	int y = by*BLOCK_SIZE + ty;
    

	float Csub = 0;
	//iterate A_WIDTH (same as B_HEIGHT) to calculate the product
	for (int k = 0; k < A_WIDTH; k++){
		Csub += d_A[y][k] * d_B[k][x]; 
	}

	// Store the product value of C matrix
	d_C[y][x] = Csub;
}

__global__ void matrixMulCUDASharedMemory()
{
    //Define some shared memory for a sub block of matrices A an B
	extern __shared__ float sm[];
	float* As = sm;
	float* Bs = sm + BLOCK_SIZE * BLOCK_SIZE;
    
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
    //Running sum of product of A and B matrices
    float Csub = 0;
 
	//iterate through the number of sub matrices of A and B
	for (int i = 0; i < NUM_SUBS; i++){
		//TODO: Calculate indices of A and B matrix required to load the shared block of memory
		int a_x = i * BLOCK_SIZE + tx; // All columns of A
		int a_y = by * BLOCK_SIZE + ty; // row of A
		int b_x = bx * BLOCK_SIZE + tx; // column of B
		int b_y = i * BLOCK_SIZE + ty; // All rows of B
        
        //TODO: Each thread should load a single element of sub block of matrices A an B into shared memory
		As[ty*BLOCK_SIZE + tx] = d_A[a_y][a_x];
		Bs[ty*BLOCK_SIZE + tx] = d_B[b_y][b_x];
        
        // Sync to ensure sub matrix is fully loaded
		__syncthreads();
        
        //TODO: sum products of A and B sub matrices
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty*BLOCK_SIZE + k] * Bs[k*BLOCK_SIZE + tx];
		}
        
        // Sync to prevent run ahead (blocks loading new SM values before others have completed)
		__syncthreads();
        
	}

    //TODO: caluclate the indices of sub matrix C
	int c_x = bx * BLOCK_SIZE + tx;
	int c_y = by * BLOCK_SIZE + ty;
    
	// Store the product value of C matrix
	d_C[c_y][c_x] = Csub;
}


unsigned int requiredSM(unsigned int blockSize) {
	return (blockSize * sizeof(float) * 2);
}


int main(int argc, char **argv)

{
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	unsigned int x, y, errors;
	int maxActiveBlocks;
	float msec, occupancy;
	cudaDeviceProp props;
	cudaEvent_t start, stop;

	if (A_WIDTH != B_HEIGHT){
		printf("Error: A_HEIGHT and B_WIDTH do not match\n");
	}

	mem_size_A = sizeof(float)* A_WIDTH* A_HEIGHT;
	mem_size_B = sizeof(float)* B_WIDTH* B_HEIGHT;
	mem_size_C = sizeof(float)* C_WIDTH* C_HEIGHT;

	// Initialise A
	for (y = 0; y < A_HEIGHT; y++)
	for (x = 0; x <A_WIDTH; x++)
		h_A[y][x] = (float)rand() / RAND_MAX*2.f;
	// Initialise B
	for (y = 0; y < B_HEIGHT; y++)
	for (x = 0; x <B_WIDTH; x++)
		h_B[y][x] = (float)rand() / RAND_MAX*2.f;

	// copy host memory to device
	cudaMemcpyToSymbol(d_A, h_A, mem_size_A);
	cudaMemcpyToSymbol(d_B, h_B, mem_size_B);
	checkCUDAError("CUDA memcpy");

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCUDAError("CUDA event creation");

	// Setup execution parameters
	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, matrixMulCUDASharedMemory, requiredSM);
	blockSize = pow(4, floor(log(blockSize)/log(4)));
	blockSize = sqrt(blockSize);
	printf("Calculated Block Size with highest occupancy: %d\n", blockSize);


	int gridSize_X = (C_WIDTH + blockSize - 1) / blockSize;
	int gridSize_Y = (C_HEIGHT + blockSize - 1) / blockSize;

	cudaMemcpyToSymbol(BLOCK_SIZE, &blockSize, sizeof(int));
	int numSubs = A_WIDTH / blockSize;
	cudaMemcpyToSymbol(NUM_SUBS, &numSubs, sizeof(int));

	dim3 threads(blockSize, blockSize);
	dim3 grid(gridSize_X, gridSize_Y);
	cudaEventRecord(start);
	
    
    //matrixMulCUDA << < grid, threads >> >();
    //TODO: Comment out the above line and complete the shared memory version of the kernel
	matrixMulCUDASharedMemory << < grid, threads, requiredSM(blockSize*blockSize) >> > ();

    
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	checkCUDAError("CUDA kernel execution and timing");

	cudaEventElapsedTime(&msec, start, stop);
	cudaThreadSynchronize();
	checkCUDAError("CUDA timing");

	// TODO: Compute the ocupancy
	// Device properties
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	for (int dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		int maxActiveBlocksPerMP;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerMP, matrixMulCUDASharedMemory, blockSize * blockSize, 0);
		occupancy = (float)maxActiveBlocksPerMP * blockSize * blockSize / (deviceProp.maxThreadsPerMultiProcessor); // max active (threads or warps) / total (threads or warps)
	}

	// Copy result from device to host
	cudaMemcpyFromSymbol(h_C, d_C, mem_size_C);
	checkCUDAError("CUDA memcpy results");

	// Compute reference CPU version
	matrixMulCPU(h_A, h_B, h_C_ref);

	// Check for errors
	errors = matrixMulTest(h_C, h_C_ref);
	if (errors)
		printf("%d total errors\n", errors);
	else
		printf("Test passed successfully\n");

	printf("Kernel time was %f with theoretical occupancy of %f\n", msec, occupancy);

}


void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH])
{
	int x, y, k;
	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			C[y][x] = 0;
			for (k = 0; k < A_WIDTH; k++){
				C[y][x] += A[y][k] * B[k][x];
			}
		}
	}

}

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH])
{
	int errors = 0;
	int y, x;

	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			float diff = C[y][x] - Cref[y][x];
			if (diff < 0) diff = -diff;
			if (diff > 1e-3) {
				errors++;
				printf("Device item c[%d][%d] = %f does not mach host result %f\n", y, x, C[y][x], Cref[y][x]);
				return errors;
			}
		}
	}

	return errors;
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