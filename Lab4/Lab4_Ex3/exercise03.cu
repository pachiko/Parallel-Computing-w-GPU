#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048
#define M 1024
#define THREADS_PER_BLOCK_Y 32
#define THREADS_PER_BLOCK_X 16

void checkCUDAError(const char*);
void random_ints(int *a);

__global__ void matrixAdd(int *a, int *b, int *c) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x);
	c[i] = a[i] + b[i];
}

void matrixAddCPU(int* a, int* b, int* c, int max) {
	for (int i = 0; i < max; i++) {
		c[i] = a[i] + b[i];
	}
}

int validate(int* test, int* ref, int max) {
	int err_count = 0;
	for (int i = 0; i < max; i++) {
		int x = test[i];
		int y = ref[i];
		if (x != y) {
			err_count++;
			printf("ERROR at index %d! Expected: %d; Got: %d\n", i, y, x);
		}
	}
	return err_count;
}

int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * M * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	int blocks_X = M / THREADS_PER_BLOCK_X;
	if (blocks_X * THREADS_PER_BLOCK_X < M) blocks_X++;

	int blocks_Y = N / THREADS_PER_BLOCK_Y;
	if (blocks_Y * THREADS_PER_BLOCK_Y < N) blocks_Y++;

	dim3 blocksPerGrid(blocks_X, blocks_Y, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
	matrixAdd <<< blocksPerGrid, threadsPerBlock >>> (d_a, d_b, d_c);
	checkCUDAError("CUDA kernel");

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	// Check!
	matrixAddCPU(a, b, c_ref, N*M);
	errors = validate(c, c_ref, N*M);
	if (errors > 0) {
		printf("Error count: %d\n", errors);
	}

	// Cleanup
	free(a); free(b); free(c);
	free(c_ref);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

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
	for (unsigned int i = 0; i < N*M; i++){
		a[i] = rand();
	}
}
