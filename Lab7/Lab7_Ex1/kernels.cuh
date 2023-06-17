#ifndef KERNEL_H //ensures header is only included once
#define KERNEL_H

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_RECORDS 2048
#define THREADS_PER_BLOCK 256
#define SQRT_THREADS_PER_BLOCK sqrt(THREADS_PER_BLOCK)

struct student_record{
	int student_id;
	float assignment_mark;
};

struct student_records{
	int student_ids[NUM_RECORDS];
	float assignment_marks[NUM_RECORDS];
};

typedef struct student_record student_record;
typedef struct student_records student_records;

__device__ float d_max_mark = 0;
__device__ int d_max_mark_student_id = 0;

// lock for global Atomics
#define UNLOCKED 0
#define LOCKED   1
__device__ volatile int lock = UNLOCKED;

// Function creates an atomic compare and swap to save the maximum mark and associated student id
__device__ void setMaxMarkAtomic(float mark, int id) {
	bool needlock = true;

	while (needlock){
		// get lock to perform critical section of code
		if (atomicCAS((int *)&lock, UNLOCKED, LOCKED) == 0){

			//critical section of code
			if (d_max_mark < mark){
				d_max_mark_student_id = id;
				d_max_mark = mark;
			}

			// free lock
			atomicExch((int*)&lock, 0);
			needlock = false;
		}
	}
}

// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float mark = d_records->assignment_marks[idx];
	int id = d_records->student_ids[idx];

	setMaxMarkAtomic(mark, id);

}

//Exercise 2) Recursive Reduction
__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Exercise 2.1) Load a single student record into shared memory
	extern __shared__ student_record shared_records[];

	shared_records[threadIdx.x].assignment_mark = d_records->assignment_marks[idx];
	shared_records[threadIdx.x].student_id = d_records->student_ids[idx];

	__syncthreads();

	//Exercise 2.2) Compare two values and write the result to d_reduced_records
	if (idx % 2 == 0) {
		const student_record* s1 = shared_records + threadIdx.x;
		const student_record* s2 = shared_records + threadIdx.x + 1;
		const student_record* winner = (s1->assignment_mark > s2->assignment_mark) ? s1 : s2;

		idx >>= 1;
		d_reduced_records->assignment_marks[idx] = winner->assignment_mark;
		d_reduced_records->student_ids[idx] = winner->student_id;
	}

}


//Exercise 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Exercise 3.1) Load a single student record into shared memory
	extern __shared__ student_record shared_records[];

	shared_records[threadIdx.x].assignment_mark = d_records->assignment_marks[idx];
	shared_records[threadIdx.x].student_id = d_records->student_ids[idx];

	__syncthreads();

	//Exercise 3.2) Strided shared memory conflict free reduction
	for (int s = blockDim.x / 2; s > 0; s >>= 1) { // decreasing half-block
		if (threadIdx.x < s) { // lower half of block does the work
			float me = shared_records[threadIdx.x].assignment_mark;
			float them = shared_records[threadIdx.x + s].assignment_mark; // upper half
			
			if (them > me) {
				shared_records[threadIdx.x].assignment_mark = them;
				shared_records[threadIdx.x].student_id = shared_records[threadIdx.x + s].student_id;
			}
		}
		__syncthreads();
	}

	//Exercise 3.3) Write the result
	if (threadIdx.x == 0) {
		d_reduced_records->assignment_marks[blockIdx.x] = shared_records[0].assignment_mark;
		d_reduced_records->student_ids[blockIdx.x] = shared_records[0].student_id;
	}
}

//Exercise 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	//Exercise 4.1) Complete the kernel
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int laneId = idx % warpSize;

	float maxScore = d_records->assignment_marks[idx];
	int topId = d_records->student_ids[idx];

	unsigned mask = 0xffffffff;

	bool xor = true;

	if (xor) {
		for (int i = warpSize / 2; i > 0; i >>= 1) {
			float compare = __shfl_xor_sync(mask, maxScore, i);
			int potential = __shfl_xor_sync(mask, topId, i);

			bool swap = compare > maxScore;
			maxScore = swap ? compare : maxScore;
			topId = swap ? potential : topId;
		}
	}
	else {
		for (int i = warpSize / 2; i > 0; i >>= 1) {
			float compare = __shfl_down_sync(mask, maxScore, i);
			int potential = __shfl_down_sync(mask, topId, i);

			bool swap = compare > maxScore;
			maxScore = swap ? compare : maxScore;
			topId = swap ? potential : topId;
		}
	}


	d_reduced_records->assignment_marks[idx] = maxScore;
	d_reduced_records->student_ids[idx] = topId;
}

#endif //KERNEL_H