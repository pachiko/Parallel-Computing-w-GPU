#include <stdio.h>
#include <limits.h>
#include "random.h"

#define NUM_VALUES 250

long long value[NUM_VALUES];

int main()
{
	init_random();
	unsigned long long sum = 0;
	unsigned char i = 0;

	for (i = 0; i < NUM_VALUES; i++) 
	{
		value[i] = random_uint();
		//printf("i: %d, value[i]: %d\n", i, value[i]);
		sum += value[i];
	}

	printf("Sum: %llu\n", sum);
	unsigned long long average = sum / NUM_VALUES;
	printf("Average: %llu\n", average);

	long long minVal = LLONG_MAX, maxVal = LLONG_MIN;
	for (i = 0; i < NUM_VALUES; i++)
	{
		long long v = (long long) value[i] - average;

		minVal = minVal > v ? v : minVal;
		maxVal = maxVal < v ? v : maxVal;
	}

	printf("Min: %lld\n", minVal);
	printf("Max: %lld\n", maxVal);

	return 0;
}