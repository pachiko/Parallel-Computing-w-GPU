#include <stdio.h>
#include "random.h"

#define NUM_VALUES 250

unsigned int value[NUM_VALUES];

int main()
{
	init_random();
	unsigned int sum = 0;
	unsigned char i = 0;

	for (i = 0; i < NUM_VALUES; i++) 
	{
		value[i] = random_ushort();
		//printf("i: %d, value[i]: %d\n", i, value[i]);
		sum += value[i];
	}

	printf("Sum: %d\n", sum);
	unsigned int average = sum / NUM_VALUES;
	printf("Average: %d\n", average);

	int minVal = INT_MAX, maxVal = INT_MIN;
	for (i = 0; i < NUM_VALUES; i++)
	{
		int v = (int) value[i] - average;

		minVal = minVal > v ? v : minVal;
		maxVal = maxVal < v ? v : maxVal;
	}

	printf("Min: %d\n", minVal);
	printf("Max: %d\n", maxVal);

	return 0;
}