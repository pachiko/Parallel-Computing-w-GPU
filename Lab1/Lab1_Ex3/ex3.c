#include <stdio.h>
#include <limits.h>
#include "random.h"

#define NUM_VALUES 250

float value[NUM_VALUES];

int main()
{
	init_random();
	float sum = 0;
	unsigned char i = 0;

	for (i = 0; i < NUM_VALUES; i++) 
	{
		value[i] = random_float();
		//printf("i: %d, value[i]: %d\n", i, value[i]);
		sum += value[i];
	}

	printf("Sum: %.0f\n", sum);
	float average = sum / NUM_VALUES;
	printf("Average: %.0f\n", average);

	float minVal = LLONG_MAX, maxVal = LLONG_MIN;
	for (i = 0; i < NUM_VALUES; i++)
	{
		float v = (float) value[i] - average;

		minVal = minVal > v ? v : minVal;
		maxVal = maxVal < v ? v : maxVal;
	}

	printf("Min: %.0f\n", minVal);
	printf("Max: %.0f\n", maxVal);

	return 0;
}