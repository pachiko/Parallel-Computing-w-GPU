#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#define BUFFER_SIZE 32
#pragma warning(disable : 4996)

int readLine(char buffer[]);

int main()
{
    float in_value = 0, sum;
	char buffer [BUFFER_SIZE];
	char command [4];
    sum = 0;

	const char* fName = "commands.calc";
	FILE* f = fopen(fName, "r");

	if (f == NULL) {
		fprintf(stderr, "File not found: %s\n", fName);
		exit(1);
	}

    while (readLine(buffer, f)){
		//4.5 Check that the line contains 3 letters and a spaceextract
		bool valid = isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3] == ' ';

		if (!valid) {
			fprintf(stderr, "Incorrect command format\n");
			exit(1);
		}

		//4.6 Extract the command and in_value using sscanf
		sscanf(buffer, "%s %f", command, &in_value);


		if (strcmp(command, "add") == 0){ //4.7 Change condition to check command to see if it is "add"
			sum += in_value;
		}
		//4.8 Add else if conditions for sub, mul and div
		else if (strcmp(command, "sub") == 0) {
			sum -= in_value;
		}
		else if (strcmp(command, "mul") == 0) {
			sum *= in_value;
		}
		else if (strcmp(command, "div") == 0) {
			sum /= in_value;
		}
		else {
			fprintf(stderr, "Unknown command\n");
			exit(1);
		}
	}

	printf("Sum is %f\n", sum);
	fclose(f);

    return 0;
}

int readLine(char buffer[], FILE* f){
	int i=0;
	char c=0;

	bool done = false;
	do {
		c = fgetc(f);
		//4.1 Add character to buffer
		buffer[i++] = c;
		//4.2 Check index for overflow
		if (i > BUFFER_SIZE) {
			fprintf(stderr, "Potential Overflow!\n");
			exit(1);
		}

		done = c == '\n' || c == EOF;
	} while (!done);


	//4.3 Ensure the buffer is correctly terminated
	if (c == EOF) {
		return 0;
	}
	buffer[i] = '\0';

	//4.4 Return 0 if buffer = "exit" otherwise return 1
	if (strcmp(buffer, "exit") == 0) {
		return 0;
	}
	return 1;
}