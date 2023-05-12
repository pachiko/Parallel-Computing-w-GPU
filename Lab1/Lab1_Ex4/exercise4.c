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

	printf("Welcome to basic COM4521 calculator\nEnter command: ");

    while (readLine(buffer)){

		//4.5 Check that the line contains 3 letters and a spaceextract
		bool valid = isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3] == ' ';

		if (!valid) {
			fprintf(stderr, "Incorrect command format\n");
			continue;
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
			if (strncmp(command, "ad", 2) == 0) {
				printf("Did you mean add?\n");
			}
			else if (strncmp(command, "su", 2) == 0) {
				printf("Did you mean sub?\n");
			}
			else if (strncmp(command, "mu", 2) == 0) {
				printf("Did you mean mul?\n");
			}
			else if (strncmp(command, "di", 2) == 0) {
				printf("Did you mean div?\n");
			}
			else {
				printf("Unknown command\n");
			}
		}

		printf("\tSum is %f\n", sum);
		printf("Enter next command: ");
	}

    return 0;
}

int readLine(char buffer[]){
	int i=0;
	char c=0;

	while ((c = getchar()) != '\n'){
        //4.1 Add character to buffer
		buffer[i++] = c;
		//4.2 Check index for overflow
		if (i > BUFFER_SIZE) {
			fprintf(stderr, "Potential Overflow!\n");
			exit(1);
		}
	}
	//4.3 Ensure the buffer is correctly terminated
	buffer[i] = '\0';

	//4.4 Return 0 if buffer = "exit" otherwise return 1
	if (strcmp(buffer, "exit") == 0) {
		return 0;
	}
	return 1;
}