#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4
#pragma warning(disable : 4996)

struct student{
	char* forename;
	char* surname;
	float average_module_mark;
};

void print_student(const struct student* s);

void main(){
	struct student* students = malloc(sizeof(struct student)*NUM_STUDENTS);
	int i = 0;

	FILE *f = NULL;
	f = fopen("../students2.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find students2.bin file \n");
		exit(1);
	}

	while (1) {
		int n;
		size_t elemsRead = fread(&n, sizeof(int), 1, f);
		if (elemsRead < 1) break;

		// Forename
		if (n > 0) {
			char* fore = malloc(sizeof(char)*n);
			fread(fore, sizeof(char), n, f);
			students[i].forename = fore;
		} else {
			fprintf(stderr, "Error: Length of forename is less than 1 \n");
			exit(1);
		}

		// Surname
		fread(&n, sizeof(int), 1, f);
		if (n > 0) {
			char* sur = malloc(sizeof(char)*n);
			fread(sur, sizeof(char), n , f);
			students[i].surname = sur;
		} else {
			fprintf(stderr, "Error: Length of surname is less than 1 \n");
			exit(1);
		}

		// Score
		float score;
		fread(&score, sizeof(float), 1, f);
		students[i].average_module_mark = score;

		i++;
	}

	fclose(f);

	for (i = 0; i < NUM_STUDENTS; i++) {
		print_student(&students[i]);
	}


	for (i = 0; i < NUM_STUDENTS; i++) {
		free(students[i].forename);
		free(students[i].surname);
	}
	free(students);
}

void print_student(const struct student* s){
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}

