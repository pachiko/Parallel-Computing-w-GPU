#include <stdio.h>
#include <stdlib.h>
#include "linked_list.h"


#define NUM_STUDENTS 4
#pragma warning(disable : 4996)

typedef struct student{
	char* forename;
	char* surname;
	float average_module_mark;
} student;

void print_student(const student* s);

void main(){
	llitem* start = NULL;
	llitem* end = start;
	int i = 0;

	FILE *f = NULL;
	f = fopen("../students2.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find students2.bin file \n");
		exit(1);
	}

	print_callback = &print_student;

	while (1) {
		student* stud = malloc(sizeof(student));

		int n;
		size_t elemsRead = fread(&n, sizeof(int), 1, f);
		if (elemsRead < 1) break;

		// Forename
		if (n > 0) {
			char* fore = malloc(sizeof(char)*n);
			fread(fore, sizeof(char), n, f);
			stud->forename = fore;
		} else {
			fprintf(stderr, "Error: Length of forename is less than 1 \n");
			exit(1);
		}

		// Surname
		fread(&n, sizeof(int), 1, f);
		if (n > 0) {
			char* sur = malloc(sizeof(char)*n);
			fread(sur, sizeof(char), n , f);
			stud->surname = sur;
		} else {
			fprintf(stderr, "Error: Length of surname is less than 1 \n");
			exit(1);
		}

		// Score
		float score;
		fread(&score, sizeof(float), 1, f);
		stud->average_module_mark = score;

		// Create LL node for student
		llitem* hold = create_linked_list();
		hold->record = (void*) stud;
		if (end != NULL) {
			add_to_linked_list(end, hold);
		}
		else {
			start = hold;
		}
		end = hold;
		i++;
	}

	fclose(f);

	print_items(start);

	// Clear students
	llitem* iter = start;
	while (iter != NULL) {
		student* stud = (student*) iter->record;
		free(stud->forename);
		free(stud->surname);
		free(stud);
		iter = iter->next;
	}

	// Clear list
	free_linked_list(start);
}

void print_student(const student* s){
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}

