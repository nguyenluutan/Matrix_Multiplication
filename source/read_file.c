// https://stackoverflow.com/questions/3501338/c-read-file-line-by-line

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "libraries/array_2d.h"
#include <dirent.h>

void read_single_file(char *file_name, bool is_show_log) {

	struct group_array group;
	read_file(file_name, &group);
	struct array_2d result = multiply(group.matrix_a, group.matrix_b);
	bool is_equals = is_matrices_equal(group.matrix_c, result);
	if (is_show_log) {
		printf("Print test matrix a\n");
		print_matrix(group.matrix_a);

		printf("Print test matrix b\n");
		print_matrix(group.matrix_b);

		printf("Print test matrix c\n");
		print_matrix(group.matrix_c);

		printf("Print actual result c\n");
		print_matrix(result);
	}

	printf("Is calculation true: %s", is_equals?"True":"False");
}

int main()
{
	char * directory_path = "../Data/data_small/";
    DIR *directory = opendir(directory_path);;
    struct dirent* file;
    FILE *a;
    char ch;

    while ((file=readdir(directory)) != NULL) {
		char * path = "../Data/data_small/";
		char * filename;
		sprintf(filename, "%s/%s", path, file->d_name);
        printf("%s\n", filename);
		read_single_file(filename, false);
		printf("\n");
    }
    closedir(directory);

	// char *file_name = "../Data/data_small/Test_236_3x9_dot_9x7_False_10000_-10000.txt";
	// read_single_file(file_name, false);
	exit(EXIT_SUCCESS);
	return 0;
}
