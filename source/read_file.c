// https://stackoverflow.com/questions/3501338/c-read-file-line-by-line

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "libraries/array_2d.h"

int main()
{
	char *file_name = "../Data/data_small/Test_236_3x9_dot_9x7_False_10000_-10000.txt";
	struct group_array group;
	
	read_file(file_name, &group);
	printf("Print test matrix a\n");
	print_matrix(group.matrix_a);

	printf("Print test matrix b\n");
	print_matrix(group.matrix_b);

	printf("Print test matrix c\n");
	print_matrix(group.matrix_c);

	printf("Print actual result c\n");
	struct array_2d result = multiply(group.matrix_a, group.matrix_b);
	print_matrix(result);

	printf("Is right result: %s", is_matrices_equal(group.matrix_c, result)?"True":"False");
	exit(EXIT_SUCCESS);
	return 0;
}
