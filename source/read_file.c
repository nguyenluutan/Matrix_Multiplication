// https://stackoverflow.com/questions/3501338/c-read-file-line-by-line

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

struct array_2d
{
	double **arr;
	int column; // col
	int row; // row
};

double **createArray(int column, int row)
{
	double *values = calloc(column * row, sizeof(double));
	double **rows = malloc(column * sizeof(double *));
	for (int i = 0; i < column; ++i)
	{
		rows[i] = values + i * row;
	}
	return rows;
}

void destroyArray(double **arr)
{
	free(*arr);
	free(arr);
}

// void drawLine(const float **coords, int column, int row);

struct array_2d get_array(char *line)
{
	char *ch = strtok(line, "x");
	int row = atoi(ch);
	ch = strtok(NULL, " ,");
	int column = atoi(ch);
	struct array_2d result = {.arr = createArray(row, column), .column = column, .row = row};
	// return createArrwhay(row,column);
	// int result[row][column];
	// return result;
	return result;
}

void parse_row_matrix(char *line, struct array_2d matrix, int row)
{
	char *ch = strtok(line, " ");
	int column_count = 0;
	while (ch != NULL)
	{
		double number = atof(ch);
		matrix.arr[row][column_count++] = number;
		ch = strtok(NULL, " ,");
	}
}

void print_matrix(struct array_2d matrix)
{
	for (int i = 0; i < matrix.row; i++)
	{
		for (int j = 0; j < matrix.column; j++)
		{
			printf("%.4f ", matrix.arr[i][j]);
		}
		printf("\n");
	}
}

bool is_matrices_equal(struct array_2d matrix_a, struct array_2d matrix_b)
{
	if (matrix_a.column != matrix_b.column)
		return false;

	if (matrix_a.row != matrix_b.row)
		return false;

	for(int i = 0; i< matrix_a.row; i++){
		for(int j = 0; j< matrix_a.column; j++) {
			double a = roundl(matrix_a.arr[i][j]*10000)/10000;
			double b = roundl(matrix_b.arr[i][j]*10000)/10000;
			if (a == b)
				continue;
			
			return false;
		}
	}

	return true;
}

struct array_2d multiply(struct array_2d matrix_a, struct array_2d matrix_b)
{
	// 2x3
	// 1 2 3
	// 4 5 6

	// 3x2
	// 4 4
	// 3 3
	// 2 1

	// 1*4+2*3+3*2 1*4+2*3+3*2
	// 4*4+5*3+6*2 4*4+5*3+6*1
	
	// Loop each row of matrix a
	// Loop each column of matrix b
	// Loop each column of matrix a
	// Get element in matrix a multiply with element in matrix b

	struct array_2d result = (struct array_2d){.arr = createArray(matrix_a.row, matrix_b.column), .column = matrix_b.column, .row = matrix_a.row};
	for (int i = 0; i < matrix_a.row; i++)
	{
		for (int j = 0; j < matrix_b.column; j++)
		{
			result.arr[i][j] = 0; 
			for (int k = 0; k < matrix_a.column; k++)
			{
				result.arr[i][j] += matrix_a.arr[i][k] * matrix_b.arr[k][j];
				// printf("%.2fx%.2f with A[%d,%d] B[%d,%d]\n", matrix_a.arr[i][k], matrix_b.arr[k][j], i,k,k,j);
			}
		}
	}

	return result;
}

int main()
{
	FILE *fp;
	char *line = NULL;
	size_t len = 0;
	ssize_t read;
	char *file_name = "../Data/data_small/Test_236_3x9_dot_9x7_False_10000_-10000.txt";

	fp = fopen(file_name, "r");
	if (fp == NULL)
		exit(EXIT_FAILURE);

	int line_count = 0;
	struct array_2d matrix_a, matrix_b, matrix_c;
	while ((read = getline(&line, &len, fp)) != -1)
	{
		line_count++;

		if (line_count == 1)
		{
			printf("%s", line);
			continue;
		}

		if (line_count == 2)
		{
			matrix_a = get_array(line);
			printf("%d x %d\n", matrix_a.row, matrix_a.column);
			continue;
		}
		if (line_count == 3)
		{
			matrix_b = get_array(line);
			printf("%d x %d\n", matrix_b.row, matrix_b.column);

			matrix_c = (struct array_2d){.arr = createArray(matrix_a.row, matrix_b.column), .column = matrix_b.column, .row = matrix_a.row};
			continue;
		}

		if (line_count <= matrix_a.row + 3)
		{
			parse_row_matrix(line, matrix_a, line_count - 4);
			continue;
		}

		if (line_count <= matrix_b.row + matrix_a.row + 3)
		{
			parse_row_matrix(line, matrix_b, line_count - matrix_a.row - 4);
			continue;
		}

		if (line_count <= matrix_c.row + matrix_b.row + matrix_a.row + 3)
		{
			parse_row_matrix(line, matrix_c, line_count - matrix_b.row - matrix_a.row - 4);
			continue;
		}

		// If all lines parse_row_matrixd, there will not show any line up here
		printf("%s", line);
	}

	fclose(fp);
	if (line)
		free(line);

	printf("Print test matrix a\n");
	print_matrix(matrix_a);

	printf("Print test matrix b\n");
	print_matrix(matrix_b);

	printf("Print test matrix c\n");
	print_matrix(matrix_c);

	printf("Print actual result c\n");
	struct array_2d result = multiply(matrix_a, matrix_b);
	print_matrix(result);

	printf("Is right result: %s", is_matrices_equal(matrix_c, result)?"True":"False");
	exit(EXIT_SUCCESS);
	return 0;
}
