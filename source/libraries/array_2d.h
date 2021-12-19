#include <stdio.h>
#include <stdlib.h>
#define _GNU_SOURCE

struct array_2d
{
	double **arr;
	int column;
	int row;
};

struct group_array {
	struct array_2d matrix_a;
	struct array_2d matrix_b;
	struct array_2d matrix_c;
};

void destroyArray(double **arr)
{
	free(*arr);
	free(arr);
}

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

struct array_2d get_array(char *line)
{
	char *ch = strtok(line, "x");
	int row = atoi(ch);
	ch = strtok(NULL, " ,");
	int column = atoi(ch);
	struct array_2d result = {.arr = createArray(row, column), .column = column, .row = row};
	return result;
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

void read_file(char *file_name, struct group_array * group) {
	FILE *fp;
	char *line = NULL;
	size_t len = 0;
	ssize_t read;
	struct array_2d matrix_a, matrix_b, matrix_c; 

	fp = fopen(file_name, "r");
	if (fp == NULL)
		exit(EXIT_FAILURE);

	int line_count = 0;
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

	group->matrix_a = matrix_a;
	group->matrix_b = matrix_b;
	group->matrix_c = matrix_c;
}