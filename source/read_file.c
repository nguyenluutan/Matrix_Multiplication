// https://stackoverflow.com/questions/3501338/c-read-file-line-by-line

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

struct array_2d {
	float** arr;
	int m;
	int n;
};

float** createArray(int m, int n)
{
    float* values = calloc(m*n, sizeof(float));
    float** rows = malloc(m*sizeof(float*));
    for (int i=0; i<m; ++i)
    {
        rows[i] = values + i*n;
    }
    return rows;
}

void destroyArray(float** arr)
{
    free(*arr);
    free(arr);
}

void drawLine(const float** coords, int m, int n);


struct array_2d get_array(char* line) {
	char *ch = strtok(line, "x");
	int n = atoi(ch);
	ch = strtok(NULL, " ,");
	int m = atoi(ch);
	struct array_2d result = {.arr = createArray(n,m), .m = m, .n = n};
	// return createArrwhay(n,m);
	// int result[n][m];
	// return result;
	return result;
} 

void parse(char* line, struct array_2d matrix, int row) {
	char *ch = strtok(line, " ");
	int column_count = 0;
	while (ch != NULL) {
		float number = atof(ch);
		matrix.arr[row][column_count++] = number;
		ch = strtok(NULL, " ,");
	}
}

int main()
{
	FILE *fp;
	char *line = NULL;
	size_t len = 0;
	ssize_t read;
	char *file_name = "../Data/data_small/Test_151_3x7_dot_7x8_False_10000_0.txt";

	fp = fopen(file_name, "r");
	if (fp == NULL)
		exit(EXIT_FAILURE);

	int line_count = 0;
	struct array_2d matrix_a, matrix_b, matrix_c;
	while ((read = getline(&line, &len, fp)) != -1)
	{
		++line_count;

		if (line_count == 1) {
			printf("%s", line);
			continue;
		}

		if (line_count == 2)
		{
			matrix_a = get_array(line);
			printf("%d x %d\n", matrix_a.n, matrix_a.m);
			continue;
		}
		if (line_count == 3) {
			matrix_b = get_array(line);
			printf("%d x %d\n", matrix_b.n, matrix_b.m);

			matrix_c =  (struct array_2d) {.arr = createArray(matrix_a.n, matrix_b.m), .m = matrix_b.m, .n = matrix_a.n};
			continue;
		}

		if (line_count <= matrix_a.n + 3) {
			parse(line, matrix_a, line_count-4);
			continue;
		}

		if (line_count <= matrix_b.n + matrix_a.n + 3) {
			parse(line, matrix_b, line_count-matrix_a.n-4);
			continue;
		}

		if (line_count <= matrix_c.n + matrix_b.n + matrix_a.n + 3) {
			parse(line, matrix_c, line_count-matrix_b.n-matrix_a.n-4);
			continue;
		}

		// If all lines parsed, there will not show any line up here
		printf("%s", line);
	}

	fclose(fp);
	if (line)
		free(line);
	exit(EXIT_SUCCESS);

	return 0;
}
