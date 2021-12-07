// https://stackoverflow.com/questions/3501338/c-read-file-line-by-line
// https://www.geeksforgeeks.org/regular-expressions-in-c/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <regex.h>

int print_result(int value);
int try_regex(char* first_line);

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

	bool first_line = true;
	while ((read = getline(&line, &len, fp)) != -1)
	{
		if (first_line)
		{
			printf("Process first line here\n");
			try_regex(line);
			first_line = false;
			continue;
		}

		printf("%s", line);
	}

	fclose(fp);
	if (line)
		free(line);
	exit(EXIT_SUCCESS);

	return 0;
}

int try_regex(char* first_line)
{
	// Variable to create regex
	regex_t regex;

	// Variable to store the return
	// value after creation of regex
	int value;

	// Function call to create regex
	value = regcomp(&regex, "[:word:]", 0);

	// If compilation is successful
	if (value == 0)
		printf("RegEx compiled successfully.");
	else
		printf("Compilation error.");

	value = regexec(&regex, first_line, 0, NULL, 0);
	print_result(value);
	return 0;
}

// Function to print the result
int print_result(int value)
{
	// If pattern found
	if (value == 0)
	{
		printf("Pattern found.\n");
		return 0;
	}

	// If pattern not found
	if (value == REG_NOMATCH)
	{
		printf("Pattern not found.\n");
		return 0;
	}

	// If error occured during Pattern
	// matching
	printf("An error occured.\n");
	return 0;
}