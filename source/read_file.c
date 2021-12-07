#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

int main() {
  // FILE *fp; // declaration of file pointer
  // char con[100000]; // variable to read the content

  // fp = fopen("./Data/data_12_07_2021_08_18/Test_1_8x4_True_10_0.txt", "r");
  // if (!fp)
  //   printf("That is not possible with this file!");
  
  // printf("Reading file");

  // while(fgets(con, 100000, fp) != NULL)
  //   printf("%s", con);

  // fclose(fp);

    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char * file_name = "../Data/data_small/Test_4_6x1_True_10_0.txt";

    fp = fopen("../Data/data_small/Test_4_6x1_True_10_0.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1) {
        // printf("Retrieved line of length %zu:\n", read);
        printf("%s", line);
    }

    fclose(fp);
    if (line)
        free(line);
    exit(EXIT_SUCCESS);

  return 0;
}
