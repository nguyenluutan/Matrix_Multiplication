#include <stdio.h>

int main() {
  FILE *fp; // declaration of file pointer
  char con[1000]; // variable to read the content
  fp = fopen("input_test_summa.txt", "r"); // opening a file mit specified name
  if (!fp) // chekcing for an error, same as 'fp == NULL'
  printf("oh, that is not possible with this file!");
  while( fgets(con, 1000, fp) != NULL) // reading file content, if not empty
  printf("%s", con);
  fclose(fp); // closing filen
  return 0;
}
