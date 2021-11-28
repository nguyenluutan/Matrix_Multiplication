/******* Matrix Input *********/
#include <stdio.h>
int main() {
  // declare and initialize an array
  int arr[2][2] = { {50,60}, {70,80} };

  // display the 2x2 matrix with a for-loop
  printf("the matrix elements are:\n");

  // outer loop for rows
  for (int i=0; i<2; i++) {
    // inner loop for columns
    for (int j=0; j<2; j++) {
      // print it out
      printf("%d ", arr[i][j]);
    }
    printf("\n"); // second row on a new line!
  }
  return 0;
}
