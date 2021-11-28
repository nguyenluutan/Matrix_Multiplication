/********* Matrix Input...but of ANY size! ***********/
#include <stdio.h>

int main() {
  // variables
  int row, column;

  // define row-and columnsize
  printf("enter the row size: ");
  scanf("%d", &row);
  printf("enter column size: ");
  scanf("%d", &column);

  // declare array
  int arr[row][column];

  // take matrix elements as input
  printf("enter elements for %dx%d matrix:\n", row, column);

  for (int i=0; i<row; i++) {
    for (int j=0; j<column; j++) {
      printf("arr[%d][%d]: ", i, j);
      scanf("%d", &arr[i][j]);
    }
    printf("\n");
  }
  // display the matrix again with a for-loop
  printf("the %dx%d matrix elements are:\n", row, column);

  for (int i=0; i<row; i++) {
    for (int j=0; j<column; j++) {
      printf("%d ", arr[i][j]);
    }
    printf("\n"); // next row on a new line!
  }
  return 0;
}
