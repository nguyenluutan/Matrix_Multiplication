/********* BASIC MATRIX CALCULATIONS ***********/
#include <stdio.h>
#include <stdlib.h>

// add 2 matrices of dimension 3x3
void add(int m[3][3], int n[3][3], int sum[3][3]) {
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      sum[i][j] = m[i][j] + n[i][j];
    }
  }
}

// substract 2 matrices of dimension 3x3
void substract(int m[3][3], int n[3][3], int difference[3][3]) {
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      difference[i][j] = m[i][j] - n[i][j];
    }
  }
}

// multiply 2 matrices of dimension 3x3
void multiply(int m[3][3], int n[3][3], int result[3][3]) {
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      result[i][j] = 0; // assign value '0' to result matrix
      // calculate the product
      for (int k=0; k<3; k++) {
        result[i][j] += m[i][k] * n[k][j];
      }
    }
  }
}

// transpose of a matrix
void transpose(int matrix[3][3], int trans[3][3]) {
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      trans[i][j] = matrix[j][i];
    }
  }
}

// function to display 3x3 matrix
void display(int matrix[3][3]) {
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {
      printf("%d\t",matrix[i][j]);
    }
    printf("\n"); // next row on a new line
  }
}

// main function
int main() {
  // matrix
  int a[][3] = { {5,6,7}, {8,9,10}, {3,1,2} };
  int b[][3] = { {1,2,3}, {4,5,6}, {7,8,9} };
  int c[3][3];

  // display both matrices
  printf("first matrix:\n");
  display(a);
  printf("second matrix:\n");
  display(b);

  // variable to take choice
  int choice;

  // menu-driven
  do {
    // menu for choosing an operation
    printf("\nchoose the matric operation, \n");
    printf("----------------------------\n");
    printf("1. Addition\n");
    printf("2. Subtraction\n");
    printf("3. Multiplication\n");
    printf("4. Transpose\n");
    printf("5. Exit\n");
    printf("----------------------------\n");
    printf("Enter your choice: ");
    scanf("%d", &choice);

    switch (choice) {
      case 1:
        add(a, b, c);
        printf("sum of matrix: \n");
        display(c);
        break;
      case 2:
        substract(a, b, c);
        printf("Subtraction of matrix: \n");
        display(c);
        break;
      case 3:
        multiply(a, b, c);
        printf("Multiplication of matrix: \n");
        display(c);
        break;
      case 4:
        printf("Transpose of the first matrix: \n");
        transpose(a, c);
        display(c);
        printf("Transpose of the second matrix: \n");
        transpose(b, c);
        display(c);
        break;
      case 5:
        printf("thank you. \n");
        exit(0);
      default:
        printf("invalid input. \n");
        printf("please enter the correct input. \n");
    }
  }
  while(1);
  return 0;
}
