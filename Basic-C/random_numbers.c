#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

   int row;
   printf( "Enter the row-size of array: " );
   scanf( "%d", &row );

   int column;
   printf( "Enter the column-size of array: " );
   scanf( "%d", &column );

   float randArray[row][column];
   int i, j;

   srand(time(0)); // set seed for random number generator!

   for ( i=0; i<row; i++) {
     for (j=0; j<column; j++) {
       randArray[i][j] = ( (float)rand() / (float)RAND_MAX ) * (float)(1.0); // numbers between 0 and 1
     }
   }

   printf("Elements of the %dx%d array\n", row, column);

   for( i=0; i<row; i++ ) {
     for( j=0; j<column; j++ ) {
        printf( "%.2f\t", randArray[i][j] );
     }
     printf( "\n" );
   }

   return 0;
}
