#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <string.h>

#define min(a, b) ((a < b) ? a : b)
#define SZ 4000     //Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.


/***   Allocate space for a two-dimensional array **/
double **alloc_2d_double(int n_rows, int n_cols) {
    int i;
    double **array;
    array = (double **)malloc(n_rows * sizeof (double *));
    array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
    for (i=1; i<n_rows; i++){
            array[i] = array[0] + i * n_cols;
    }
    return array;
}


/**
*   Initialize arrays A and B with random numbers, and array C with zeros.
*   Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz){
    int i, j;
    double value;
    // Set random values...technically it is already random and this is redundant
    for (i=0; i<blck_sz; i++){
        for (j=0; j<blck_sz; j++){
            lA[i][j] = (double)rand() / (double)RAND_MAX;
            lB[i][j] = (double)rand() / (double)RAND_MAX;
            lC[i][j] = 0.0;
        }
    }
}

void multiplymatrix(double **a, double **b, double **c, int m, int k, int n){
  // Multiply two matrices a(of size mxk) and b(of size kxn), and add the result to c(of size mxn)
  int im, ik, in;
  for (im = 0; im < m; im++) {
    for (in = 0; in < n; in++) {
      for (ik = 0; ik < k; ik++) {
        *c[im + in * m] += *a[im + ik * m] * *b[ik + in * k];
      }
    }
  }
}

/***    Perform the SUMMA matrix multiplication. */
void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
                        double **my_B, double **my_C){

    //Add your implementation of SUMMA algorithm

    // Create row communicators
    MPI_Comm row_comm;
    int row_color = my_rank / proc_grid_sz;
    printf("Inside matmul");
    MPI_Comm_split(MPI_COMM_WORLD, row_color, my_rank, &row_comm);

    // Create column communicators
    MPI_Comm col_comm;
    int col_color = my_rank % proc_grid_sz + proc_grid_sz;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, my_rank, &col_comm);

    double ** Atemp, ** Btemp;
    Atemp = alloc_2d_double(block_sz, block_sz);
    Btemp = alloc_2d_double(block_sz, block_sz);

    // Matrix size of smaller matrix
    int size = (SZ / block_sz) * (SZ / block_sz);

    // DO SUMMA COMPUTATION
    for (int k = 0; k < block_sz; ++k) {
        if (col_color == k + block_sz)
            memcpy(Atemp, my_A, size);

        if (row_color == k)
            memcpy(Btemp, my_B, size);

        MPI_Bcast(Atemp, size, MPI_DOUBLE, k, row_comm);
        MPI_Bcast(Btemp, size, MPI_DOUBLE, k, col_comm);

        multiplymatrix(Atemp, Btemp, my_C, SZ/block_sz, SZ/block_sz, SZ/block_sz);
    }

}

int main(int argc, char *argv[]) {
    int rank, num_proc;                         //process rank and total number of processes
    double start_time, end_time, total_time;    // for timing
    int block_sz;                               // Block size length for each processor to handle
    int proc_grid_sz;                           // 'q' from the slides

    srand(time(NULL));                          // Seed random numbers

    /* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/

    MPI_Status status;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    /* Assign values to 1) proc_grid_sz and 2) block_sz */
    proc_grid_sz = (int)sqrt((double)num_proc);

    block_sz = (SZ / proc_grid_sz) * (SZ / proc_grid_sz);

    if (SZ % proc_grid_sz != 0){
        printf("Matrix size cannot be evenly split amongst resources!\n");
        printf("Quitting....\n");
        exit(-1);
    }
    printf("rank, num_proc, block_size, proc_grid_size: %d, %d, %d, %d \n", rank, num_proc, block_sz, proc_grid_sz);
    // Create the local matrices on each process
    double **A, **B, **C;
    A = alloc_2d_double(SZ, SZ);
    B = alloc_2d_double(SZ, SZ);
    C = alloc_2d_double(SZ, SZ);
    initialize(A, B, C, SZ);

    // Use SUMMA algorithm to calculate product C
    matmul(rank, proc_grid_sz, block_sz, A, B, C);

    free(A);
    free(B);
    free(C);

    MPI_Finalize();

    return 0;
}
