/********* SUMMMA ALGORITHM *************/

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>


# define min(a, b) ((a < b) ? a : b)
# define SZ 2 // dimension size of matrices !


/***   Allocate space for a two-dimensional array **/
double **alloc_2d_double(int n_rows, int n_cols) {
    int i; // int counter declared
    double **array; // double array-pointer declared
    array = (double **)malloc(n_rows * sizeof (double *)); // memory allocation for the array with given dimensions times type-size
    array[0] = (double *) malloc(n_rows * n_cols * sizeof(double)); // memory allocation for adress of 1. element in 1. row
    for (i=1; i<n_rows; i++){
            array[i] = array[0] + i * n_cols;  // iterates over rows of the array-pointer and allocates memory to the adresses
    }
    return array;
}


/**
*   Initialize arrays A and B with random numbers, and array C with zeros.
*   Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz) { // 3 matrices and the Size of Blocks for the separation
    int i, j;
    double value;
    // Set random values...technically it is already random and this is redundant
    for (i=0; i<blck_sz; i++){
        for (j=0; j<blck_sz; j++){
            lA[i][j] = (double)rand() / (double)RAND_MAX; // here in double, but maybe better in FLOAT ???
            lB[i][j] = (double)rand() / (double)RAND_MAX; // - '' -
            lC[i][j] = 0.0; // initialize C only with Zeros
        }
    }
}

void multiplymatrix(double **a, double **b, double **c, int m, int k, int n) { // inputs again just the adresses
  // Multiply two matrices a(of size mxk) and b(of size kxn), and add the result to c(of size mxn)
  int im, ik, in; // counter for dimension sizes of the matrices
  for (im = 0; im < m; im++) {
    for (in = 0; in < n; in++) {
      for (ik = 0; ik < k; ik++) {
        *c[im + in * m] += *a[im + ik * m] * *b[ik + in * k]; // c += ab, but with pointers
      }
    }
  }
}

/***    Perform the SUMMA matrix multiplication. */
void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A, double **my_B, double **my_C) {

    // Create row communicators
    MPI_Comm row_comm; // communicator just for all rows
    int row_color = my_rank / proc_grid_sz; // ... for row-wise local communication in the smaller grids
    printf("Inside matmul");
    MPI_Comm_split(MPI_COMM_WORLD, row_color, my_rank, &row_comm); // splitting of the row-communicator in the smaller grids

    // Create column communicators
    MPI_Comm col_comm; // communicator just for all columns
    int col_color = my_rank % proc_grid_sz + proc_grid_sz; // for column-wise local communication in the smaller grids
    MPI_Comm_split(MPI_COMM_WORLD, col_color, my_rank, &col_comm); // splitting the column-comuunicator into smaller grids

    double **Atemp, **Btemp; // local or temporal arrays for A and B (again: just the pointers) declared and...
    Atemp = alloc_2d_double(block_sz, block_sz); // via the predefined function created
    Btemp = alloc_2d_double(block_sz, block_sz);

    // Matrix size of smaller matrix
    int size = (SZ / block_sz) * (SZ / block_sz); // local matrix size out of grids calculated

    // DO SUMMA COMPUTATION
    for (int k = 0; k < block_sz; ++k) { // loop until size of the local blocks
        if (col_color == k + block_sz) // WHY THIS IF-CLAUSE?
            memcpy(Atemp, my_A, size); // Copy of characters of the size of "size" from my_A to Atemp, but again: pointers

        if (row_color == k) // WHY THIS OTHER IF-CLAUSE?
            memcpy(Btemp, my_B, size); // Copying to Btemp...

        MPI_Barrier(MPI_COMM_WORLD); // dont sure if this MPI_Barrier Commands are needed?!

        MPI_Bcast(Atemp, size, MPI_DOUBLE, k, row_comm); // SOMETHIN IN THE MPI_BROADCAST IS WRONG!

        MPI_Barrier(MPI_COMM_WORLD); // dont sure if this MPI_Barrier Commands are needed?!

        MPI_Bcast(Btemp, size, MPI_DOUBLE, k, col_comm);

        MPI_Barrier(MPI_COMM_WORLD); // dont sure if this MPI_Barrier Commands are needed?!

        multiplymatrix(Atemp, Btemp, my_C, SZ/block_sz, SZ/block_sz, SZ/block_sz); // m, k and n all with same size
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

    if (SZ % proc_grid_sz != 0) {
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
