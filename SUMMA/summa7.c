// Compile MacBook: mpicc -g -Wall -std=c11 summa7.c -o summa7 -lm
// Compile MacBook and run with host_file (more processors settings): mpirun --hostfile host_file --np 4 summa7 256
// Compile Cluster: mpicc -g -Wall -std=c11 summa7.c -o summa7 -lm

// Run: mpirun --np <number of procs> ./summa7 <dimension>
// <number of procs> must be perfect square
// <dimension> must be dividable by sqrt(<number of procs>)
// ATTENTION: current version of program works with square matrices only

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h> // Fix implicit declaration of function 'getpid' is invalid in C99

#define MAX_VAL 10
#define MIN_VAL 1

//const double THRESHOLD = 0.001; // Threshold to check for accuracy of different calculations

// matrix sizes
// A[m,n], B[n,k], C[m,k] --> m, n, k = dimension!
int dimension;
int myrank;

/********************** METHODS *******************************/
unsigned long mix(unsigned long a, unsigned long b, unsigned long c); // for random values
void randomMatrix(double *matr, const int dimension); // for matrizes A & B
void zeroMatrix(double *matr, const int dimension); // for matrix C
void seqMult(const int dimension, const double *A, const double *B, double *C); // sequential multiplication for outer products
void addMatrix(const int dimension, double *A, double *B, double *C); // matrix addition for 'real' matrix C
void SUMMA(MPI_Comm comm_cart, const int block_size, double *A_loc, double *B_loc, double *C_loc); // SUMMA calculation function

/********************* MAIN FUNCTION ***************************/
int main(int argc, char *argv[]) {
		// declare some variables
		int nprocs, n_proc_rows, n_proc_cols;
		double tstart_summa, tend_summa, diff_time_summa // take start and end times of calculations for comparison
		double tstart_seq, tend_seq, diff_time_seq;
		// initialize MPI
    MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		// read the dimension specification from the shell script
		dimension = atoi(argv[1]);
		if ( !(dimension > 0) ) {
        if (myrank == 0) {
            fprintf(stderr, "ERROR: dimension must be positive integers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (myrank == 0) {
        printf("dimension = %d\n", dimension);
    }

		// check number of processes
		n_proc_rows = (int)sqrt(nprocs); // nprocs is a perfect square !
		n_proc_cols = n_proc_rows;	// local rows and columns for processes
		fprintf(stderr, "%d cols, %d rows, required: %d == %d\n", n_proc_cols, n_proc_rows, n_proc_cols*n_proc_rows, nprocs);
    if (n_proc_cols*n_proc_rows != nprocs) {
        fprintf(stderr, "ERROR: Number of proccessors must be a perfect square!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

		// design 2D-Cartesian Grid-System
    int nDims = 2;   // create 2D cartesian communicator from `nprocs` procs
    const int dims[2] = {n_proc_rows, n_proc_cols};   // dimensions of the new communicator is [n_proc_rows, n_proc_cols].
    const int periods[2] = {0, 0}; // set periods to 0, not needed
    int reorder = 1;
    MPI_Comm comm_cart;   // create 2D cartesian communicator
    // MPI_COMM_WORLD is initial communicator
    MPI_Cart_create( MPI_COMM_WORLD, nDims, dims, periods, reorder, &comm_cart );
    int my_grid_rank;// my rank in the new communicator
    MPI_Comm_rank( comm_cart, &my_grid_rank );
    int my_coords[2];
    MPI_Cart_coords( comm_cart, my_grid_rank, nDims, my_coords );

		// matrix dimensions are dividable by proc grid size
    // each process determines its local block sizes
    int block_size = dimension / n_proc_rows; // n_proc_rows = n_proc_cols!
		if (block_size*n_proc_rows != dimension) {
        fprintf(stderr, "ERROR: dimension must be dividable by n_proc_rows\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // each processor allocates memory for local portions of A, B, C for SUMMA
    double *A_loc = NULL;
    double *B_loc = NULL;
    double *C_loc = NULL;
    A_loc = (double *) calloc(block_size*block_size, sizeof(double));
    B_loc = (double *) calloc(block_size*block_size, sizeof(double));
    C_loc = (double *) calloc(block_size*block_size, sizeof(double));
		// memory allocation for matrices A, B, C for sequential multiplication
		/*double *A_seq = NULL;
    double *B_seq = NULL;
    double *C_seq = NULL;
    A_seq = (double *) calloc(dimension*dimension, sizeof(double));
    B_seq = (double *) calloc(dimension*dimension, sizeof(double));
    C_seq = (double *) calloc(dimension*dimension, sizeof(double));*/

    // initialize matrices A and B with random values and C with zeros for SUMMA
    randomMatrix(A_loc, block_size);
    randomMatrix(B_loc, block_size);
		zeroMatrix(C_loc, block_size);
		// initialize matrices A and B with random values and C with zeros for seqMult
		//randomMatrix(A_seq, dimension);
    //randomMatrix(B_seq, dimension);
		//zeroMatrix(C_seq, dimension);

/****************************** TESTING ***************************************/
		// call SUMMA (MPI) and sequential multiplication
		MPI_Barrier( MPI_COMM_WORLD ); // to synchronise time
		tstart_summa = MPI_Wtime(); 		// Take time of SUMMA run
		SUMMA(comm_cart, block_size, A_loc, B_loc, C_loc);
		MPI_Barrier( MPI_COMM_WORLD ); // to synchronise time again
		tend_summa = MPI_Wtime();
		diff_time_summa = tend_summa - tstart_summa;
		if (myrank == 0) { printf("SUMMA took %f sec\n", diff_time_summa); }

		// Take the time of sequential multiplication
		tstart_seq = MPI_Wtime();
		seqMult( dimension, A_loc, B_loc, C_loc );
		tend_seq = MPI_Wtime();
		diff_time_seq = tend_seq - tstart_seq;
		if (myrank == 0) { printf("Seequential-Matrix-Multiplication took %f sec\n", diff_time_seq); }

		// deallocate matrices from memory
    free(A_loc);
    free(B_loc);
    free(C_loc);
		//free(A_seq);
		//free(B_seq);
		//free(C_seq);

		// Finalize and return hopefully successful!
    MPI_Finalize();
    return 0;
}

/****************************** FUNCTIONS *************************************/
/************* set seet for random numbers generator ****************/
unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
	a=a-b;  a=a-c;  a=a^(c >> 13);
	b=b-c;  b=b-a;  b=b^(a << 8);
	c=c-a;  c=c-b;  c=c^(b >> 13);
	a=a-b;  a=a-c;  a=a^(c >> 12);
	b=b-c;  b=b-a;  b=b^(a << 16);
	c=c-a;  c=c-b;  c=c^(b >> 5);
	a=a-b;  a=a-c;  a=a^(c >> 3);
	b=b-c;  b=b-a;  b=b^(a << 10);
	c=c-a;  c=c-b;  c=c^(b >> 15);
	return c;
 }

/************ create square matrizes A & B with random values ************/
void randomMatrix(double *matr, const int dimension) {
		// set seed with mix()-function
    unsigned long seed = mix(clock(), time(NULL), getpid());
    srand(seed);
		// initialize the matrix
    int j, i;
		double rnd = 0.0;
    for (i=0; i<dimension; i++) {
        for (j=0; j<dimension; j++) {
            rnd = rand() % MAX_VAL + MIN_VAL;
            matr[i*dimension + j] = rnd;
        }
    }
}

/************ create a square matrix C with zeros ************/
void zeroMatrix(double *matr, const int dimension) {
		// set seed with mix()-function
		unsigned long seed = mix(clock(), time(NULL), getpid());
		srand(seed);
		// initialize the matrix
		int j, i;
		for (i=0; i<dimension; i++) {
				for (j=0; j<dimension; j++) {
						matr[i*dimension + j] = 0.0;
				}
		}
}

/******** non-parallel sequential algorithm for matrix multiplication *******/
void seqMult(const int dimension, const double *A, const double *B, double *C) {
		// perform ijk basic multipliction
    int i, j, k;
    for (i=0; i<dimension; i++) {
        for (j=0; j<dimension; j++) {
						C[i*dimension + j] = 0.0;
            for (k=0; k<dimension; k++) {
                C[i*dimension + j] += A[i*dimension + k] * B[k*dimension + j];
            }
        }
    }
}

/************* Local matrix addition: C = A + B ******************/
void addMatrix(const int dimension, double *A, double *B, double *C) {
		// addition of the outer products to get the corresponding value for C
		int i, j;
    for (i=0; i<dimension; i++) {
        for (j=0; j<dimension; j++) {
            C[i*dimension + j] = A[i*dimension + j] + B[i*dimension + j];
        }
    }
}

/***************** SUMMA with MPI **************************/
void SUMMA(MPI_Comm comm_cart, const int block_size, double *A_loc, double *B_loc, double *C_loc) {
    // determine my cart coords
    int coords[2];
    MPI_Cart_coords( comm_cart, myrank, 2, coords );
		// define row and column communicators
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int my_col = coords[0];
    int my_row = coords[1];

    int remainingDims[2];
    // create row comms for A
    remainingDims[0] = 1;
    remainingDims[1] = 0;
    MPI_Cart_sub(comm_cart, remainingDims, &row_comm);
    // create col comms for B
    remainingDims[0] = 0;
    remainingDims[1] = 1;
    MPI_Cart_sub(comm_cart, remainingDims, &col_comm);
		// reserve memory space for the matrices dependet on their size
    double *A_loc_save = (double *) calloc(block_size*block_size, sizeof(double));
    double *B_loc_save = (double *) calloc(block_size*block_size, sizeof(double));
    double *C_loc_tmp = (double *) calloc(block_size*block_size, sizeof(double));
    // each process saves its own A_loc, B_loc and C_loc_tmp
    memcpy(A_loc_save, A_loc, block_size*block_size*sizeof(double));
    memcpy(B_loc_save, B_loc, block_size*block_size*sizeof(double));
		memset(C_loc_tmp, 0, block_size*block_size*sizeof(double)); // only zeros in C_loc_tmp
		// determine the number of (local) blocks to for the processes
    int nBlocks = dimension / block_size;

    // root column (or row) should loop though nBlocks columns (rows).
    //
    // If processor's column coordinate equals to root, it broadcasts
    // its local portion of A within its 'row_comm' communicator.
    //
    // If processor's row coordinate equals to root, it broadcasts
    // its local portion of B within its 'col_comm' communicator.
    //
    // After broadcasting, call sequential multiplication to multiply local portions
    // which each processor have received from others
    // and stored it in partial / temporal sum 'C_loc_tmp'
    //
    // add temporal sums of 'C_loc_tmp' to 'C_loc' on each iteration

    int bcast_root;

    for ( bcast_root=0; bcast_root<nBlocks; ++bcast_root) {
        if ( my_col == bcast_root ) {
            memcpy( A_loc, A_loc_save, block_size*block_size*sizeof(double) );
				}
        MPI_Bcast( A_loc, block_size*block_size, MPI_DOUBLE, bcast_root, row_comm );

        if ( my_row == bcast_root ) {
            memcpy( B_loc, B_loc_save, block_size*block_size*sizeof(double) );
				}
        MPI_Bcast( B_loc, block_size*block_size, MPI_DOUBLE, bcast_root, col_comm );
				// sequential matrix multiplication for outer products of the grid-panel
        seqMult( block_size, A_loc, B_loc, C_loc_tmp );
				// add products together to receive actual value for specific cell of C
        addMatrix( block_size, C_loc_tmp, C_loc_tmp, C_loc );
    }
    free(A_loc_save);
    free(B_loc_save);
    free(C_loc_tmp);
}
