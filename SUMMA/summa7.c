// Compile MacBook: mpicc -g -Wall -std=c99 summa7.c -o summa7 -lm
// Run MacBook with host_file (more processors settings): mpirun --hostfile host_file --np <number of procs from 4 to 20> summa7 <dimension>
// Compile Cluster: mpicc -g -Wall -std=c99 summa7.c -o summa7 -lm
// Run Cluster: mpirun --np <number of procs> summa7 <dimension>
// <dimension> must be dividable by sqrt(<number of procs>)

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memset()
#include <math.h> // for sqrt()
#include <time.h> // for timer and random number generator
#include <mpi.h> // for MPI
#include <unistd.h> // for implicit declaration of getpid() in C99

#define MAX_VAL 10
#define MIN_VAL 1

//const double THRESHOLD = 0.001; // Threshold to check for accuracy of different calculations

// matrix sizes
// A[m,n] x B[n,k] = C[m,k] --> m, n, k = dimension!
int dimension;
int myrank;

/********************** METHODS *******************************/
unsigned long mix(unsigned long a, unsigned long b, unsigned long c); // for random values
void randomMatrix(double* matr, const int dimension); // for matrizes A & B
void zeroMatrix(double* matr, const int dimension); // for matrix C
void seqMult(const int dimension, double* A, double* B, double* C); // sequential multiplication for outer products
void addMatrix(const int dimension, double* A, double* B, double* C); // matrix addition for 'real' matrix C
void SUMMA(MPI_Comm comm_cart, const int block_size, double* A_loc, double* B_loc, double* C_loc); // SUMMA calculation function

/********************* MAIN FUNCTION ***************************/
int main(int argc, char *argv[]) {
		// declare some variables
		int nprocs, n_proc_rows, n_proc_cols;
		double tstart_summa, tend_summa, diff_time_summa; // take start and end times of calculations for comparison
		double mintime=0.0, maxtime=0.0, avgtime=0.0; // for statistical analysis of processor times

		// initialize MPI
    MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		// read the dimension specification from the shell script
		dimension = atoi(argv[1]);
		if ( dimension <= 0 ) {
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
		//fprintf(stderr, "%d cols, %d rows, required: %d == %d\n", n_proc_cols, n_proc_rows, n_proc_cols*n_proc_rows, nprocs);
    if (n_proc_cols*n_proc_rows != nprocs) {
        fprintf(stderr, "ERROR: Number of proccessors must be a perfect square!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

		// design 2D-Cartesian Grid-System
    int nDims = 2;   // create 2D cartesian communicator from `nprocs` procs
    int dims[2] = {n_proc_rows, n_proc_cols};   // dimensions of the new communicator
		MPI_Dims_create( nprocs, nDims, dims ); // create a division of processors in the cartesian grid
		int periods[2] = {0, 0}; // not needed
    int reorder = 1; // reordering/reallocation of ranks allowed
    MPI_Comm comm_cart;   // create 2D cartesian communicator
		// create cartesian system
    MPI_Cart_create( MPI_COMM_WORLD, nDims, dims, periods, reorder, &comm_cart ); // MPI_COMM_WORLD is initial communicator
    int my_grid_rank;// "my" rank in the new communicator
    MPI_Comm_rank( comm_cart, &my_grid_rank );
    int my_coords[2]; // "my" coords in grid / coordinate system
    MPI_Cart_coords( comm_cart, my_grid_rank, nDims, my_coords ); // to find coordinates of the processors given the rank in the grid system

		// matrix dimensions are dividable by proc grid size
    // each process determines its local block sizes
    int block_size = dimension / n_proc_rows; // n_proc_rows = n_proc_cols!
		if (block_size*n_proc_rows != dimension) {
        fprintf(stderr, "ERROR: dimension must be dividable by n_proc_rows\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // each processor allocates memory for local portions of A, B, C for SUMMA
    double* A_loc = NULL;
    double* B_loc = NULL;
    double* C_loc = NULL;
    A_loc = (double *) calloc(block_size*block_size, sizeof(double));
    B_loc = (double *) calloc(block_size*block_size, sizeof(double));
    C_loc = (double *) calloc(block_size*block_size, sizeof(double));

    // initialize matrices A and B with random values and C with zeros for SUMMA
    randomMatrix(A_loc, block_size);
    randomMatrix(B_loc, block_size);
		zeroMatrix(C_loc, block_size);

/****************************** TESTING ***************************************/
		// prepare file to write
		FILE* fp;
		fp = fopen("SummaTest.txt", "a+");

		// call SUMMA (MPI) and sequential multiplication
		MPI_Barrier( MPI_COMM_WORLD ); // to synchronise time
		tstart_summa = MPI_Wtime(); 		// Take time of SUMMA run
		SUMMA(comm_cart, block_size, A_loc, B_loc, C_loc);
		MPI_Barrier( MPI_COMM_WORLD ); // to synchronise time again
		tend_summa = MPI_Wtime();
		diff_time_summa = tend_summa - tstart_summa;

		// check for processor times and write into file
		MPI_Reduce(&diff_time_summa, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // max time of processor
		MPI_Reduce(&diff_time_summa, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); // min time of processor
		MPI_Reduce(&diff_time_summa, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // sum of all processor time to calculate average processor time
		// calculate average time and print results into file
		if (myrank == 0) {
			avgtime = avgtime / nprocs;
			// File write
			fprintf(fp, "----------------------------------\n");
			fprintf(fp, "Test : SUMMA Multiply        \n");
			fprintf(fp, "----------------------------------\n");
			fprintf(fp, "Dimension : %d\n", dimension);
			fprintf(fp, "..................................\n");
			printf("max processor took for SUMMA %f sec\n", maxtime);
			fprintf(fp, "minimum processor time: %f\n", mintime);
			fprintf(fp, "maximum processor time: %f\n", maxtime);
			fprintf(fp, "average processor time: %f\n", avgtime);
		}

		// deallocate matrices from memory
		fclose(fp);
    free(A_loc);
    free(B_loc);
    free(C_loc);

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
void randomMatrix(double* matrix, const int dimension) {
		// set seed with mix()-function
    unsigned long seed = mix(clock(), time(NULL), getpid()); // special header for getpid with c99 needed!
    srand(seed);
		// initialize the matrix
    int j, i;
		double rnd = 0.0;
    for (i=0; i<dimension; i++) {
        for (j=0; j<dimension; j++) {
            rnd = rand() % MAX_VAL + MIN_VAL; // MAX_VAL & MIN_VAL defined at the beginning of the script!
            matrix[i*dimension + j] = rnd; // for faster calculation better than the array version matr[i][j] !!
        }
    }
}

/************ create a square matrix C with zeros ************/
void zeroMatrix(double* matrix, const int dimension) {
		// initialize the matrix
		int j, i;
		for (i=0; i<dimension; i++) {
				for (j=0; j<dimension; j++) {
						matrix[i*dimension + j] = 0.0; // similar to upper function, just set C to 0
				}
		}
}

/******** non-parallel sequential algorithm for matrix multiplication *******/
void seqMult(const int dimension, double* A, double* B, double* C) {
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
void addMatrix(const int dimension, double* A, double* B, double* C) {
		// addition of the outer products to get the corresponding value for C
		int i, j; // Iterators
    for (i=0; i<dimension; i++) {
        for (j=0; j<dimension; j++) {
            C[i*dimension + j] = A[i*dimension + j] + B[i*dimension + j];
        }
    }
}

/***************** SUMMA with MPI **************************/
void SUMMA(MPI_Comm comm_cart, const int block_size, double* A_loc, double* B_loc, double* C_loc) {
    // declare coordinates
    int coords[2];
    MPI_Cart_coords( comm_cart, myrank, 2, coords ); // determine my cart coords
		// define row and column communicators
    MPI_Comm row_comm;
    MPI_Comm col_comm;
		// find processors position by coordinates in the grid
    int my_col = coords[0];
    int my_row = coords[1];

		// create row communicators for A for the broadcasting
    int remainingDims[2];
    remainingDims[0] = 1;
    remainingDims[1] = 0;
		// Partitioning of the original cartesian grid-system into a sub-grid-system for the (new) row communicator of A
    MPI_Cart_sub(comm_cart, remainingDims, &row_comm);

    // create col communicator for B
    remainingDims[0] = 0;
    remainingDims[1] = 1;
    MPI_Cart_sub(comm_cart, remainingDims, &col_comm); // same as for A, but with column communicator for the broadcasting

		// reserve memory space for the matrices dependent on their size
    double *A_loc_save = (double *) calloc(block_size*block_size, sizeof(double));
    double *B_loc_save = (double *) calloc(block_size*block_size, sizeof(double));
    double *C_loc_tmp = (double *) calloc(block_size*block_size, sizeof(double));
    // each process saves its own A_loc, B_loc and C_loc_tmp
    memcpy(A_loc_save, A_loc, block_size*block_size*sizeof(double));
    memcpy(B_loc_save, B_loc, block_size*block_size*sizeof(double));
		memset(C_loc_tmp, 0, block_size*block_size*sizeof(double)); // only zeros in C_loc_tmp

		// START BROADCASTING
    int nBlocks = dimension / block_size; // determine the number of (local) blocks to for the processes
    int bcast_root; // declare loop-index for broadcasting
		// root column / row loops through all columns/rows of the given number of blocks
    for ( bcast_root=0; bcast_root<nBlocks; ++bcast_root) { // ++bcast_root, because we to send the values also to the sending row or column itself!

				if ( my_col == bcast_root ) // if coordinates of the columns of processor equals loop-index ...
            memcpy( A_loc, A_loc_save, block_size*block_size*sizeof(double) ); // copy of local A from "helper" A ...
        MPI_Bcast( A_loc, block_size*block_size, MPI_DOUBLE, bcast_root, row_comm ); // broadcast this column of A to the (local) row communicator

        if ( my_row == bcast_root ) // if coordinates of the rows of processor equals loop-index ...
            memcpy( B_loc, B_loc_save, block_size*block_size*sizeof(double) ); // copy of local B from "helper" B ...
        MPI_Bcast( B_loc, block_size*block_size, MPI_DOUBLE, bcast_root, col_comm ); // broadcast this row of B to the (local) column communicator

				// sequential matrix multiplication for outer products of the grid-panel of each processors received and broadcasted data
        seqMult( block_size, A_loc, B_loc, C_loc_tmp );
				// add temporal product of temporal C togeter with current C_loc to iterate column-per-column & row-per-row to final values of C
        addMatrix( block_size, C_loc, C_loc_tmp, C_loc );
    }
		// deallocate memory of helper variables
    free(A_loc_save);
    free(B_loc_save);
    free(C_loc_tmp);
}
