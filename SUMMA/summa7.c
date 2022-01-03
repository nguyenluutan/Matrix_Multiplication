// Compile MacBook: mpicc -openmp -g -Wall -std=c11 summa7.c -o summa7_mpi -lm
// Compile MacBook and run with host_file (more processors settings): mpirun --hostfile host_file --np 4 summa7_mpi 1024 1024 1024
// Compile Cluster: mpicc -fopenmp -g -Wall -std=c11 summa7.c -o summa7_mpi -lm

// Run: mpirun --np <number of procs> ./summa <m> <n> <k>
// <number of procs> must be perfect square
// <m>, <n> and <k> must be dividable by sqrt(<number of procs>)
// NOTE: current version of program works with square matrices only
// <m> == <n> == <k>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>
#include <unistd.h> // Fix implicit declaration of function 'getpid' is invalid in C99

// Parameter
#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

const double THRESHOLD = 0.001; // Threshold to check for accuracy of different calculations
// 1 Dimensional matrix on stack (not on heap)
double flatA[MAX_DIM];
double flatB[MAX_DIM];

// Method signatures
double** randomSquareMatrix(int dimension);
double** zeroSquareMatrix(int dimension);
void displaySquareMatrix(double** matrix, int dimension);
void convert(double** matrixA, double** matrixB, int dimension);
// Matrix multiplication methods
double sequentialMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension);


/****** Function to set seed for a random numbers generator **********/
unsigned long mix(unsigned long a, unsigned long b, unsigned long c) {
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

 /******** create a random square matrix ************/
 double** randomSquareMatrix(int dimension) {
 // Generate 2 dimensional random TYPE matrix
 	double** matrix = malloc(dimension * sizeof(double*));
 	int i,j; // Iterators
 	for(i=0; i<dimension; i++) {
 		matrix[i] = malloc(dimension * sizeof(double));
 	}
 // set Random seed
  	unsigned long seed = mix(clock(), time(NULL), getpid()); // call mix-function
  	srand(seed);

 	for(i=0; i<dimension; i++) {
 		for(j=0; j<dimension; j++) {
 			matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
 		}
 	}
 	return matrix;
}

/************ create a square matrix from zeros ************/
double** zeroSquareMatrix(int dimension) {
// Generate 2 dimensional zero double matrix
	double** matrix = malloc(dimension * sizeof(double*));
	int i, j; // Iterators
	for(i=0; i<dimension; i++) {
		matrix[i] = malloc(dimension * sizeof(double));
	}
// set Random seed --> actually this is stupid, because we just use zeros :D
 	unsigned long seed = mix(clock(), time(NULL), getpid()); // call mix-function
 	srand(seed);

	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			matrix[i][j] = 0;
		}
	}
	return matrix;
}

/************* Display the squared matrix **************/
void displaySquareMatrix(double** matrix, int dimension) {
	int i, j; // Iterators
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			printf("%f\t", matrix[i][j]);
		}
		printf("\n"); // print line-by-line
	}
}

/*********** Sequential Matrix Multiplication ***********/
// non-parallel!
// used by root processor to verify result of parallel algorithm
// C[m,k] = A[m,n] * B[n,k]
void sequentialMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension) {
// Sequentiall multiply given input matrices and return resultant matrix
	int i, j, k; // Iterators
	double tot; // in-between save
	/* Head */
	convert(matrixA, matrixB, dimension);
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			tot = 0.0;
			for(k=0; k<dimension; k++) {
				tot += flatA[i*dimension + k] * flatB[j*dimension + k];
			}
			matrixC[i][j] = tot;
		}
	}
}
// convert-function for optimized matrix multiplication with OpenMP
void convert(double** matrixA, double** matrixB, int dimension) {
	int i,j;
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			flatA[i*dimension + j] = matrixA[i][j];
			flatB[j*dimension + i] = matrixB[i][j];
		}
	}
}

/******** Local matrix addition ************/
// C = A + B
void plus_matrix(double** matrixA, double** matrixB, double** matrixC, int dimension) {
    int i, j, k, tot;
		convert(matrixA, matrixB, dimension);
    for (i=0; i<dimension; i++) {
			for (j=0; j<dimension; j++) {
				tot = 0.0;
				for(k=0; k<dimension; k++) {
					tot += flatA[i*dimension + k] + flatB[j*dimension + k];
        }
				matrixC[i][j] = tot;
    	}
		}
}

/********* SUMMA with MPI ***********/
void SUMMA(MPI_Comm comm_cart, int block_size,
    double *A_loc, double *B_loc, double *C_loc) {

    // determine my cart coords
    int coords[2];
    MPI_Cart_coords( comm_cart, myrank, 2, coords );

    MPI_Comm row_comm;
    MPI_Comm col_comm;

    int my_col = coords[0];
    int my_row = coords[1];

    int remain_dims[2];

    // create row comms for A
    remain_dims[0] = 1;
    remain_dims[1] = 0;

    MPI_Cart_sub(comm_cart, remain_dims, &row_comm);

    // create col comms for B
    remain_dims[0] = 0;
    remain_dims[1] = 1;

    MPI_Cart_sub(comm_cart, remain_dims, &col_comm);

    double** A_loc_save = (double**) calloc(block_size*block_size, sizeof(double));
    double** B_loc_save = (double**) calloc(block_size*block_size, sizeof(double));
    double** C_loc_tmp = (double**) calloc(block_size*block_size, sizeof(double));

    // each proc should save its own A_loc, B_loc
    memcpy(A_loc_save, A_loc, block_size*block_size*sizeof(double));
    memcpy(B_loc_save, B_loc, block_size*block_size*sizeof(double));

    // C_loc = 0.0
    memset(C_loc, 0, block_size*block_size*sizeof(double));

    int nblks = dimension / block_size;

    // root column (or row) should loop though nblks columns (rows).
    //
    // If processor's column coordinate equals to root, it broadcasts
    // its local portion of A within its `row_comm` communicator.
    //
    // If processor's row coordinate equals to root, it broadcasts
    // its local portion of B within its `col_comm` communicator.
    //
    // After broadcasting, call multiply_naive to multiply local portions
    // which each processor have received from others
    // and store it in partial sum `C_loc_tmp`.
    //
    // Finally, accumulate partials sums of `C_loc_tmp` to `C_loc` on each iteration
    // using `plus_matrix` function.

    int bcast_root;

		// start to take time for SUMMA
		struct timeval t0, t1;
		gettimeofday(&t0, 0);
		/* Head */
    for ( bcast_root=0; bcast_root<nblks; bcast_root++) {
        if ( my_col == bcast_root )
            memcpy( A_loc, A_loc_save, block_size*block_size*sizeof(double) );
        MPI_Bcast( A_loc, block_size*block_size, MPI_DOUBLE, bcast_root, row_comm );

        if ( my_row == bcast_root )
            memcpy( B_loc, B_loc_save, block_size*block_size*sizeof(double) );
        MPI_Bcast( B_loc, block_size*block_size MPI_DOUBLE, bcast_root, col_comm );

        sequentialMultiply( A_loc, B_loc, C_loc_tmp, block_size );

        plus_matrix( C_loc_tmp, C_loc_tmp, C_loc, block_size );
    }
		/* Tail */
		gettimeofday(&t1, 0); // end to take time for SUMMA
		double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
		// free to local matrices
    free(A_loc_save);
    free(B_loc_save);
    free(C_loc_tmp);

		return elapsed;
}

void parse_cmdline(int argc, char *argv[]) {
    if (argc != 3) {
        if (myrank == 0) {
            fprintf(stderr, "USAGE:\n"
                    "mpirun --np <number of procs> ./summa --args <dimension> <iterations>\n"
                    "<number of procs> must be perfect square\n"
                    "<dimension> must be dividable by sqrt(<number of procs>)\n"
                    "NOTE: current version of program works with square matrices only\n");
    int i;
		for (i=0; i<argc; i++) {
                printf("%s\n", argv[i]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    dimension = atoi(argv[1]);
	 	iterations = atoi(argv[2]);

    if ( !(dimension>0 && iterations>0) ) {
        if (myrank == 0) {
            fprintf(stderr, "ERROR: dimensions and iterations must be positive integers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (myrank == 0) {
        printf("dimension = %d, iterations = %d\n", dimension, iterations);
    }
}


/************** MAIN SCRIPT ***************/
int main(int argc, char *argv[]) {
// declare important parameters
		int myrank; // Rank for MPI-Communicator
		int dimensions, iterations; // matrices size & iterations
		// A[m,n], B[n,k], C[m,k] --> but all dimensions are similar !!
// parse values to variables of dimension and iterations and check them
    parse_cmdline(argc, argv);

// Generate Necessary files
// Create Sequential Multiply test log
		FILE* fp;
		fp = fopen("SUMMATest.txt", "w+");
		fclose(fp);
// Run test scripts in for-loops
		for(dimension=200; dimension<=400; dimension+=200) {
			SUMMATest(dimension, iterations);
		}
		return 0;
}


/************* Test Scripts for Multiplications ***************/
// Sequential Test Script
void SUMMATest(int dimension, int iterations) {
	FILE* fp;
	fp = fopen("SUMMATest.txt", "a+");

	// Console write
	printf("----------------------------------\n");
	printf("Test : SUMMA Multiplication        \n");
	printf("----------------------------------\n");
	printf("Dimension: %d\n", dimension);
	printf("..................................\n");

	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : SUMMA Multiplication        \n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension : %d\n", dimension);
	fprintf(fp, "..................................\n");

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	// assume for SUMMA simplicity that nprocs is perfect square
	// and allow only this nproc
	int nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	int n_proc_rows = (int)sqrt(nprocs);
	int n_proc_cols = n_proc_rows;

	fprintf(stderr, "%d cols, %d rows, required: %d == %d\n", n_proc_cols, n_proc_rows, n_proc_cols*n_proc_rows, nprocs);
	if (n_proc_cols*n_proc_rows != nprocs) {
			fprintf(stderr, "ERROR: Number of proccessors must be a perfect square!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// create 2D cartesian communicator from `nprocs` procs
	int ndims = 2;
	const int dims[2] = {n_proc_rows, n_proc_cols};
	const int periods[2] = {0, 0};
	int reorder = 0;
	MPI_Comm comm_cart;

	// Create 2D cartesian communicator using MPI_Cart_Create function
	// MPI_COMM_WORLD is initial communicator
	// We do not need periodicity in dimensions for SUMMA, so we set periods to 0
	// We also do not need to reorder ranking, so we set reorder to 0 too
	//
	// Dimensions of the new communicator is [n_proc_rows, n_proc_cols].

	MPI_Cart_create( MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart );

	// my rank in the new communicator
	int my_grid_rank;
	MPI_Comm_rank( comm_cart, &my_grid_rank );

	int my_coords[2];
	MPI_Cart_coords( comm_cart, my_grid_rank, ndims, my_coords );

	// Print my location in the 2D cartesian topology.
	printf("[MPI process %d] I am located at (%d, %d) in the initial 2D cartesian topology.\n",
	myrank, my_coords[0], my_coords[1]);

	// assume for simplicity that matrix dims are dividable by proc grid size
	// each proc determines its local block sizes
	int block_size = dimension / n_proc_rows; // n_proc_rows == n_proc_cols !!

	// each processor allocates memory for local portions of A, B and C
	double** A_loc = NULL;
	double** B_loc = NULL;
	double** C_loc = NULL;
	A_loc = (double**) calloc(block_size*block_size, sizeof(double));
	B_loc = (double**) calloc(block_size*block_size, sizeof(double));
	C_loc = (double**) calloc(block_size*block_size, sizeof(double));

	double* opmLatency = malloc(iterations * sizeof(double));
	// init matrices: fill A_loc and B_loc with random values
	A_loc = randomSquareMatrix(block_size);
	B_loc = randomSquareMatrix(block_size);

	/* double** matrixA = randomSquareMatrix(block_size);
	double** matrixB = randomSquareMatrix(block_size); */

	// Iterate and measure performance
	int i;
	for(i=0; i<iterations; i++) {
		C_loc = zeroSquareMatrix(block_size);
		opmLatency[i] = SUMMA(comm_cart, block_size, A_loc, B_loc, C_loc);
		free(C_loc);

		// Console write
		printf("%d.\t%f\n", i+1, opmLatency[i]);
		// File write
		fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
	}

	// Console write
	printf("\n");
	printf("----------------------------------\n");
	printf("Analyze Measurements              \n");
	printf("----------------------------------\n");

	// File write
	fprintf(fp, "\n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Analyze Measurements              \n");
	fprintf(fp, "----------------------------------\n");

	double sum = 0.0;
	double sumSquared = 0.0;

	// Statistical analyze
	for(i=0; i<iterations; i++) {
		sum += opmLatency[i];
		sumSquared += pow(opmLatency[i], 2.0);
	}

	double mean = sum / iterations;
	double squareMean = sumSquared / iterations;
	double standardDeviation = sqrt(squareMean - pow(mean, 2.0));

	// Console write
	printf("Mean               : %f\n", mean);
	printf("Standard Deviation : %f\n", standardDeviation);
	printf("----------------------------------\n");

	//File write
	fprintf(fp, "Mean               : %f\n", mean);
	fprintf(fp, "Standard Deviation : %f\n", standardDeviation);
	fprintf(fp, "----------------------------------\n");

	// Releasing memory
	fclose(fp);
	free(opmLatency);
	/* free(matrixA);
	free(matrixB); */

	// deallocate matrices
	free(A_loc);
	free(B_loc);
	// free(C_loc);

	MPI_Finalize();
	return 0;
}
