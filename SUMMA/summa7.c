// Compile MacBook: mpicc -g -Wall -std=c11 summa7.c -o summa7_mpi
// Compile MacBook and run with host_file (more processors settings): mpirun --hostfile host_file --np 4 summa7_mpi
// Compile Cluster: mpicc -g -Wall -std=c11 summa7.c -o summa7_mpi
// ATTENTION: CURRENT VERSION WORKS ONLY WITH SQUARED MATRICES !

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <unistd.h> // Fix implicit declaration of function 'getpid' is invalid in C99

// Parameter
#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

const double THRESHOLD = 0.001; // Threshold to check for accuracy of different calculations
const int ITERATIONS = 10; // Number of Iterations to run through while testing

// 1 Dimensional matrix on stack (not on heap)
double flatA[MAX_DIM];
double flatB[MAX_DIM];

// Method signatures
double** randomSquareMatrix(int dimension);
double** zeroSquareMatrix(int dimension);
void displaySquareMatrix(double** matrix, int dimension);
void convert(double** matrixA, double** matrixB, int dimension);
// Matrix multiplication methods
void sequentialMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension);
// Matrix test methods
double SUMMA(MPI_Comm comm_cart, int block_size, int dimension, int myrank,
	double** A_loc, double** B_loc, double** C_loc);


	/************** MAIN SCRIPT ***************/
	int main(int argc, char *argv[]) {
		/********	 declare important parameters **********/
		int dimension; // size of the matrices; rows x cols
		int myrank; // Rank for MPI-Communicator
		int block_size; // size of each block for each processor of MPI
		int nprocs; // nprocs MUST be perfect square !
		int n_proc_rows, n_proc_cols; // rows and columns of the cartesian grid-system

		// Create SUMMA Multiply test log
		FILE* fp;
		fp = fopen("SUMMATest.txt", "w+");
		fclose(fp);

		/* HERE FOR LOOP OVER MATRIX DIMENSION SIZES !!! */
		for(dimension=64; dimension<=128; dimension=dimension*2) {
			// Initialize MPI
			MPI_Init(&argc, &argv);
			MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // process rank
			MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // number of processes

			n_proc_rows = (int)sqrt((int)nprocs); // PERFECT SQUARE !
			n_proc_cols = n_proc_rows; // assign the same value
			fprintf(stderr, "%d cols, %d rows, required: %d == %d\n",
			n_proc_cols, n_proc_rows, n_proc_cols*n_proc_rows, nprocs);
			if (n_proc_cols*n_proc_rows != nprocs) {
				fprintf(stderr, "ERROR: Number of proccessors must be a perfect square!\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			if (myrank == 0) {
				// open the Text-file to write the results
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
			}
			// each proc determines its local block sizes
			block_size = dimension / n_proc_rows; // n_proc_rows == n_proc_cols !!
			// create 2D cartesian communicator from `nprocs` procs
			int ndims = 2;
			const int dims[2] = {n_proc_rows, n_proc_cols};
			const int periods[2] = {0, 0}; // We do not need periodicity in dimensions for SUMMA, so periods are 0 !
			int reorder = 0; // We do not need to reorder ranking, so reorder is 0 !
			MPI_Comm comm_cart; // local Communicator comm_cart
			// Create 2D cartesian communicator using MPI_Cart_Create function
			// We do not need periodicity in dimensions for SUMMA, so periods are 0 !
			// Dimensions of the new communicator is [n_proc_rows, n_proc_cols].
			MPI_Cart_create( MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart );
			// my rank in the new communicator
			int my_grid_rank;
			MPI_Comm_rank( comm_cart, &my_grid_rank );
			// my local coordinates in the grid system
			int my_coords[2];
			MPI_Cart_coords( comm_cart, my_grid_rank, ndims, my_coords );
			// Print my location in the 2D cartesian topology.
			printf("[MPI process %d] I am located at (%d, %d) in the initial 2D cartesian topology.\n",
			myrank, my_coords[0], my_coords[1]);

			// each processor allocates memory for local portions of A, B and C
			double** A_loc = NULL;
			double** B_loc = NULL;
			double** C_loc = NULL;
			A_loc = (double**) calloc(block_size*block_size, sizeof(double));
			B_loc = (double**) calloc(block_size*block_size, sizeof(double));
			C_loc = (double**) calloc(block_size*block_size, sizeof(double));
			// init matrices: fill A_loc and B_loc with random values and C_loc with zeros
			A_loc = randomSquareMatrix(block_size);
			B_loc = randomSquareMatrix(block_size);
			C_loc = zeroSquareMatrix(block_size);
			// define array to save measured calculation times of the SUMMA
			double* mpiLatency = malloc(ITERATIONS * sizeof(double));

			// Iterate and measure performance
			int i;
			for(i=0; i<ITERATIONS; i++) {
				mpiLatency[i] = SUMMA(comm_cart, block_size, dimension, myrank, A_loc, B_loc, C_loc);
				free(C_loc);

				// Console write
				printf("%d.\t%f\n", i+1, mpiLatency[i]);
				// File write
				fprintf(fp, "%d.\t%f\n", i+1, mpiLatency[i]);
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
			for(i=0; i<ITERATIONS; i++) {
				sum += mpiLatency[i];
				sumSquared += pow(mpiLatency[i], 2.0);
			}

			double mean = sum / ITERATIONS;
			double squareMean = sumSquared / ITERATIONS;
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
			free(mpiLatency);
			// deallocate matrices
			free(A_loc);
			free(B_loc);

			MPI_Finalize();
		}
		return 0;
	}

	/********** REST OF THE FUNCTIONS ***************/
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
	// convert-function for optimized sequential multiplication in 1D rather than 2D!
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
	void plus_matrix(double** matrixA, double** matrixB, double** matrixC, int dimension) {
		int i, j, k, tot;
		convert(matrixA, matrixB, dimension);
		for (i=0; i<dimension; i++) {
			for (j=0; j<dimension; j++) {
				tot = 0.0;
				for(k=0; k<dimension; k++) {
					tot += flatA[i*dimension + k] + flatB[j*dimension + k];
				}
				matrixC[i][j] = tot; // C = A + B
			}
		}
	}

	/********* SUMMA with MPI ***********/
	double SUMMA(MPI_Comm comm_cart, int block_size, int dimension, int myrank, double** A_loc, double** B_loc, double** C_loc) {
		// number of local blocks of given block_size
		int nBlocks;
		// number of blocks defined by size of matrices and local sizes of each process
		nBlocks = dimension / block_size;
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
		// initalize in between (local) save matrices
		double** A_loc_save = (double**) calloc(block_size*block_size, sizeof(double));
		double** B_loc_save = (double**) calloc(block_size*block_size, sizeof(double));
		double** C_loc_tmp = (double**) calloc(block_size*block_size, sizeof(double));
		// each proc should save its own A_loc, B_loc
		memcpy(A_loc_save, A_loc, block_size*block_size*sizeof(double));
		memcpy(B_loc_save, B_loc, block_size*block_size*sizeof(double));
		memset(C_loc, 0, block_size*block_size*sizeof(double));

		// root column (or row) should loop though nBlocks columns (rows).
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
		for ( bcast_root=0; bcast_root<nBlocks; ++bcast_root) {
			if ( my_col == bcast_root )
			memcpy( A_loc, A_loc_save, block_size*block_size*sizeof(double) );
			MPI_Bcast( A_loc, block_size*block_size, MPI_DOUBLE, bcast_root, row_comm );

			if ( my_row == bcast_root )
			memcpy( B_loc, B_loc_save, block_size*block_size*sizeof(double) );
			MPI_Bcast( B_loc, block_size*block_size, MPI_DOUBLE, bcast_root, col_comm );

			sequentialMultiply( A_loc, B_loc, C_loc_tmp, block_size );

			plus_matrix( C_loc_tmp, C_loc_tmp, C_loc, block_size );
		}
		/* Tail */
		gettimeofday(&t1, 0); // end to take time for SUMMA
		double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f; // define elapsed-time variable
		// free to local matrices
		free(A_loc_save);
		free(B_loc_save);
		free(C_loc_tmp);

		return elapsed; // return elapsed calculation time to Latency-Variable !
	}

	/* --> NOT NEEDED AT THE MOMENT, BECAUSE DIMENSION IN FOR-LOOP AND ITERATIONS AS CONSTANT DEFINED!
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
fprintf(stderr, "ERROR: dimension & iterations must be positive integers\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}
}
if (myrank == 0) {
printf("dimension = %d, iterations = %d\n", dimension, iterations);
}
}
*/
