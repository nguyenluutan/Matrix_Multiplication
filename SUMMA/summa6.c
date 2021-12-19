// Compile MacBook: mpicc -openmp -g -Wall -std=c99 summa6.c -o summa6_mpi -lm
// Compile MacBook and run with host_file (more processors settings): 
// mpicc -openmp -g -Wall -std=c99 summa6.c -o summa6_mpi -lm && mpirun --hostfile host_file --np 4 summa6_mpi 2 2 2

// Compile Cluster: mpicc -fopenmp -g -Wall -std=c99 summa6.c -o summa6_mpi -lm
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
#include <mpi.h>
#include <omp.h>
#include <unistd.h> // Fix implicit declaration of function 'getpid' is invalid in C99

// global matrices size
// A[m,n], B[n,k], C[m,k]
int m;
int n;
int k;

int myrank;

// set seet for random numbers generator
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

void init_matrix(double *matr, const int rows, const int cols) {

    unsigned long seed = mix(clock(), time(NULL), getpid());
    srand(seed);

    double rnd = 0.0;
    int j, i;
    for (j = 0; j < rows; ++j) {
        for (i = 0; i < cols; ++i) {

            rnd = rand() * 1.0 / RAND_MAX;

            matr[j*cols + i] = rnd;
        }
    }
}

// naive algorithm for matrix multiplication
// non-parallel!
// used by root processor to verify result of parallel algorithm
// C[m,k] = A[m,n] * B[n,k]
void matmul_naive(const int m, const int n, const int k, const double *A, const double *B, double *C) {

    int j, i, l;
    for (j = 0; j < m; ++j) {
        for (i = 0; i < n; ++i) {

            C[j*k + i] = 0.0;

            for (l = 0; l < k; ++l) {
                C[j*k + i] += A[j*n + l] * B[l*k + i];
            }
        }
    }
}

// Local matrix addition
// C = A + B
void plus_matrix(const int m, const int n, double *A, double *B, double *C) {
    int j, i;
    for (j = 0; j < m; ++j) {
        for (i = 0; i < n; ++i) {
            int idx = j*m + i;

            C[idx] = A[idx] + B[idx];
        }
    }
}

/********* MPI ***********/
void SUMMA(MPI_Comm comm_cart, const int mb, const int nb, const int kb,
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

    double *A_loc_save = (double *) calloc(mb*nb, sizeof(double));
    double *B_loc_save = (double *) calloc(nb*kb, sizeof(double));
    double *C_loc_tmp = (double *) calloc(mb*kb, sizeof(double));

    // each proc should save its own A_loc, B_loc
    memcpy(A_loc_save, A_loc, mb*nb*sizeof(double));
    memcpy(B_loc_save, B_loc, nb*kb*sizeof(double));

    // C_loc = 0.0
    memset(C_loc, 0, mb*kb*sizeof(double));

    int nblks = n / nb;

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

    for ( bcast_root=0; bcast_root<nblks; ++bcast_root) {
        if ( my_col == bcast_root )
            memcpy( A_loc, A_loc_save, mb*nb*sizeof(double) );
        MPI_Bcast( A_loc, mb*nb, MPI_DOUBLE, bcast_root, row_comm );

        if ( my_row == bcast_root )
            memcpy( B_loc, B_loc_save, nb*kb*sizeof(double) );
        MPI_Bcast( B_loc, nb*kb, MPI_DOUBLE, bcast_root, col_comm );

        matmul_naive( mb, nb, kb, A_loc, B_loc, C_loc_tmp );

        plus_matrix( mb, nb, C_loc_tmp, C_loc_tmp, C_loc );
    }

    free(A_loc_save);
    free(B_loc_save);
    free(C_loc_tmp);

}

void parse_cmdline(int argc, char *argv[]) {
    if (argc != 4) {
        if (myrank == 0) {
            fprintf(stderr, "USAGE:\n"
                    "mpirun --np <number of procs> ./summa --args <m> <n> <k>\n"
                    "<number of procs> must be perfect square\n"
                    "<m>, <n> and <k> must be dividable by sqrt(<number of procs>)\n"
                    "NOTE: current version of program works with square matrices only\n"
                    "<m> == <n> == <k>\n");
            	int i;
		for (i = 0; i < argc; i++) {
                printf("%s\n", argv[i]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    if ( !(m > 0 && n > 0 && k > 0) ) {
        if (myrank == 0) {
            fprintf(stderr, "ERROR: m, n, k must be positive integers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (myrank == 0) {
        printf("m, n, k = %d, %d, %d\n", m, n, k);
    }
}

/*********** OpenMP  ************/
// Transpose Function
void transpose(const int m, const int n, const double *A, double *B) {

    int i, j;
    for(j=0; j<m; j++) {
		for(i=0; i<n; i++) {
			B[i*m + j] = A[j*n + i];
		}
	}
}
// Matrix-Multiplication with transposed matrices
void matmul_transp(const int m, const int n, const int k,
            const double *A, const double *B, double *C) {

    int i, j, l;
	double *B2;
	B2 = (double*) malloc(sizeof(double)*n*m);
    transpose(m, n, B, B2);
	for (j=0; j<m; j++) {
		for (i=0; i<n; i++) {
            C[j*k + i] = 0.0;
			for (l=0; l<k; l++) {
				C[j*k + i] += A[j*m + l] * B2[l*k + i];
			}
		}
	}
	free(B2);
}
// Matrix-Multiplication with OpenMP Functions
void matmul_omp(const int m, const int n, const int k,
                const double *A, const double *B, double *C) {

    #pragma omp parallel
	{
		int i, j, l;
		#pragma omp for
		for (j=0; j<m; j++) {
			for (i=0; i<n; i++) {
				C[j*k + i] = 0.0;
				for (l=0; l<k; l++) {
					C[j*k + i] += A[j*m + l] * B[l*k + i];
				}
			}
		}
	}
}
// Matrix-Multiplications with OpenMP Functions and transposed matrices
void matmul_omp_transp(const int m, const int n, const int k,
                const double *A, const double *B, double *C) {

    double *B2;
	B2 = (double*) malloc(sizeof(double)*n*m);
    transpose(m, n, B, B2);
	#pragma omp parallel
	{
		int i, j, l;
		#pragma omp for
		for (j=0; j<m; j++) {
			for (i=0; i<n; i++) {
				C[j*k + i] = 0.0;
				for (l=0; l<k; l++) {
					C[j*k + i] += A[j*m + l] * B2[l*k + i];
				}
			}
		}
	}
	free(B2);
}


/************** MAIN SCRIPT ***************/
int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    parse_cmdline(argc, argv);

    // assume for SUMMA simplicity that nprocs is perfect square
    // and allow only this nproc
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_proc_rows = (int)sqrt(nprocs);
    int n_proc_cols = n_proc_rows;

    fprintf(stderr, "%d cols, %d rows, required: %d == %d\n", n_proc_cols, n_proc_rows, n_proc_cols * n_proc_rows, nprocs);
    if (n_proc_cols * n_proc_rows != nprocs) {
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
    int mb = m / n_proc_rows;
    int nb = n / n_proc_cols; // == n / n_proc_rows
    int kb = k / n_proc_cols;

    if (mb * n_proc_rows != m) {
        fprintf(stderr, "ERROR: m must be dividable by n_proc_rows\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (nb * n_proc_cols != n) {
        fprintf(stderr, "ERROR: n must be dividable by n_proc_cols\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (kb * n_proc_cols != k) {
        fprintf(stderr, "ERROR: k must be dividable by n_proc_cols\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // each processor allocates memory for local portions of A, B and C
    double *A_loc = NULL;
    double *B_loc = NULL;
    double *C_loc = NULL;
    A_loc = (double *) calloc(mb * nb, sizeof(double));
    B_loc = (double *) calloc(nb * kb, sizeof(double));
    C_loc = (double *) calloc(mb * kb, sizeof(double));

    // init matrices: fill A_loc and B_loc with random values
    init_matrix(A_loc, mb, nb);
    init_matrix(B_loc, nb, kb);

    // call SUMMA (MPI), matmul_transpose, matmul (OMP), matmul_transpose (OMP)
    // take start and end times of calculations for comparison
    double tstart_mpi, tend_mpi, diff_time_mpi;
    double tstart_transp, tend_transp, diff_time_transp;
    double tstart_omp, tend_omp, diff_time_omp;
    double tstart_omp_transp, tend_omp_transp, diff_time_omp_transp;
    double tstart_naive, tend_naive, diff_time_naive;

    // Take time of SUMMA run
    tstart_mpi = MPI_Wtime();

    SUMMA(comm_cart, mb, nb, kb, A_loc, B_loc, C_loc);

    tend_mpi = MPI_Wtime();

    // Each processor will spend different time doing its
    // portion of work in SUMMA algorithm. To understand how long did
    // find out slowest processor in SUMMA by MPI_REDUCE
    diff_time_mpi = tend_mpi - tstart_mpi;
   // double max_diff_time_mpi = 0.0;

    // Determine maximum value of `etime` across all processors in MPI_COMM_WORLD
    // and save it in max_diff_time variable on root processor (rank 0).

    //MPI_Reduce( &diff_time_mpi, &max_diff_time_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );

    //if (myrank == 0) { printf("max processor-time took %f sec\n", max_diff_time_mpi); }
    if (myrank == 0) { printf("SUMMA took %f sec\n", diff_time_mpi); }

/***    tstart_naive = MPI_Wtime();

    matmul_naive( mb, nb, kb, A_loc, B_loc, C_loc );

    tend_naive = MPI_Wtime();

    diff_time_naive = tend_naive - tend_naive;
    if (myrank == 0) { printf("Naive matrix-multiplication took %f sec\n", diff_time_naive); }
****/

    // take time of transposed matrix-multilplication run
    tstart_transp = MPI_Wtime();

    matmul_transp( mb, nb, kb, A_loc, B_loc, C_loc );

    tend_transp = MPI_Wtime();

    diff_time_transp = tend_transp - tstart_transp;

    if (myrank == 0) { printf("Transposed-Matrix-Multiplication took %f sec\n", diff_time_transp); }

    // take time of omp matrix multiplication run
    tstart_omp = MPI_Wtime();

    matmul_omp( mb, nb, kb, A_loc, B_loc, C_loc );

    tend_omp = MPI_Wtime();

    diff_time_omp = tend_omp - tstart_omp;

    if (myrank == 0) { printf("OpenMP-Matrix-Multiplication took %f sec\n", diff_time_omp); }

    // take time of omp AND transposed matrix multiplication
    tstart_omp_transp = MPI_Wtime();

    matmul_omp_transp( mb, nb, kb, A_loc, B_loc, C_loc );

    tend_omp_transp = MPI_Wtime();

    diff_time_omp_transp = tend_omp_transp - tstart_omp_transp;

    if (myrank == 0) { printf("OpenMP-transposed-Matrix-Multiplication took %f sec\n", diff_time_omp_transp); }

    // deallocate matrices
    free(A_loc);
    free(B_loc);
    free(C_loc);

    MPI_Finalize();
    return 0;
}
