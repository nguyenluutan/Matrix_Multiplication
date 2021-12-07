// Compile: mpicc -g -Wall -std=c99 summa4.c -o summa4_mpi -lm
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


// if CHECK_NUMERICS is defined, program will gather global matrix C
// calculated by SUMMA to root processor and compare it with
// C_naive, calculated by naive matrix multiply algorithm.
// Use for algorithm debugging only:
// very large matrices will not fit single cpu memory.
#define CHECK_NUMERICS

// tolerance for validation of matrix multiplication
#define TOL 1e-4

// Init matrix with non-random numbers: each local matrix
// will contain elements equal to processor's rank.
// Use for algorithm debugging only.
//#define DEBUG

// global matrices size
// A[m,n], B[n,k], C[m,k]
int m;
int n;
int k;

// each processor will keep its rank in `myrank`
int myrank;

void init_matrix(double *matr, const int rows, const int cols) {
    // in real life each proc calculates its portion
    // from some equations, e.g. from PDE
    // here we will use random values

#ifndef DEBUG
    srand((unsigned int) time(NULL));
#endif

    double rnd = 0.0;
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
#ifdef DEBUG
            // for debugging it is useful to init local matrix by proc's rank
            rnd = myrank;
#else
            rnd = rand() * 1.0 / RAND_MAX;
#endif
            matr[j*cols + i] = rnd;
        }
    }
}

void print_matrix(const int rows, const int cols, const double *matr) {
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            printf("%12.6f", matr[j*cols + i]);
        }
        printf("\n");
    }

}


// naive algorithm for matrix multiplication
// non-parallel!
// used by root processor to verify result of parallel algorithm
// C[m,k] = A[m,n] * B[n,k]
void matmul_naive(const int m, const int n, const int k,
        const double *A, const double *B, double *C) {

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < k; ++i) {

            C[j*k + i] = 0.0;

            for (int l = 0; l < n; ++l) {
                C[j*k + i] += A[j*n + l] * B[l*k + i];
            }

        }
    }

}

// eps = max(abs(Cnaive - Csumma)
double validate(const int m, const int n, const double *Csumma, double *Cnaive) {

    double eps = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i*n + j;
            Cnaive[idx] = fabs(Cnaive[idx] - Csumma[idx]);

            if (eps < Cnaive[idx]) { eps = Cnaive[idx]; }
        }
    }
    return eps;
}

//
// gather global matrix from all processors in a 2D proc grid
// needed for debugging and numeric validation only
// never happens in real life in production runs because global matrix does not fit any single proc memory
void gather_glob(const int mb, const int nb, const double *A_loc,
    const int m, const int n, double *A_glob) {

    double *A_tmp = NULL;

    if (myrank == 0) { A_tmp = (double *) calloc(m * n, sizeof(double)); }

    MPI_Gather(A_loc, mb*nb, MPI_DOUBLE, A_tmp, mb*nb, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // only rank 0 has something to do, others are free
    if (myrank != 0) return;

    // fix data layout
    // http://stackoverflow.com/questions/5585630/mpi-type-create-subarray-and-mpi-gather

    int nblks_m = m / mb;
    int nblks_n = n / nb;
    int idx = 0;

    for (int blk_i = 0; blk_i < nblks_m; ++blk_i) {
        for (int blk_j = 0; blk_j < nblks_n; ++blk_j) {

            // position in global matrix where current block should start
            int blk_start_row = blk_i * mb;
            int blk_start_col = blk_j * nb;

            for (int i = 0; i < mb; ++i) {
                for (int j = 0; j < nb; ++j) {
                    A_glob[(blk_start_row + i)*n + (blk_start_col + j)] = A_tmp[idx];
                    idx++;
                }
            }

        }
    }
    free(A_tmp);
}

// Local matrix addition
// C = A + B
void plus_matrix(const int m, const int n, double *A, double *B, double *C) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i*m + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

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

    // Implement main SUMMA loop here:
    // root column (or row) should loop though nblks columns (rows).
    //
    // If processor's column coordinate equals to root, it should broadcast
    // its local portion of A within its `row_comm` communicator.
    //
    // If processor's row coordinate equals to root, it should broadcast
    // its local portion of B within its `col_comm` communicator.
    //
    // After broadcasting, call multiply_naive to multiply local portions
    // which each processor have received from others
    // and store it in partial sum `C_loc_tmp`.
    //
    // Finally, accumulate partials sums of `C_loc_tmp` to `C_loc` on each iteration
    // using `plus_matrix` function.
    //
    // Pseudo code:
    // for (int bcast_root = 0; bcast_root < nblks; ++bcast_root) {
    //
    //    int root_col = bcast_root;
    //    int root_row = bcast_root;
    //
    //    // owner of A_loc[root_col,:] will broadcast its block within row comm
    //    if (my_col == root_col) {
    //        // copy A_loc_save to A_loc
    //    }
    //    // broadcast A_loc from root_col within row_comm
    //
    //    // owner of B_loc[:,root_row] will broadcast its block within col comm
    //    if (my_row == root_row) {
    //        // copy B_loc_cave to B_loc
    //    }
    //    // broadcast B_loc from root_row within col_comm
    //
    //    // multiply local blocks A_loc, B_loc using matmul_naive
    //    // and store in C_loc_tmp
    //
    //    // C_loc = C_loc + C_loc_tmp using plus_matrix
    //}
    // ====================================================

    /********** REAL CODE - Eric, 04.12.2021 ***************/
    int bcast_root;

    for ( bcast_root=0; bcast_root<nblks; ++bcast_root) {
        if ( my_col == bcast_root ) {
            memcpy( A_loc, A_loc_save, mb*nb*sizeof(double) );
        }
        MPI_Bcast( A_loc, mb*nb, MPI_DOUBLE, bcast_root, row_comm );

        if ( my_row == bcast_root ) {
            memcpy( B_loc, B_loc_save, nb*kb*sizeof(double) );
        }
        MPI_Bcast( B_loc, nb*kb, MPI_DOUBLE, bcast_root, col_comm );

        matmul_naive( nb, mb, kb, A_loc, B_loc, C_loc_tmp );

        plus_matrix( m, n, C_loc_tmp, C_loc_tmp, C_loc );
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
            for (int i = 0; i < argc; i++) {
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

/*********** OpenMP - Part, Eric 06.12.2021 ************/
void transpose(const int n, const int m, double *A, double *B) {
    int i, j;
    for(j=0; j<n; j++) {
		for(i=0; i<m; i++) {
			B[i*n + j] = A[j*m + i];
		}
	}
} /****** THIS IS THE ORIGINAL TRANSPOSE FUNCTION! *****/

void gemmT(const int n, const int m, const int k,
            double *A, double *B, double *C) {

    int i, j, l;
	double *B2;
	B2 = (double*) malloc(sizeof(double)*n*m);
    transpose(n, m, B, B2);
	for (j=0; j<n; j++) {
		for (i=0; i<m; i++) {
            C[j*k + i] = 0.0;
			for (l=0; l<k; l++) {
				C[j*k + i] += A[j*n + l] * B2[l*k + i];
			}
		}
	}
	free(B2);
}

void gemm_omp(const int n, const int m, const int k,
                double *A, double *B, double *C) {

    #pragma omp parallel
	{
		int i, j, l;
		#pragma omp for
		for (j=0; j<n; j++) {
			for (i=0; i<m; i++) {
				C[j*k + i] = 0.0;
				for (l=0; l<k; l++) {
					C[j*k + i] += A[j*n + l] * B[l*k + i];
				}
			}
		}
	}
}

void gemmT_omp(const int n, const int m, const int k,
                double *A, double *B, double *C) {

    double *B2;
	B2 = (double*) malloc(sizeof(double)*n*m);
    transpose(n, m, B, B2);
	#pragma omp parallel
	{
		int i, j, l;
		#pragma omp for
		for (j=0; j<n; j++) {
			for (i=0; i<m; i++) {
				C[j*k + i] = 0.0;
				for (l=0; l<k; l++) {
					C[j*k + i] += A[j*n + l] * B2[l*k + i];
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

    int n_proc_rows = sqrt(nprocs);
    int n_proc_cols = n_proc_rows;
    if (n_proc_cols * n_proc_rows != nprocs) {
        fprintf(stderr, "ERROR: number of proccessors must be a perfect square!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // create 2D cartesian communicator from `nprocs` procs
    int ndims = 2;
    const int dims[2] = {n_proc_rows, n_proc_cols};
    const int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm comm_cart;

    // Create 2D cartesian communicator using MPI_Cart_Create function
    // MPI_COMM_WORLD is your initial communicator
    // We do not need periodicity in dimensions for SUMMA, so we set periods to 0.
    // We also do not need to reorder ranking, so we set reorder to 0 too.
    //
    // Dimensions of the new communicator should be [n_proc_rows, n_proc_cols].
    // New communicator with Cartesian topology should be assigned to

    /******* REAL CODE - Eric, 04.12.2021 *****///////
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

#ifdef CHECK_NUMERICS
    // rank 0 allocates matrices A_glob, B_glob, C_glob, C_glob_naive for checking
    double *A_glob = NULL;
    double *B_glob = NULL;
    double *C_glob = NULL;
    double *C_glob_naive = NULL;
    if (myrank == 0) {
        A_glob = (double *) calloc(m * n, sizeof(double));
        B_glob = (double *) calloc(n * k, sizeof(double));
        C_glob = (double *) calloc(m * k, sizeof(double));
        C_glob_naive = (double *) calloc(m * k, sizeof(double));
    }
#endif

    // init matrices: fill A_loc and B_loc with random values
    // in real life A_loc and B_loc are calculated by each proc
    // from e.g. partial differential equations
    init_matrix(A_loc, mb, nb);
    init_matrix(B_loc, nb, kb);

    // gather A_glob, B_glob for further checking
#ifdef CHECK_NUMERICS
    gather_glob(mb, nb, A_loc, m, n, A_glob);
    gather_glob(nb, kb, B_loc, n, k, B_glob);
#endif

    // call SUMMA (MPI), matmul_transpose, matmul (OMP), matmul_transpose (OMP)
    // take start and end times of calculations for comparison
    double tstart_mpi, tend_mpi, diff_time_mpi;
    double tstart_transp, tend_transp, diff_time_transp;
    double tstart_omp, tend_omp, diff_time_omp;
    double tstart_omp_transp, tend_omp_transp, diff_time_omp_transpose;

    tstart_mpi = MPI_Wtime();

    // You should implement SUMMA algorithm in SUMMA function.
    // SUMMA stub function is in this file (see above).
    SUMMA(comm_cart, mb, nb, kb, A_loc, B_loc, C_loc);

    tend_mpi = MPI_Wtime();

    // Each processor will spend different time doing its
    // portion of work in SUMMA algorithm. To understand how long did
    // SUMMA execution take overall we should find time of the slowest processor.
    // We should be using MPI_Reduce function with MPI_MAX operation
    diff_time_mpi = tend_mpi - tstart_mpi;
    double max_diff_time_mpi = 0.0;

    // Determine maximum value of `etime` across all processors in MPI_COMM_WORLD
    // and save it in `max_etime` variable on root processor (rank 0).

    /********* REAL CODE - Eric, 04.12.2021 *********/
    MPI_Reduce( &diff_time_mpi, &max_diff_time_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );

    if (myrank == 0) { printf("SUMMA took %f sec\n", max_diff_time_mpi); }

    tstart_transp = MPI_Wtime();




#ifdef CHECK_NUMERICS
    // gather C_glob
    gather_glob(mb, kb, C_loc, m, k, C_glob);

    if (myrank == 0) {
        matmul_naive(m, n, k, A_glob, B_glob, C_glob_naive);

#ifdef DEBUG
        printf("C_glob_naive:\n");
        print_matrix(m, k, C_glob_naive);
        printf("C_glob:\n");
        print_matrix(m, k, C_glob);
#endif

        double eps = validate(n, k, C_glob, C_glob_naive);
        if (eps > TOL) {
            fprintf(stderr, "ERROR: eps = %f\n", eps);
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            printf("SUMMA: OK: eps = %f\n", eps);
        }
    }

    free(A_glob);
    free(B_glob);
    free(C_glob);
    free(C_glob_naive);
#endif

    // deallocate matrices
    free(A_loc);
    free(B_loc);
    free(C_loc);

    MPI_Finalize();
    return 0;
}
