/******************************************************************************************
*
*	Filename:	summa.c
*	Purpose:	A paritally implemented program for MSCS6060 HW. Students will complete
*			the program by adding SUMMA implementation for matrix multiplication C = A * B.
*	Assumptions:    A, B, and C are square matrices n by n;
*			the total number of processors (np) is a square number (q^2).
*	To compile, use
*	    mpicc -o summa summa.c
*       To run, use
*	    mpiexec -n $(NPROCS) ./summa

* originally from: https://github.com/zhongyr/SUMMA_MPI/blob/master/summa.c
* SUMMA_MPI / summa.c

* ATTENTION: NOT POSSIBLE FOR ME WITH JUST 2 CORES / 2 PROCESSES !!!
*********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <mpi.h>


#define min(a, b) ((a < b) ? a : b)
#define SZ 1024		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.




void printDiff(double **data1, double **data2, int n, int iListLength, float fListTol) {

	printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
  int i,j,k;
  int error_count=0;

  for (j = 0; j < n; j++) {
		if (error_count < iListLength) {
			//printf("\n  Row %d:\n", j);
    }

    for (i = 0; i < n; i++) {
			double fDiff = fabs(data1[j][i] - data2[j][i]);

      if (fDiff > fListTol) {
				if (error_count < iListLength) {
					printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[j][i], data2[j][i], fDiff);
        }

        error_count++;
      }
  	}
  }

  printf(" \n  Total Errors = %d\n", error_count);
}


/**
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {
	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));
  array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));

	for (i=1; i<n_rows; i++) {
		array[i] = array[0] + i * n_cols;
  }
  return array;
}

/**
*	Initialize arrays A and B with random numbers, and array C with zeros.
*	Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz) {
	int i, j;
	double value;
	// Set random values...technically it is already random and this is redundant
	for (i=0; i<blck_sz; i++){
		for (j=0; j<blck_sz; j++){
			lA[i][j] = rand() /(double)RAND_MAX ;
			lB[i][j] = rand() /(double)RAND_MAX;
			lC[i][j] = 0.0;
		}
	}
}

// implementation of mult C = C + A x B
void matmulAdd(double **C,double **A,double **B, int n) {
	int i,j,k;
	for(i=0;i<n;i++) {
		for(j=0;j<n;j++) {
			double result=0;
			for(k=0;k<n;k++) {
				result+=A[i][k]*B[k][j];
			}
			C[i][j]+=result;
		}
	}
}


/**
*	Perform the SUMMA matrix multiplication.
*       Follow the pseudo code in lecture slides.
*/
void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A, double **my_B, double **my_C) {

	//Add your implementation of SUMMA algorithm

	/*create row and column comms*/
	MPI_Comm grid_comm;
	int dimsizes[2];
	int wraparound[2];
	int coordinates[2];
	int free_coords[2];
	int reorder = 1;
	int my_grid_rank, grid_rank;
	int row_test, col_test;

	MPI_Comm row_comm;
	MPI_Comm col_comm;
	//q = (int)sqrt((double)p);
	dimsizes[0] = dimsizes[1] = proc_grid_sz;
	wraparound[0] = wraparound[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
	MPI_Comm_rank(grid_comm, &my_grid_rank);
	MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);

	//printf("Process %d > my_grid_rank = %d, coords = (%d, %d), grid_rank = %d\n", my_rank, my_grid_rank, coordinates[0], coordinates[1], grid_rank);

	free_coords[0] = 0;
	free_coords[1] = 1;
	MPI_Cart_sub(grid_comm,free_coords,&row_comm);
 	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid_comm, free_coords,&col_comm);
  int x_sz;
  //MPI_Comm_size(row_comm,&x_sz);
  //printf("%d\n",x_sz);
	int k;

	/**
	 * ALLOCATE BUFFER FOR RECVED BLOCK IN A AND BLOCK IN B
	 * **/
	double **buffA, **buffB;
	buffA = alloc_2d_double(block_sz,block_sz);
	buffB = alloc_2d_double(block_sz,block_sz);

	for(k = 0;k<proc_grid_sz;k++){
    printf("root k : %d\n",k);

		if (coordinates[1]==k)
			memcpy(*buffA,*my_A,block_sz*block_sz*sizeof(double));
		MPI_Bcast(*buffA, block_sz*block_sz, MPI_DOUBLE, k, row_comm); //broadcast buffA from (j,k) to row j  where j = 0..proc_grid_sz

		if(coordinates[0]==k)
			memcpy(*buffB,*my_B,block_sz*block_sz*sizeof(double));
		MPI_Bcast(*buffB, block_sz*block_sz, MPI_DOUBLE, k, col_comm); // broadcast buffB from (k,j) to column j where j = 0..proc_grid_sz

		//matmulAdd(my_C,buffA,buffB,block_sz);


	  if (coordinates[0]==k && coordinates[1]==k)
		 	matmulAdd(my_C,my_A,my_B,block_sz);  // grid (k,k) buffA==my_A buff_B==my_B

		else if(coordinates[0]==k)
		 	matmulAdd(my_C,buffA,my_B,block_sz); //grid (k,j) buffB == my_B

		else if (coordinates[1])
		 	matmulAdd(my_C,my_A,buffB,block_sz); // grid (j,k) buffA == my_A

		else
		 	matmulAdd(my_C,buffA,buffB,block_sz);

		//this fragment is used in the slides but I think this is equal to just use buffA and buffB
	}
  int i;
	//free((double *)buffA[0]);
	//free((double *)buffB[0]);
	free(*buffA);
	free(*buffB);
  free(buffA);
  free(buffB);
}

//print mat for test
void printMat(double **A, int n) {
	int i,j;
	for(i=0;i<n;i++) {
		for(j=0;j<n;j++) {
			printf("%.4lf  ",A[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides
	MPI_Status status;

	//num_proc = 4;


	srand(time(NULL));							// Seed random numbers
  printf("MPI initiate------------");
/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/
	MPI_Init(&argc, &argv); //1) start process
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc); //2) get total number of processors
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 3) get process rank
	printf("success\n");


/* assign values to 1) proc_grid_sz and 2) block_sz*/
	proc_grid_sz = (int)sqrt(num_proc); // 1) proc_grid_sz = q = sqrt(np),   since np = q^2

	if (proc_grid_sz != sqrt(num_proc)) {
		printf("num_proc need to be q^2\n");
		exit(1);
	}

	block_sz = SZ / proc_grid_sz; //2) get block_sz   block_sz = n/q

	if (SZ % proc_grid_sz != 0) {
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process

	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);


	initialize(A, B, C, block_sz);
  printf("-----------------------start matmul%d-------------------\n",rank);
	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();


	// Use SUMMA algorithm to calculate product C
	matmul(rank, proc_grid_sz, block_sz, A, B, C);


	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();
  //printf("finish\n");


	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time;

	// Insert statements for testing
	printf("finish %d\n",rank);
	//send local mat to process 0;
 	if(rank!=0) {
		MPI_Send(&A[0][0],block_sz*block_sz,MPI_DOUBLE,0,rank+100,MPI_COMM_WORLD);
		MPI_Send(&B[0][0],block_sz*block_sz,MPI_DOUBLE,0,rank+200,MPI_COMM_WORLD);
		MPI_Send(&C[0][0],block_sz*block_sz,MPI_DOUBLE,0,rank+300,MPI_COMM_WORLD);
 	}

	if (rank == 0) {
		double **matrixA;
		double **matrixB;
		double **matrixC;
	  matrixA = alloc_2d_double(SZ, SZ);
    matrixB = alloc_2d_double(SZ, SZ);
    matrixC = alloc_2d_double(SZ, SZ);
    int i;

		for( i=0; i<num_proc; i++ ) {
			double *tempA = (double *)malloc(block_sz*block_sz*sizeof(double));
			double *tempB = (double *)malloc(block_sz*block_sz*sizeof(double));
			double *tempC = (double *)malloc(block_sz*block_sz*sizeof(double));

			if( i!=0 ) {
				MPI_Recv(tempA,block_sz*block_sz,MPI_DOUBLE,i,i+100,MPI_COMM_WORLD,&status);
				MPI_Recv(tempB,block_sz*block_sz,MPI_DOUBLE,i,i+200,MPI_COMM_WORLD,&status);
				MPI_Recv(tempC,block_sz*block_sz,MPI_DOUBLE,i,i+300,MPI_COMM_WORLD,&status);
      }

			else {
				memcpy(tempA,*A,block_sz*block_sz*sizeof(double));
       	memcpy(tempB,*B,block_sz*block_sz*sizeof(double));
       	memcpy(tempC,*C,block_sz*block_sz*sizeof(double));
      }
			printf("finish recv\n");

			int p=0;
      int j,k;

			for(j=0; j<block_sz; j++) {
				for(k=0; k<block_sz; k++) {
					matrixA[ j+(int)(i/proc_grid_sz)*block_sz][k+i%proc_grid_sz*block_sz ] = tempA[p];
					matrixB[ j+(int)(i/proc_grid_sz)*block_sz][k+i%proc_grid_sz*block_sz ] = tempB[p];
					matrixC[ j+(int)(i/proc_grid_sz)*block_sz][k+i%proc_grid_sz*block_sz ] = tempC[p];

					p++;
				}
			}
      printf("finish gather\n");

			free(tempA);
			free(tempB);
			free(tempC);

		}

   	int j, k;
  	double **solution;
    solution = alloc_2d_double(SZ, SZ);

		for(j=0; j<SZ; j++) {
      for(k=0; k<SZ; k++) {
        solution[j][k] = 0.0;
      }
    }

		matmulAdd(solution, matrixA, matrixB, SZ);

		// Print in pseudo csv format for easier results compilation
		printf("Mat A : \n");
		printMat(matrixA,SZ);

		printf("Mat B : \n");
		printMat(matrixB,SZ);

		printf("Mat C : \n");
		printMat(matrixC,SZ);

    printf("Mat S : \n");
    printMat(solution,SZ);

		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n", SZ, num_proc, total_time);

		printDiff(solution,matrixC,SZ,100,1.0e-5f);
  //for(i=0;i<SZ;i++){
    //free((void*)matrixA[i]);
    //free((void*)matrixB[i]);
   // free((void*)matrixC[i]);
  //}
    //free(matrixA);
    //free(matrixB);
    //free(matrixC);
	}
 //int i;
  //for(i=0;i<block_sz;i++){
   // free((void*)A[0]);
    //free((void*)B[0]);
   // free((void*)C[0]);
  //}
  //free(A);
  //free(B);
  //free(C);

	// Destroy MPI processes

	MPI_Finalize();

	return 0;
}
