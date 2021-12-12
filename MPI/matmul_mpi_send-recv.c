/*
 *
 * This is a simple code using two processes. It must be run with -np
 * To compile, use mpicc -o matmul_mpi matmul_mpi.c
 * To run, use mpiexec -np 2 ./matmul_mpi matrixsize
 * E.g.:       mpiexec -np 2 ./matmul_mpi 1000
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

//#define n 4

int main(int argc, char *argv[])
{

	int ii;
	int myrank;
	int p;

	MPI_Status status;
 	int tag;
	int i,j,k;
  int n;

	int **a, **b, **c;

        if (argc == 2)
                n = atoi(argv[1]);
        else
                n = 4;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);


/*      allocate space for a */
        printf("malloc for a\n");

        a = (int **) malloc(n/p * sizeof(int *));
        a[0] = (int *) malloc (n * n/p * sizeof(int));
        for ( i = 1; i < n/p; i++)
                a[i] = a[0] + n * i;

        b = (int **) malloc(n * sizeof(int *));
        b[0] = (int *) malloc (n * n * sizeof(int));
        for ( i = 1; i < n; i++)
                b[i] = b[0] + n * i;

        c = (int **)malloc(n/p * sizeof(int *));
        c[0] = (int *) malloc (n * n/p * sizeof(int));
        for ( i = 1; i < n/p; i++)
                c[i] = c[0] + n * i;

/* populate for b */
	for (i = 0; i < n; i++)
		for(j = 0; j < n; j++)
		{
			b[i][j] = i + j + 2;
		}

	for ( i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
			printf("%d\t", b[i][j]);
		printf("\n");
	}

	/* Data distribution */

	if( myrank != 0 ) {
		MPI_Recv( &a[0][0], n*n/p, MPI_INT, 0, tag, MPI_COMM_WORLD, &status );
		MPI_Recv( &b[0][0], n*n, MPI_INT, 0, tag, MPI_COMM_WORLD, &status );
	} else {
		for( i=1; i<p; i++ ) {
			for (ii = 0; ii < n/p; ii++)
				for(j = 0; j < n; j++)
				{
					a[ii][j] = i * n/p + ii + j + 1;
				}
//                      printf("sending a to %d\n", i);
			MPI_Send( &a[0][0], n*n/p, MPI_INT, i, tag, MPI_COMM_WORLD );
			MPI_Send( &b[0][0], n*n, MPI_INT, i, tag, MPI_COMM_WORLD );
		}
	}

//	printf("populate root's a\n");
	if (myrank == 0){
		for (ii = 0; ii < n/p; ii++)
			for(j = 0; j < n; j++)
			{
				a[ii][j] = ii + j + 1;
			}
	}


	for ( i = 0; i < n/p; i++){
		for ( j = 0; j < n; j++)
			printf("%d\t", a[i][j]);
		printf("\n");
	}

	/* Computation */

  printf("calculating...\n");

	for ( i=0; i<n/p; i++)
		for (j=0; j<n; j++) {
			c[i][j]=0;
			for (k=0; k<n; k++)
				c[i][j] += a[i][k] * b[k][j];
 		}
	/*Print c on rank 0*/

	if (myrank == 0){
		for ( i = 0; i < n/p; i++){
			for ( j = 0; j < n; j++)
				printf("%d\t", c[i][j]);
			printf("\n");
		}
	}

	/* Result gathering */

	if (myrank != 0)
		MPI_Send( &c[0][0], n*n/p, MPI_INT, 0, tag, MPI_COMM_WORLD);
	else
  		for (i=1; i<p; i++){
				MPI_Recv( &c[0][0], n*n/p, MPI_INT, i, tag, MPI_COMM_WORLD, &status);

				for ( ii = 0; ii < n/p; ii++){
					for ( j = 0; j < n; j++)
						printf("%d\t", c[ii][j]);
						printf("\n");
	      }

      }

	MPI_Finalize();

	return 0;
}
