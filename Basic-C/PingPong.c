#include <stdio.h>
#include <mpi.h>

int main ( int argc, char** argv ) {
  int a = 10;
  int b = 20;
  int count_0 = 150;
  int count_1 = 150;
  int world_size, world_rank;

  MPI_Init ( NULL,NULL );
  MPI_Comm_size ( MPI_COMM_WORLD,&world_size );
  MPI_Comm_rank ( MPI_COMM_WORLD,&world_rank );

  while ( count_0>0 && count_1>0 && a!=0 && b!=0 ) {
    if ( world_rank!=0 ) {
      MPI_Recv ( &a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      MPI_Send ( &b, 1, MPI_INT, 0, 1, MPI_COMM_WORLD );
      count_1 -= a;
      printf ( "Process 1: %d\n", count_1 );
    } else {
        MPI_Send ( &a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD );
        MPI_Recv ( &b, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        count_0 -= b;
        printf ("Process 0: %d\n", count_0 );
    }
  }
  if ( count_1 <= 0 ) {
    MPI_Recv ( &a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    b = 0;
    MPI_Send ( &b, 1, MPI_INT, 0, 1, MPI_COMM_WORLD );
    printf ("Game over, process 1: %d\n", count_1 );
  }
  if ( count_0 <= 0 ) {
    a = 0;
    MPI_Send ( &a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD );
    MPI_Recv ( &b, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    printf ("Game over, process 0: %d\n", count_0 );
  }

  MPI_Finalize();
  return 0;
}
