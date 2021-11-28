#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

const int MAX_STRING = 1000;

int main ( int argc, char **argv ) {
  int comm_sz;
  int my_rank;

  MPI_Init ( NULL,NULL );
  MPI_Comm_size ( MPI_COMM_WORLD, &comm_sz );
  MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank );

  if ( my_rank == 0 ) { printf ("\nComm size: %d\n\n", comm_sz ); }

  int i;
  for ( i=0; i<comm_sz; i++ ) {
    if ( i==my_rank ) {
      if ( !i ) {
        int x = 42;
        printf ( "Process %d sending initial message: %d\n", i, x );
        MPI_Send ( &x, sizeof(x), MPI_INT, i+1, 0, MPI_COMM_WORLD );
      } else if ( i<comm_sz-1 ) {
          int y;
          MPI_Recv ( &y, INT_MAX, MPI_INT, i-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
          printf ("Process %d sending message to next process. Message: %d\n", i, y );
          MPI_Send ( &y, sizeof(y), MPI_INT, i+1, 0, MPI_COMM_WORLD);
      } else {
        int z;
        MPI_Recv ( &z, INT_MAX, MPI_INT, i-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        printf ( "Last process with rank %d received message. Message: %d\n", i, z );
      }
    }
  }

  MPI_Finalize();
  return 0;
}
