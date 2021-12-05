/*topfcns.c test basic topology functions
Algorithm:
	1.Build a 2−dimensional Cartesian communicator from MPI_Comm_world
	2.Print topology information for each process
	3.Use MPI_Cart_sub to build a communicator for each row of the Cartesian communicator
	4.Carry out a broadcast across each row communicator
	5.Print results of broadcast
	6.Use MPI_Cart_sub to build a communicator for each column of the Cartesian communicator
	7.Carry out a broadcast across each column  communicator
	8.Print results of broadcast

Note:Assumes the number of processes, p, is a perfect square
*/

#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[])
{
	int p, myrank, q; // number of processes, current rank, square of the processes
	MPI_Comm grid_comm; // new communicator inside the smaller grids
	int dimsizes[2]; // array called dimsizes with 2 elements
	int wraparound[2]; // array called wraparound with 2 elements
	int coordinates[2]; // array with coordinates for smaller grids with 2 elements
	int free_coords[2]; // array with free coordinates for allocation with 2 elements
	int reorder = 1;
	int my_grid_rank, grid_rank; // local grid-ranks and ????
	int row_test, col_test; // ?????

	MPI_Comm row_comm; // local row-communicator
	MPI_Comm col_comm; // local column-communicator

	MPI_Init(&argc, &argv); // Initialisation of MPI
	MPI_Comm_size(MPI_COMM_WORLD, &p); // Coomunicator size with pointer to number of processes p
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // communicator rank as typical to current rank

	q = (int)sqrt((double)p); // partitioning of the processes p into q
	dimsizes[0] = dimsizes[1] = q; // vector of dimension sizes gets values of partitioned q
	wraparound[0] = wraparound[1] = 1; // wraparound just gets an integer "1" ???

	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm); //(cartesian) communicator with topological information given by i.e. the dimension sizes created
	MPI_Comm_rank(grid_comm, &my_grid_rank); // local communicator of grids get their ranks
	MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates); // determination of process coordinates for topology by given ranks, actual coordinates and max dimension sizes
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank); // determination of process ranks in the communicator given their cartesian location in the local grids

	printf("Process %d > my_grid_rank = %d, coords = (%d, %d), grid_rank = %d\n", myrank, my_grid_rank, coordinates[0], coordinates[1], grid_rank);

	free_coords[0] = 0; // assigning values to free coordinates
	free_coords[1] = 1; // ...
	MPI_Cart_sub(grid_comm, free_coords, &row_comm); // partitioning of the (local) grid communicator (of the rows) into smaller sub-groups (just 0, 1 --> given by the free coordinates!)

	free_coords[0] = 1; // reassigning the values of the free coordinates
	free_coords[1] = 0; // ...
	MPI_Cart_sub(grid_comm, free_coords, &col_comm); // same partitioning, but for column-communicator and with "new" free coordinates

	if(coordinates[1] == 0)
		row_test = coordinates[0]; // assigning row_test variable as 0 by value of 2. coordinates element ???
	else
		row_test = -1; // WHY -1 ???
  int k = 0; // temporal k as the "root" process defined
	MPI_Bcast(&row_test, 1, MPI_INT, k, row_comm); // value adressed in row_test broadcasted / send by root process 0 (=k) as an integer value to all other communicators in the row-communicator
	printf("Process %d > coords = (%d, %d), row_test = %d\n", myrank, coordinates[0], coordinates[1], row_test);

	if(coordinates[0] == 0)
		col_test = coordinates[1]; // same for col_test as for rows
	else
		col_test= -1;
	MPI_Bcast(&col_test, 1, MPI_INT, k, col_comm);
	printf("Process %d > coords = (%d, %d), col_test = %d\n", myrank, coordinates[0], coordinates[1], col_test);
	//int k = 0;
  MPI_Bcast(&row_test, 1, MPI_INT, k, row_comm);
	printf("Process %d > coords = (%d, %d), row_test = %d\n", myrank, coordinates[0], coordinates[1], row_test);

	MPI_Finalize();

	return 0;
}
