#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {

	MPI_Init(NULL, NULL);
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int counter = 0;
	int next_rank = (world_rank + 1) % world_size;
	int previous_rank = (world_size + world_rank - 1) % world_size;
	if (world_rank == 0) {
		MPI_Send(&counter, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);

	}
	MPI_Recv(&counter, 1, MPI_INT, previous_rank, 0, MPI_COMM_WORLD,
		MPI_STATUS_IGNORE);
	printf("Rank %2d received %2d from rank %2d\n",
		 world_rank, counter, previous_rank);
	if (world_rank != 0) {
		counter++;
		MPI_Send(&counter, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}
