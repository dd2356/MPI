#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "pi.h"

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("Usage: mpirun -n <processes> bin/pi.out <trials>\n");
		return 0;
	}

	uint64_t trials = atol(argv[1]);
	uint64_t count = 0;
	double pi;

	MPI_Init(NULL, NULL);
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	compute_pi(trials, &count, &pi);
	if (world_rank == 0) {
		printf("pi: %f\n", pi);
	}

	MPI_Finalize();

}