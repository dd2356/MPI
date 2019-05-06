#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

void compute_pi(uint64_t trials, uint64_t *count, double *pi) {
	double x, y;
	uint64_t count_sum;

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	srand ( time ( NULL) * (world_rank + 1));
	uint64_t mpi_trials = trials / world_size;
	if (world_rank == world_size - 1) {
		mpi_trials = trials - mpi_trials * (world_size - 1);
	}

	for (uint64_t i = 0; i < mpi_trials; i++) {
		x = (double)rand()/RAND_MAX;
		y = (double)rand()/RAND_MAX;
		*count += x*x + y*y < 1.0;
	}
	*pi = *count / (double)trials;

	printf("rank %2d: %6lu / %6lu = %.6f\n", 
		world_rank, *count, trials, *pi);

	MPI_Reduce(&*count, &count_sum, 1, MPI_LONG,
		MPI_SUM, 0, MPI_COMM_WORLD);

	if (world_rank == 0) {
		*pi = 4 * count_sum / (double)trials;
	}
}