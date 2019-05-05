#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void compute_pi(int trials, int *count, double *pi) {
	double x, y;
	int count_sum;

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	srand ( time ( NULL) * world_rank);
	int mpi_trials = trials / world_size;
	if (world_rank == world_size - 1) {
		mpi_trials = trials - mpi_trials * (world_size - 1);
	}

	for (int i = 0; i < mpi_trials; i++) {
		x = (double)rand()/RAND_MAX;
		y = (double)rand()/RAND_MAX;
		*count += x*x + y*y < 1;
	}
	*pi = *count / (double)trials;

	printf("rank %2d: %6d / %6d = %.6f\n", world_rank, *count, trials, *pi);

	MPI_Reduce(
		&*count,
		&count_sum,
		1,
		MPI_INT,
		MPI_SUM,
		0,
		MPI_COMM_WORLD
	);

	if (world_rank == 0) {
		*pi = 4 * count_sum / (double)trials;
	}
}