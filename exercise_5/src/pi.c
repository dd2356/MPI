#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

#define DEBUG 0

void compute_pi(uint64_t trials, uint64_t *count, double *pi) {
	double x, y;
	uint64_t count_sum;

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_File fh;
	MPI_Status status;

	srand ( time ( NULL) * (world_rank + 1));
	uint64_t mpi_trials = trials / world_size;
	if (world_rank == world_size - 1) {
		mpi_trials = trials - mpi_trials * (world_size - 1);
	}

	clock_t start, end;
	double cpu_time_used;
	
	start = clock();

	for (uint64_t i = 0; i < mpi_trials; i++) {
		x = (double)rand()/RAND_MAX;
		y = (double)rand()/RAND_MAX;
		*count += x*x + y*y <= 1.0;
	}
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	*pi = *count / (double)trials;
	if (DEBUG) {
		printf("Loop time: %.5f\n", cpu_time_used);
	}

	// printf("rank %2d: %6lu / %6lu = %.6f\n", 
		// world_rank, *count, trials, *pi);

	char *str = (char*)malloc(16 * sizeof(char));;
	sprintf(str, "%5d %.6f\n", world_rank, *pi);

	MPI_File_open(MPI_COMM_SELF, "results.txt", MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_write_at(fh, 16 * world_rank, str,
		16, MPI_CHAR, &status);

	MPI_Reduce(&*count, &count_sum, 1, MPI_LONG,
		MPI_SUM, 0, MPI_COMM_WORLD);

	if (world_rank == 0) {
		*pi = 4 * count_sum / (double)trials;

		char *str = (char*)malloc(15 * sizeof(char));;
		sprintf(str, "pi = %.6f\n", *pi);
		MPI_File_write_at(fh, 16 * world_size, str,
			15, MPI_CHAR, &status);
	}
	MPI_File_close(&fh);
}
