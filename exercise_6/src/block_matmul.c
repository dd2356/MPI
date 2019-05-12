#include <stdio.h>
#include <stdlib.h>
#include "block_matmul.h"

struct Config {
	/* MPI Files */
	MPI_File A_file, B_file, C_file;
	char *outfile;

	/* MPI Datatypes for matrix blocks */
	MPI_Datatype block;

	/* Matrix data */
	double *A, *A_tmp, *B, *B_tmp, *C;

	/* Cart communicators */
	MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;

	/* Cart communicator dim and ranks */
	int dim[2], coords[2];
	int world_rank, world_size, grid_rank;
	int row_rank, row_size, col_rank, col_size;

	/* Full matrix dim */
	int A_dims[2];
	int B_dims[2];
	int C_dims[2];
	int matrix_size;

	/* Process local matrix dim */
	int local_dims[2];
	int local_size;
};

int is_square(int number) {
	int sq = (int)sqrt(number);
	return sq*sq == number;
}

struct Config config;

void init_matmul(char *A_file, char *B_file, char *outfile)
{
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (!is_square(world_size)) {
		if (world_rank == 0) {
			printf("Number of processes needs to be a perfect square\n");
		}
		exit(1);
	}

	MPI_File_open(MPI_COMM_WORLD, A_file, 
		MPI_MODE_RDONLY, MPI_INFO_NULL, &config.A_file);
	MPI_File_open(MPI_COMM_WORLD, B_file, 
		MPI_MODE_RDONLY, MPI_INFO_NULL, &config.B_file);
	MPI_File_open(MPI_COMM_WORLD, outfile, 
		MPI_MODE_WRONLY, MPI_INFO_NULL, &config.C_file);
	config.block = MPI_DOUBLE;

	MPI_File_read_at(config.A_file, 0, &config.A_dims,
		2, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_read_at(config.B_file, 0, &config.B_dims,
		2, MPI_INT, MPI_STATUS_IGNORE);

	config.C_dims[0] = config.A_dims[0]; config.C_dims[1] = config.B_dims[1];
	config.matrix_size = config.C_dims[0] * config.C_dims[1];
	printf("A dims: %d x %d\n", config.A_dims[0], config.A_dims[1]);

	config.grid_comm = MPI_COMM_WORLD;
	MPI_Comm_split(MPI_COMM_WORLD, world_rank / config.A_dims[1], 
		world_rank, &config.row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, world_rank % config.A_dims[1], 
		world_rank, &config.col_comm);

	config.world_rank = world_rank;
	config.world_size = world_size;
	config.grid_rank = world_rank;

	config.dim[0] = config.dim[1] = (int)sqrt(world_size);
	MPI_Comm_rank(config.row_comm, &config.row_rank);
	MPI_Comm_rank(config.col_comm, &config.col_rank);
	MPI_Comm_size(config.row_comm, &config.row_size);
	MPI_Comm_size(config.col_comm, &config.col_size);

	config.local_dims[0] = config.A_dims[0] / config.dim[0];
	config.local_dims[1] = config.A_dims[1] / config.dim[1];
	config.local_size = config.local_dims[0] * config.local_dims[1];
	printf("local dims: %d x %d\n", config.local_dims[0], config.local_dims[1]);

	config.A = (double*)malloc(sizeof(double) * config.local_size);
	config.A_tmp = (double*)malloc(sizeof(double) * config.local_size);
	config.B = (double*)malloc(sizeof(double) * config.local_size);
	config.B_tmp = (double*)malloc(sizeof(double) * config.local_size);
	config.C = (double*)calloc(sizeof(double), config.local_size);

	// config.A = new double[config.local_size];
	// config.A_tmp = new double[config.local_size];
	// config.B = new double[config.local_size];
	// config.B_tmp = new double[config.local_size];
	// config.C = new double[config.local_size]();


	for (int i = 0; i < config.local_dims[0]; i++) {
		int row_offset = i * config.A_dims[0] 
			+ config.col_rank * config.local_dims[0] * config.A_dims[1];
		int col_offset = config.local_dims[1] * config.row_rank;
		int array_offset_bytes = (col_offset + row_offset) * sizeof(double);
		int start_index = 2 * sizeof(int) + array_offset_bytes;

		MPI_File_read_at(config.A_file, start_index, 
			&config.A[i * config.local_dims[1]], config.local_dims[1], 
			MPI_DOUBLE, MPI_STATUS_IGNORE);
		MPI_File_read_at(config.B_file, start_index, 
			&config.B[i * config.local_dims[1]], config.local_dims[1], 
			MPI_DOUBLE, MPI_STATUS_IGNORE);
	}

	for (int i = 0; i < config.local_size; i++) {
		printf("%d: A %d: %f\n", config.world_rank, i, config.A[i]);
	}

	// make sure that A_tmp and B_tmp have the correct data to begin with (could be done with MPI)
	memcpy(config.A_tmp, config.A, config.local_size * sizeof(double));
	memcpy(config.B_tmp, config.B, config.local_size * sizeof(double));

	/* Verify dim of A and B matches for matul and both are square*/

	/* Create Cart communicator for NxN processes */

	/* Sub div cart communicator to N row communicator */

	/* Sub div cart communicator to N col communicator */

	/* Setup sizes of full matrices */

	/* Setup sizes of local matrix tiles */

	/* Create subarray datatype for local matrix tile */

	/* Create data array to load actual block matrix data */

	/* Set fileview of process to respective matrix block */

	/* Collective read blocks from files */

	/* Close data source files */
	// return;
	MPI_File_close(&config.A_file);
	MPI_File_close(&config.B_file);
}

void cleanup_matmul()
{
	// return;
	if (config.world_rank == 0) {
		printf("cleaning up\n");
		MPI_File_write_at(config.C_file, 0, config.C_dims,
			2, MPI_INT, MPI_STATUS_IGNORE);
		printf("header written!\n");
	}

	for (int i = 0; i < config.local_dims[0]; i++) {
		int row_offset = i * config.A_dims[0] 
			+ config.col_rank * config.local_dims[0] * config.A_dims[1];
		int col_offset = config.local_dims[1] * config.row_rank;
		int array_offset_bytes = (col_offset + row_offset) * sizeof(double);
		int start_index = 2 * sizeof(int) + array_offset_bytes;
		MPI_File_write_at(config.C_file, start_index, &config.C[i * config.local_dims[1]],
			config.local_dims[1], MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&config.C_file);
	free(config.A);
	free(config.A_tmp);
	free(config.B);
	free(config.B_tmp);
	free(config.C);
	printf("%d: cleanup complete!\n", config.world_rank);

}

// Uses the config struct exclusively, does not need any arguments
void multiply() {
	for (int m = 0; m < config.local_dims[0]; m++) {
		for (int n = 0; n < config.local_dims[1]; n++) {
			for (int k = 0; k < config.local_dims[1]; k++) {
				config.C[m * config.local_dims[1] + n] 
					+= config.A_tmp[m * config.local_dims[1] + k] 
					* config.B_tmp[k * config.local_dims[1] + n];
			}
		}
	}
}

void compute_fox()
{

	/* Compute source and target for verticle shift of B blocks */

	for (int i = 0; i < config.dim[0]; i++) {
		multiply();
		/* Diag + i broadcast block A horizontally and use A_tmp to preserve own local A */

		/* dgemm with blocks */
		
		/* Shfting block B upwards and receive from process below */

	}
}
