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
	MPI_Comm_rank(MPI_COMM_WORLD, &config.world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &config.world_size);
    if (config.world_rank == 0) { 
        printf("Size: %d\n",config.world_size); 
    }

	if (!is_square(config.world_size)) {
		if (config.world_rank == 0) {
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

    /* Copy output file name to configuration */
    config.outfile = outfile; 

	/* Get matrix size header */
    if (config.world_rank == 0) { 
        MPI_File_read_at(config.A_file, 0, &config.A_dims,
                2, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_read_at(config.B_file, 0, &config.B_dims,
                2, MPI_INT, MPI_STATUS_IGNORE);
    }

    /* Calculate result matrix dimensions */ 
    config.C_dims[0]=config.A_dims[0];
    config.C_dims[1]=config.B_dims[1];

	/* Broadcast global matrix sizes */
    MPI_Bcast(config.A_dims,2,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(config.B_dims,2,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(config.C_dims,2,MPI_INT,0,MPI_COMM_WORLD);

    /* Set dim of tiles relative to the number of processes as NxN where N=sqrt(world_size) */
	config.dim[0] = config.dim[1] = (int)sqrt(config.world_size);

	/* Verify dim of A and B matches for matul and both are square*/
	if (config.A_dims[0] != config.A_dims[1] 
		|| config.B_dims[0] != config.B_dims[1]
		|| config.C_dims[0] != config.C_dims[1]
		|| config.A_dims[0] != config.B_dims[0]
		|| config.A_dims[0] != config.C_dims[0]) {
		if (config.world_rank == 0) {
			printf("Matrices should be square and the same size\n");
		}
		exit(1);
	}

	if (config.A_dims[0] % config.dim[0] != 0 || config.A_dims[1] % config.dim[1] != 0) {
		if (config.world_rank == 0) {
			printf("Matrix not evenly divisible along grid\n");
		}
		exit(1);
	}

	/* Create Cart communicator for NxN processes */
    int wrap_around[2] = {1,0}; 
    MPI_Cart_create(MPI_COMM_WORLD,2,config.dim,wrap_around,1,&config.grid_comm); 

    MPI_Comm_rank(config.grid_comm,&config.grid_rank); 
    MPI_Cart_coords(config.grid_comm,config.grid_rank,2,config.coords);
    
	/* Sub div cart communicator to N row communicator */
    int remain_dims[2] = {0,1}; // Keep row
    MPI_Cart_sub(config.grid_comm,remain_dims,&config.row_comm); 

    MPI_Comm_rank(config.row_comm,&config.row_rank); 
    MPI_Comm_size(config.row_comm,&config.row_size); 

	/* Sub div cart communicator to N col communicator */
    remain_dims[0]=1; // Keep column
    remain_dims[1]=0; // Keep column
    MPI_Cart_sub(config.grid_comm,remain_dims,&config.col_comm); 

    MPI_Comm_rank(config.col_comm,&config.col_rank); 
    MPI_Comm_size(config.col_comm,&config.col_size); 

	/* Setup sizes of full matrices */
    config.matrix_size = config.A_dims[0] * config.B_dims[1]; 

	/* Setup sizes of local matrix tiles */
	config.local_dims[0] = config.A_dims[0] / config.dim[0];
	config.local_dims[1] = config.A_dims[1] / config.dim[1];
	config.local_size = config.local_dims[0] * config.local_dims[1];
    if (config.world_rank == 0) { 
        printf("local dims: %d x %d\n", config.local_dims[0], config.local_dims[1]);
    }

	/* Create subarray datatype for local matrix tile */
	config.block = MPI_DOUBLE;

	/* Create data array to load actual block matrix data */
	config.A = (double*)malloc(sizeof(double) * config.local_size);
	config.A_tmp = (double*)malloc(sizeof(double) * config.local_size);
	config.B = (double*)malloc(sizeof(double) * config.local_size);
	config.B_tmp = (double*)malloc(sizeof(double) * config.local_size);
	config.C = (double*)calloc(sizeof(double), config.local_size);

	/* Set fileview of process to respective matrix block */

	/* Collective read blocks from files */
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
    if (config.world_rank == 0) { 
        printf("Finished reading files\n"); 
    }
    
	for (int i = 0; i < config.local_size; i++) {
		//printf("%d: A %d: %f\n", config.world_rank, i, config.A[i]);
	}

	/* Close data source files */
	// return;
	MPI_File_close(&config.A_file);
	MPI_File_close(&config.B_file);
    
    /* 
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
	// printf("A dims: %d x %d\n", config.A_dims[0], config.A_dims[1]);

	config.dim[0] = config.dim[1] = (int)sqrt(config.world_size);

	if (config.A_dims[0] != config.A_dims[1] 
		|| config.B_dims[0] != config.B_dims[1]
		|| config.C_dims[0] != config.C_dims[1]
		|| config.A_dims[0] != config.B_dims[0]
		|| config.A_dims[0] != config.C_dims[0]) {
		if (config.world_rank == 0) {
			printf("Matrices should be square and the same size\n");
		}
		exit(1);
	}

	if (config.A_dims[0] % config.dim[0] != 0 || config.A_dims[1] % config.dim[1] != 0) {
		if (config.world_rank == 0) {
			printf("Matrix not evenly divisible along grid\n");
		}
		exit(1);
	}

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
    */

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
    printf("[ %d ] Multiplying\n",config.world_rank); 
	for (int m = 0; m < config.local_dims[0]; m++) {
		for (int n = 0; n < config.local_dims[1]; n++) {
			for (int k = 0; k < config.local_dims[1]; k++) {
				config.C[m * config.local_dims[1] + n] 
					+= config.A[m * config.local_dims[1] + k] 
					* config.B[k * config.local_dims[1] + n];
			}
		}
	}
}

void compute_fox()
{
    MPI_Status status;
    int root, source, dest; 
	/* Compute source and target for verticle shift of B blocks */
    source = (config.col_rank + 1) % config.dim[0]; 
    dest = (config.col_rank + config.dim[0] -1) % config.dim[0]; 

	for (int i = 0; i < config.dim[0]; i++) {
		/* Diag + i broadcast block A horizontally and use A_tmp to preserve own local A */
        root = (config.col_rank + i) % config.dim[0]; 
        memcpy(config.A_tmp,config.A,sizeof(double)*config.local_size); 
        MPI_Bcast(config.A,config.local_size,config.block,root,config.row_comm);


		/* dgemm with blocks */
        multiply();

        /* reset A from tmp */
        memcpy(config.A,config.A_tmp,sizeof(double)*config.local_size); 

        printf("[ %d ] Source:%d Destination:%d\n",config.world_rank, source, dest); 
		/* Shfting block B upwards and receive from process below */
        memcpy(config.B_tmp,config.B,sizeof(double)*config.local_size); 

        MPI_Sendrecv_replace(
                config.B_tmp,
                config.local_size,
                config.block,
                dest,
                0,
                source,
                0,
                config.col_comm,
                &status);	
        memcpy(config.B,config.B_tmp,sizeof(double)*config.local_size); 
	}
}
