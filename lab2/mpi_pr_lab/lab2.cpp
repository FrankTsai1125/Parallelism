#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
	// Initialize MPI
	MPI_Init(&argc, &argv);

	// Get rank and world size
	int rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Validate arguments
	if (argc != 3) {
		if (rank == 0) {
			fprintf(stderr, "must provide exactly 2 arguments!\n");
		}
		MPI_Finalize();
		return 1;
	}

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

	// Divide work: each process handles x in [start, end)
	unsigned long long chunk_size = r / world_size;
	unsigned long long remainder = r % world_size;
	
	unsigned long long start, end;
	if (rank < remainder) {
		// First 'remainder' processes get one extra element
		start = rank * (chunk_size + 1);
		end = start + chunk_size + 1;
	} else {
		// Remaining processes get standard chunk_size
		start = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size;
		end = start + chunk_size;
	}

	// Local computation
	unsigned long long local_pixels = 0;
	for (unsigned long long x = start; x < end; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		local_pixels += y;
		local_pixels %= k;
	}

	// Reduce all local results to rank 0
	unsigned long long total_pixels = 0;
	MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, 
	           MPI_SUM, 0, MPI_COMM_WORLD);

	// Rank 0 prints the final result
	if (rank == 0) {
		printf("%llu\n", (4 * total_pixels) % k);
	}

	MPI_Finalize();
	return 0;
}

