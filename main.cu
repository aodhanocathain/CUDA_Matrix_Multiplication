#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <iostream>
using std::cout;
#include <string>
using std::string;
#include <ctime>

#include "KernelTest.cuh"
#include "Constants.hpp"
#include "FMatrix.hpp"

using std::endl;

//kernels that do not use warp shuffles to generate outer products
__global__ void matrix_multiply_on_gpu_naive(element* A, element* B, element* C)
{
	//inner product approach
	//naively load this thread's row of elements from A and column of elements from B
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	//compute dot product of row of A and column of B
	element sum = 0;
	for (int k = 0; k < N; k++)
	{
		sum += A[(row * N) + k] * B[(k * N) + col];
	}

	C[(row * N) + col] = sum;
}
__global__ void matrix_multiply_on_gpu_blocktiling_warptiling_threadtiling(element* A, element* B, element* C)
{
	__shared__ element Aslices[BLOCK_ITEMS_K][TILED_KERNEL_BLOCK_WIDTH];
	__shared__ element Bslices[BLOCK_ITEMS_K][TILED_KERNEL_BLOCK_WIDTH];

	element accumulators[REPLICATION][FRAGS_PER_THREAD][REPLICATION][FRAGS_PER_THREAD];
	for (int i = 0; i < REPLICATION; i++)
	{
		for (int j = 0; j < FRAGS_PER_THREAD; j++)
		{
			for (int k = 0; k < REPLICATION; k++)
			{
				for (int l = 0; l < FRAGS_PER_THREAD; l++)
				{
					accumulators[i][j][k][l] = 0;
				}
			}
		}
	}

	int id = (threadIdx.y * WARP_SIZE) + threadIdx.x;

	int ALoadRow = (blockIdx.y * TILED_KERNEL_BLOCK_WIDTH) + (id % TILED_KERNEL_BLOCK_WIDTH);
	int BLoadCol = (blockIdx.x * TILED_KERNEL_BLOCK_WIDTH) + (id % TILED_KERNEL_BLOCK_WIDTH);
	for (int k = 0; k < N; k += BLOCK_ITEMS_K)
	{
		Aslices[id / TILED_KERNEL_BLOCK_WIDTH][id % TILED_KERNEL_BLOCK_WIDTH] = A[(ALoadRow * N) + k + (id / TILED_KERNEL_BLOCK_WIDTH)];
		Bslices[id / TILED_KERNEL_BLOCK_WIDTH][id % TILED_KERNEL_BLOCK_WIDTH] = B[((k + (id / TILED_KERNEL_BLOCK_WIDTH)) * N) + BLoadCol];
		__syncthreads();

		for (int slice = 0; slice < BLOCK_ITEMS_K; slice++)
		{
			element Afragments[REPLICATION][FRAGS_PER_THREAD];
			element Bfragments[REPLICATION][FRAGS_PER_THREAD];

			for (int rep = 0; rep < REPLICATION; rep++)
			{
				for (int frag = 0; frag < FRAGS_PER_THREAD; frag++)
				{
					Afragments[rep][frag] = Aslices[slice][
						((threadIdx.y / WARPS_ACROSS) * (THREADS_DOWN * FRAGS_PER_THREAD * REPLICATION)) +
							(rep * (THREADS_DOWN * FRAGS_PER_THREAD)) +
							((threadIdx.x / THREADS_ACROSS) * FRAGS_PER_THREAD) +
							frag
					];
					Bfragments[rep][frag] = Bslices[slice][
						((threadIdx.y % WARPS_ACROSS) * (THREADS_ACROSS * FRAGS_PER_THREAD * REPLICATION)) +
							(rep * (THREADS_ACROSS * FRAGS_PER_THREAD)) +
							((threadIdx.x % THREADS_ACROSS) * FRAGS_PER_THREAD) +
							frag
					];
				}
			}

			for (int i = 0; i < REPLICATION; i++)
			{
				for (int j = 0; j < FRAGS_PER_THREAD; j++)
				{
					for (int l = 0; l < REPLICATION; l++)
					{
						for (int m = 0; m < FRAGS_PER_THREAD; m++)
						{
							accumulators[i][j][l][m] += Afragments[i][j] * Bfragments[l][m];
						}
					}
				}
			}
		}
		__syncthreads();
	}

	int tileTop = blockIdx.y * TILED_KERNEL_BLOCK_WIDTH;
	int tileLeft = blockIdx.x * TILED_KERNEL_BLOCK_WIDTH;
	for (int i = 0; i < REPLICATION; i++)
	{
		for (int j = 0; j < FRAGS_PER_THREAD; j++)
		{
			for (int k = 0; k < REPLICATION; k++)
			{
				for (int l = 0; l < FRAGS_PER_THREAD; l++)
				{
					int row =
						tileTop +
						((threadIdx.y / THREADS_ACROSS) * (THREADS_DOWN * REPLICATION * FRAGS_PER_THREAD)) +
						(i * THREADS_DOWN * FRAGS_PER_THREAD) +
						((threadIdx.x / THREADS_ACROSS) * FRAGS_PER_THREAD) +
						j;
					int col =
						tileLeft +
						((threadIdx.y % THREADS_ACROSS) * (THREADS_ACROSS * REPLICATION * FRAGS_PER_THREAD)) + //skip warps
						(k * THREADS_ACROSS * FRAGS_PER_THREAD) + //skip replications
						((threadIdx.x % THREADS_ACROSS) * FRAGS_PER_THREAD) + //skip frags
						l;	//skip elements
					C[(row * N) + col] = accumulators[i][j][k][l];
					//C[(row * N) + col] = 69;
				}
			}
		}
	}
}

//In kernels that generate outer products with warp shuffles, I use blockDim.x = 32 to keep each warp of 32 threads confined to one row/column.
//e.g. if blockDim.x was only 8, then 32 threads in a warp would cover 4 rows/columns of 8 elements.
//If rows/columns are shorter than 32, the excess threads in the warp simply return, doing nothing.
//Keeping all threads in one warp on the same row/column ensures that warp shuffles only pass around elements from a single row/column.
//That is why I use ROTATION_VECTOR_WIDTH instead of blockDim.x in these kernels, because blockDim.x = 32 regardless.

//kernels that generate outer products using warp shuffles
__global__ void matrix_multiply_on_gpu_broadcast(element* A, element* B, element* C)
{
	//like naive, but cache a coalesced memory access from A instead of making single element accesses from A
	//i.e., iterate through a row of A in vectors
	//each thread in the warp loads a single value of the vector, and accesses others via warp shuffles

	if (threadIdx.x >= ROTATION_VECTOR_WIDTH) { return; }

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * ROTATION_VECTOR_WIDTH) + threadIdx.x;

	element sum = 0;
	for (int vectorStart = 0; vectorStart < N; vectorStart += ROTATION_VECTOR_WIDTH)
	{
		//this thread loads a single value from a vector in A's row
		element registerAvalue = A[(row * N) + vectorStart + threadIdx.x];
		for (int lane = 0; lane < ROTATION_VECTOR_WIDTH; lane++)
		{
			//this thread accesses the rest of the vector via warp shuffles from other threads that loaded the other values
			sum += __shfl_sync(-1, registerAvalue, lane) * B[((vectorStart + lane) * N) + col];
		}
	}

	//write the result
	C[(row * N) + col] = sum;
}
__global__ void matrix_multiply_on_gpu_singlerotate(element* ATranspose, element* B, element* C)
{
	//generate outer products from a row vector and column vector by rotating the column vector for each warp multiplication by the row vector,
	//as opposed to iteration through the column vector and broadcasting each value into a warp multiplication by the row vector

	//if threads still write in columns then the answer will be out of order but this can be fixed on the host side

	if (threadIdx.x >= ROTATION_VECTOR_WIDTH) { return; }

	//one warp calculating the entire tile
	//(arbitrarily) decide that each thread outputs a column of different sums to the tile even if this leaves the output incorrectly ordered
	element sums[ROTATION_VECTOR_WIDTH];
	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		sums[i] = 0;
	}

	int tileTop = blockIdx.y * ROTATION_VECTOR_WIDTH;
	int col = (blockIdx.x * ROTATION_VECTOR_WIDTH) + threadIdx.x;

	for (int pair = 0; pair < N; pair++)	//pairs consisting of a column vector from A (row vector from ATranspose), and a row vector of B
	{
		//element registerAvalue = A[((tileTop + threadIdx.x) * N) + pair];
		element registerAvalue = ATranspose[(pair * N) + tileTop + threadIdx.x];
		element registerBvalue = B[(pair * N) + col];

		for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
		{
			sums[i] += __shfl_sync(-1, registerAvalue, (threadIdx.x + i) % ROTATION_VECTOR_WIDTH) * registerBvalue;
		}
	}

	//write result
	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		C[((tileTop + i) * N) + col] = sums[i];
	}
}
__global__ void matrix_multiply_on_gpu_doublerotate(element* ATranspose, element* B, element* C)
{
	//similar to singlerotate_onewarp
	//instead of warp shuffling only a column from A, warp shuffle a column from A and a row from B
	//instead of ROTATION_VECTOR_WIDTH shuffles of the A column, can do SQRT_ROTATION_VECTOR_WIDTH shuffles of A's column and B's row
	//for large values of TILE_WIDTH_IN_ELEMENTS, this poses a significant saving

	if (threadIdx.x >= ROTATION_VECTOR_WIDTH) { return; }

	element sums[ROTATION_VECTOR_WIDTH];
	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		sums[i] = 0;
	}

	//lanes from which this thread will be receiving values during warp shuffles
	int ALanes[SQRT_ROTATION_VECTOR_WIDTH];
	int BLanes[SQRT_ROTATION_VECTOR_WIDTH];

	//for two vectors X and Y,the element-wise product of (X rotate left L)(Y rotate right R)
	//has the same elements as X(Y rotate right (L+R)) and as (X rotate left (L+R))Y, just ordered differently

	//use this fact to mimic the single rotate strategy by generating the same elements as it
	//the elements will be even more disordered but can be generated in fewer shuffles

	//count up the units B is rotated, then when it reaches SQRT_ROTATION_VECTOR_WIDTH, add this to a counter for A and reset for B.
	//this results in a pattern like this:

	// (A rotate left 0) x (B rotate right 0) has the same elements as (A rotate left 0)B
	// (A rotate left 0) x (B rotate right 1) has the same elements as (A rotate left 1)B
	// (A rotate left 0) x (B rotate right 2) has the same elements as (A rotate left 2)B
	//	...
	// (A rotate left 0) x (B rotate right SQRT_ROTATION_VECTOR_WIDTH-1) has the same elements as (A rotate left SQRT_ROTATION_VECTOR_WIDTH-1)B
	// (A rotate left SQRT_ROTATION_VECTOR_WIDTH) x (B rotate right 0) has the same elements as (A rotate left SQRT_ROTATION_VECTOR_WIDTH)B
	// ...
	// (A rotate left ((SQRT_ROTATION_VECTOR_WIDTH-1)*SQRT_ROTATION_VECTOR_WIDTH)) x (B rotate right SQRT_ROTATION_VECTOR_WIDTH-1) has the 
	// same elements as (A rotate left ROTATION_VECTOR_WIDTH-1)B

	for (int i = 0; i < SQRT_ROTATION_VECTOR_WIDTH; i++)
	{
		ALanes[i] = (threadIdx.x + (SQRT_ROTATION_VECTOR_WIDTH * i)) % ROTATION_VECTOR_WIDTH;	//rotate A left (receive from higher lanes)
		BLanes[i] = ((threadIdx.x - i) + ROTATION_VECTOR_WIDTH) % ROTATION_VECTOR_WIDTH;	//rotate B right (receive from lower lanes)
	}

	int tileTop = blockIdx.y * ROTATION_VECTOR_WIDTH;
	int col = (blockIdx.x * ROTATION_VECTOR_WIDTH) + threadIdx.x;

	for (int pair = 0; pair < N; pair++)	//pairs consisting of a column vector from A (row vector from ATranspose), and a row vector of B
	{
		//element registerAvalue = A[((tileTop + threadIdx.x) * N) + pair];
		element registerAvalue = ATranspose[(pair * N) + tileTop + threadIdx.x];
		element registerBvalue = B[(pair * N) + col];

		element AShuffles[SQRT_ROTATION_VECTOR_WIDTH];
		element BShuffles[SQRT_ROTATION_VECTOR_WIDTH];
		for (int shuffleIndex = 0; shuffleIndex < SQRT_ROTATION_VECTOR_WIDTH; shuffleIndex++)
		{
			AShuffles[shuffleIndex] = __shfl_sync(-1, registerAvalue, ALanes[shuffleIndex]);
			BShuffles[shuffleIndex] = __shfl_sync(-1, registerBvalue, BLanes[shuffleIndex]);
		}

		#pragma unroll
		for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
		{
			sums[i] += AShuffles[i / SQRT_ROTATION_VECTOR_WIDTH] * BShuffles[i % SQRT_ROTATION_VECTOR_WIDTH];
		}
	}

	//write result
	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		C[((tileTop + i) * N) + col] = sums[i];
	}
}
__global__ void matrix_multiply_on_gpu_doublerotate_morewarps(element* ATranspose, element* B, element* C)
{
	//same as doublerotate_onewarp but the main loop is split into as many ranges as warps in a tile
	//each warp executes its part of the range to get an outer product, and warps add their results at the end
	if (threadIdx.x >= ROTATION_VECTOR_WIDTH) { return; }

	element sums[ROTATION_VECTOR_WIDTH];
	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		sums[i] = 0;
	}

	//lanes from which this thread will be receiving values during warp shuffles
	int ALanes[SQRT_ROTATION_VECTOR_WIDTH];
	int BLanes[SQRT_ROTATION_VECTOR_WIDTH];
	for (int i = 0; i < SQRT_ROTATION_VECTOR_WIDTH; i++)
	{
		ALanes[i] = (threadIdx.x + (SQRT_ROTATION_VECTOR_WIDTH * i)) % ROTATION_VECTOR_WIDTH;	//rotate A left (receive from higher lanes)
		BLanes[i] = ((threadIdx.x - i) + ROTATION_VECTOR_WIDTH) % ROTATION_VECTOR_WIDTH;	//rotate B right (receive from lower lanes)
	}

	int tileTop = blockIdx.y * ROTATION_VECTOR_WIDTH;
	int col = (blockIdx.x * ROTATION_VECTOR_WIDTH) + threadIdx.x;

	//split the range into as many parts as there are warps working on tiles
	//have each warp compute an incomplete outer product, add all warps' parts to get final result at end
	for (int pair = threadIdx.y * (N / ROTATION_VECTOR_WIDTH); pair < (threadIdx.y + 1) * (N / ROTATION_VECTOR_WIDTH); pair++)
	//pairs consisting of a column vector from A (row vector from ATranspose), and a row vector of B
	{
		//element registerAvalue = A[((tileTop + threadIdx.x) * N) + pair];
		element registerAvalue = ATranspose[(pair * N) + tileTop + threadIdx.x];
		element registerBvalue = B[(pair * N) + col];

		element AShuffles[SQRT_ROTATION_VECTOR_WIDTH];
		element BShuffles[SQRT_ROTATION_VECTOR_WIDTH];
		for (int shuffleIndex = 0; shuffleIndex < SQRT_ROTATION_VECTOR_WIDTH; shuffleIndex++)
		{
			AShuffles[shuffleIndex] = __shfl_sync(-1, registerAvalue, ALanes[shuffleIndex]);
			BShuffles[shuffleIndex] = __shfl_sync(-1, registerBvalue, BLanes[shuffleIndex]);
		}

		for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
		{
			sums[i] += AShuffles[i / SQRT_ROTATION_VECTOR_WIDTH] * BShuffles[i % SQRT_ROTATION_VECTOR_WIDTH];
		}
	}

	//write result

	//have each thread add its incomplete outer product to memory row by row
	//synchronize threads to write all their rwos but at any instant to write a unique row
	//alternatively could be using atomic adds

	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		__syncthreads();
		int row = (threadIdx.y + i) % ROTATION_VECTOR_WIDTH;
		C[((tileTop + row) * N) + col] += sums[row];
	}
}
__global__ void matrix_multiply_on_gpu_doublerotate_shareA(element* ATranspose, element* B, element* C)
{
	__shared__ element ATransposeTile[ROTATION_VECTOR_WIDTH][ROTATION_VECTOR_WIDTH];

	if (threadIdx.x >= ROTATION_VECTOR_WIDTH) { return; }

	element sums[ROTATION_VECTOR_WIDTH];
	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		sums[i] = 0;
	}

	int threadIdInBlock = (threadIdx.y * ROTATION_VECTOR_WIDTH) + threadIdx.x;
	int tileLeftColInATranspose = blockIdx.y * ROTATION_VECTOR_WIDTH;	//the column index in ATranspose of the tile's left column

	//iterating in tiles, traversing a column of tiles in ATranspose and a column of tiles in B
	for (int k = 0; k < N; k += ROTATION_VECTOR_WIDTH)
	{
		//load the common tile of A into shared memory, one row per warp
		ATransposeTile[threadIdx.y][threadIdx.x] = ATranspose[((k+threadIdx.y)*N)+(tileLeftColInATranspose+threadIdx.x)];
		for (int pair = 0; pair < ROTATION_VECTOR_WIDTH; pair++)
		{
			//each warp independently loads a unique slice of B and multiplies by the next slice of the tile of ATranspose
			element BShuffles[SQRT_ROTATION_VECTOR_WIDTH];
			element registerBvalue = B[((k + pair) * N) + (blockIdx.x* ROTATION_VECTOR_WIDTH * ROTATION_VECTOR_WIDTH) + threadIdInBlock];
			for (int shuffleIndex = 0; shuffleIndex < SQRT_ROTATION_VECTOR_WIDTH; shuffleIndex++)
			{
				BShuffles[shuffleIndex] = __shfl_sync(-1, registerBvalue, ((threadIdx.x - shuffleIndex) + ROTATION_VECTOR_WIDTH) % ROTATION_VECTOR_WIDTH);
				//BShuffles[shuffleIndex] = __shfl(-1, registerBvalue, ((threadIdx.x - shuffleIndex) + ROTATION_VECTOR_WIDTH) % ROTATION_VECTOR_WIDTH);
			}

			element AShuffles[SQRT_ROTATION_VECTOR_WIDTH];
			__syncthreads();
			element registerAvalue = ATransposeTile[pair][threadIdx.x];	//a row of ATranspose is a column of A
			for (int shuffleIndex = 0; shuffleIndex < SQRT_ROTATION_VECTOR_WIDTH; shuffleIndex++)
			{
				AShuffles[shuffleIndex] = __shfl_sync(-1, registerAvalue, (threadIdx.x + (SQRT_ROTATION_VECTOR_WIDTH * shuffleIndex)) % ROTATION_VECTOR_WIDTH);
				//AShuffles[shuffleIndex] = __shfl(-1, registerAvalue, (threadIdx.x + (SQRT_ROTATION_VECTOR_WIDTH * shuffleIndex)) % ROTATION_VECTOR_WIDTH);
			}

			for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
			{
				sums[i] += AShuffles[i / SQRT_ROTATION_VECTOR_WIDTH] * BShuffles[i % SQRT_ROTATION_VECTOR_WIDTH];
			}
		}
		__syncthreads();
	}

	for (int i = 0; i < ROTATION_VECTOR_WIDTH; i++)
	{
		C[(((blockIdx.y * ROTATION_VECTOR_WIDTH) + i) * N) + (blockIdx.x * ROTATION_VECTOR_WIDTH * ROTATION_VECTOR_WIDTH) + threadIdInBlock] = sums[i];
	}
}

KernelTest AKernelTests[] = {
	KernelTest(matrix_multiply_on_gpu_naive, "naive", dim3(MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS,MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS), dim3(WARP_SIZE,ROTATION_VECTOR_WIDTH), NO_REORDERING),
	KernelTest(matrix_multiply_on_gpu_naive, "broadcast", dim3(MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS,MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS), dim3(WARP_SIZE,ROTATION_VECTOR_WIDTH), NO_REORDERING),
	KernelTest(matrix_multiply_on_gpu_blocktiling_warptiling_threadtiling, "very tiled kernel", dim3(N/ TILED_KERNEL_BLOCK_WIDTH, N/ TILED_KERNEL_BLOCK_WIDTH), dim3(WARP_SIZE,WARPS_ACROSS*WARPS_DOWN), NO_REORDERING),
};


KernelTest ATransposeKernelTests[] = {
	KernelTest(matrix_multiply_on_gpu_singlerotate, "single rotate", dim3(MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS,MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS), dim3(WARP_SIZE,1), SINGLE_ROTATE),
	KernelTest(matrix_multiply_on_gpu_doublerotate, "double rotate", dim3(MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS,MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS), dim3(WARP_SIZE,1), DOUBLE_ROTATE),
	//KernelTest(matrix_multiply_on_gpu_doublerotate_morewarps, "double rotate more warps", dim3(MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS,MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS), dim3(WARP_SIZE,ROTATION_VECTOR_WIDTH), DOUBLE_ROTATE),
	KernelTest(matrix_multiply_on_gpu_doublerotate_shareA, "double rotate share A", dim3(N/(ROTATION_VECTOR_WIDTH * ROTATION_VECTOR_WIDTH), N/ ROTATION_VECTOR_WIDTH), dim3(WARP_SIZE, ROTATION_VECTOR_WIDTH), DOUBLE_ROTATE),
};


int main()
{
#pragma region setup
	//initialize input matrices host_a and host_b
	FMatrix host_a(N, N);
	host_a.randomize();
#ifdef PRINT_MATRICES
	host_a.print("host_a");
#endif

	FMatrix host_b(N, N);
	host_b.randomize();
#ifdef PRINT_MATRICES
	host_b.print("host_b");
#endif

	//calculate or allocate output matrix host_c with trusted host implementation
	time_t start_time = clock();
	FMatrix host_c =
#ifdef CHECK_ANSWERS
		host_a * host_b;
#else
		FMatrix(N, N);
#endif
	time_t end_time = clock();
#ifdef PRINT_KERNEL_TIMES
	cout << "host implementation took " << ((element)(end_time - start_time)) / CLOCKS_PER_SEC << " seconds\n";
#endif

#ifdef PRINT_MATRICES
	host_c.print("host_c");
#endif

	//create separate output matrix on the host, into which the device result will be copied
	FMatrix host_matrix_for_device_result(N, N);
	host_matrix_for_device_result.reset();

	//allocate space on the device for matrix elements
	element* device_A;
	element* device_B;
	element* device_C;
	if (
		cudaMalloc(&device_A, sizeof(element) * N * N) ||
		cudaMalloc(&device_B, sizeof(element) * N * N) ||
		cudaMalloc(&device_C, sizeof(element) * N * N)
		)
	{
		cout << "Error allocating device memory\n";
		return 1;
	}

	//copy matrix elements from the host to the allocated device space
	if (
		cudaMemcpy(device_A, host_a.elements.get(), sizeof(element) * N * N, cudaMemcpyHostToDevice) ||
		cudaMemcpy(device_B, host_b.elements.get(), sizeof(element) * N * N, cudaMemcpyHostToDevice) ||
		//copy the 0s to the device result as well
		cudaMemcpy(device_C, host_matrix_for_device_result.elements.get(), sizeof(element) * N * N, cudaMemcpyHostToDevice)
		)
	{
		cout << "Error copying host memory to device memory\n";
		return 1;
	}
#pragma endregion

#pragma region regular_A_kernels
	for (int i = 0; i < sizeof(AKernelTests)/sizeof(KernelTest); i++)
	{
		AKernelTests[i].run(device_A, device_B, device_C, host_matrix_for_device_result, host_c);
	}
#pragma endregion

	//transpose A
	//required for kernels that use A transposed, and for cublas
	host_a.transpose();
	if (cudaMemcpy(device_A, host_a.elements.get(), sizeof(element) * N * N, cudaMemcpyHostToDevice))
	{
		cout << "Error copying A transpose to device memory\n";
	}

#pragma region A_transpose_kernels
	for (int i = 0; i < sizeof(ATransposeKernelTests) / sizeof(KernelTest); i++)
	{
		ATransposeKernelTests[i].run(device_A, device_B, device_C, host_matrix_for_device_result, host_c);
	}
#pragma endregion

#pragma region cublas

	//run cublas version for comparison
	cout << "running cublas\n";

	host_matrix_for_device_result.reset();
	if (cudaMemcpy(device_C, host_matrix_for_device_result.elements.get(), sizeof(element) * N * N, cudaMemcpyHostToDevice))
	{
		cout << "Error copying reset memory from host to device\n";
	}

	cublasHandle_t handle;
	cublasCreate(&handle);
	element alpha = 1;
	element beta = 0;

	host_b.transpose();	//cuBLAS works on column-major matrices
	if (cudaMemcpy(device_B, host_b.elements.get(), sizeof(element) * N * N, cudaMemcpyHostToDevice))
	{
		cout << "Error copying memory from host to device\n";
	}

	start_time = clock();
	for (int test = 0; test < NUM_TESTS; test++)
	{
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, device_A, N, device_B, N, &beta, device_C, N);
		cudaDeviceSynchronize();
	}
	end_time = clock();

	cudaError_t cudaError = cudaGetLastError();
	if (cudaError) {
		cout << "CUDA error: " << cudaGetErrorString(cudaError) << endl;
	}

#ifdef PRINT_KERNEL_TIMES
	cout << "cublas took " << ((element)(end_time - start_time)) / CLOCKS_PER_SEC << " seconds\n";
#endif

	//copy device results to host
	if (cudaMemcpy(host_matrix_for_device_result.elements.get(), device_C, sizeof(element) * N * N, cudaMemcpyDeviceToHost))
	{
		cout << "Error copying device memory to host memory\n";
	}
	host_matrix_for_device_result.transpose();

#ifdef PRINT_MATRICES
	host_matrix_for_device_result.print("cublas");
#endif

#ifdef CHECK_ANSWERS
	if (host_c == host_matrix_for_device_result)
	{
		cout << "results were equal\n";
	}
	else
	{
		cout << "results were not equal\n";
	}
#endif
#pragma endregion

#pragma region cleanup
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);
#pragma endregion

	return 0;
}