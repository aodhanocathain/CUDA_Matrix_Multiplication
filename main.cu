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

//In all kernels, I use blockDim.x = 32 even for tiles smaller than 32x32, to stop any 1 warp crossing different rows of the output
//This is why I use TILE_WIDTH_IN_ELEMENTS instead of blockDim.x in kernels, because blockDim.x = 32 always

__global__ void matrix_multiply_on_gpu_naive(element* A, element* B, element* C)
{
	//naively load this thread's row of elements from A and column of elements from B

	if (threadIdx.x >= TILE_WIDTH_IN_ELEMENTS) { return; }

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * TILE_WIDTH_IN_ELEMENTS) + threadIdx.x;

	//compute dot product of row of A and column of B
	element sum = 0;
	for (int k = 0; k < N; k++)
	{
		sum += A[(row * N) + k] * B[(k * N) + col];
	}

	//write result
	C[(row * N) + col] = sum;
}

__global__ void matrix_multiply_on_gpu_broadcast(element* A, element* B, element* C)
{
	//like naive, but cache a coalesced memory access from A instead of making single element accesses from A
	//i.e., iterate through a row of A in vectors
	//each thread in the warp loads a single value of the vector, and accesses others via warp shuffles

	if (threadIdx.x >= TILE_WIDTH_IN_ELEMENTS) { return; }

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * TILE_WIDTH_IN_ELEMENTS) + threadIdx.x;

	element sum = 0;
	for (int vectorStart = 0; vectorStart < N; vectorStart += TILE_WIDTH_IN_ELEMENTS)
	{
		//this thread loads a single value from a vector in A's row
		element registerAvalue = A[(row * N) + vectorStart + threadIdx.x];
		for (int lane = 0; lane < TILE_WIDTH_IN_ELEMENTS; lane++)
		{
			//this thread accesses the rest of the vector via warp shuffles from other threads that loaded the other values
			sum += __shfl_sync(-1, registerAvalue, lane) * B[((vectorStart + lane) * N) + col];
		}
	}

	//write the result
	C[(row * N) + col] = sum;
}

__global__ void matrix_multiply_on_gpu_singlerotate_onewarp(element* ATranspose, element* B, element* C)
{
	//generate outer products from a row vector and column vector by rotating the column vector for each warp multiplication by the row vector,
	//as opposed to iteration through the column vector and broadcasting each value into a warp multiplication by the row vector

	//if threads still write in columns then the answer will be out of order but this can be fixed on the host side

	if (threadIdx.x >= TILE_WIDTH_IN_ELEMENTS) { return; }

	//one warp calculating the entire tile
	//(arbitrarily) decide that each thread outputs a column of different sums to the tile even if this leaves the output incorrectly ordered
	element sums[TILE_WIDTH_IN_ELEMENTS];
	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		sums[i] = 0;
	}

	int tileTop = blockIdx.y * TILE_WIDTH_IN_ELEMENTS;
	int col = (blockIdx.x * TILE_WIDTH_IN_ELEMENTS) + threadIdx.x;

	for (int pair = 0; pair < N; pair++)	//pairs consisting of a column vector from A (row vector from ATranspose), and a row vector of B
	{
		//element registerAvalue = A[((tileTop + threadIdx.x) * N) + pair];
		element registerAvalue = ATranspose[(pair * N) + tileTop + threadIdx.x];
		element registerBvalue = B[(pair * N) + col];

		for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
		{
			sums[i] += __shfl_sync(-1, registerAvalue, (threadIdx.x + i) % TILE_WIDTH_IN_ELEMENTS) * registerBvalue;
		}
	}

	//write result
	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		C[((tileTop + i) * N) + col] = sums[i];
	}
}

__global__ void matrix_multiply_on_gpu_doublerotate_onewarp(element* ATranspose, element* B, element* C)
{
	//similar to singlerotate_onewarp
	//instead of warp shuffling only a column from A, warp shuffle a column from A and a row from B
	//instead of TILE_WIDTH_IN_ELEMENTS shuffles of the A column, can do SQRT_TILE_WIDTH shuffles of A's column and B's row
	//for large values of TILE_WIDTH_IN_ELEMENTS, this poses a significant saving

	if (threadIdx.x >= TILE_WIDTH_IN_ELEMENTS) { return; }

	element sums[TILE_WIDTH_IN_ELEMENTS];
	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		sums[i] = 0;
	}

	//lanes from which this thread will be receiving values during warp shuffles
	int ALanes[SQRT_TILE_WIDTH];
	int BLanes[SQRT_TILE_WIDTH];

	//for two vectors X and Y,the element-wise product of (X rotate left L)(Y rotate right R)
	//has the same elements as X(Y rotate right (L+R)) and as (X rotate left (L+R))Y, just ordered differently

	//use this fact to mimic the single rotate strategy by generating the same elements as it, but even more disordered

	//count up the units B is rotated, then when it reaches SQRT_TILE_WIDTH, add this to a counter for A and reset for B.
	//this results in a pattern like this:

	// (A rotate left 0) x (B rotate right 0) has the same elements as (A rotate left 0)B
	// (A rotate left 0) x (B rotate right 1) has the same elements as (A rotate left 1)B
	// (A rotate left 0) x (B rotate right 2) has the same elements as (A rotate left 2)B
	//	...
	// (A rotate left 0) x (B rotate right SQRT_TILE_WIDTH-1) has the same elements as (A rotate left SQRT_TILE_WIDTH-1)B
	// (A rotate left SQRT_TILE_WIDTH) x (B rotate right 0) has the same elements as (A rotate left SQRT_TILE_WIDTH)B
	// ...
	// (A rotate left ((SQRT_TILE_WIDTH-1)*SQRT_TILE_WIDTH)) x (B rotate right SQRT_TILE_WIDTH-1) has the 
	// same elements as (A rotate left TILE_WIDTH_IN_ELEMENTS-1)B

	for (int i = 0; i < SQRT_TILE_WIDTH; i++)
	{
		ALanes[i] = (threadIdx.x + (SQRT_TILE_WIDTH * i)) % TILE_WIDTH_IN_ELEMENTS;	//rotate A left (receive from higher lanes)
		BLanes[i] = ((threadIdx.x - i) + TILE_WIDTH_IN_ELEMENTS) % TILE_WIDTH_IN_ELEMENTS;	//rotate B right (receive from lower lanes)
	}

	int tileTop = blockIdx.y * TILE_WIDTH_IN_ELEMENTS;
	int col = (blockIdx.x * TILE_WIDTH_IN_ELEMENTS) + threadIdx.x;

	for (int pair = 0; pair < N; pair++)	//pairs consisting of a column vector from A (row vector from ATranspose), and a row vector of B
	{
		//element registerAvalue = A[((tileTop + threadIdx.x) * N) + pair];
		element registerAvalue = ATranspose[(pair * N) + tileTop + threadIdx.x];
		element registerBvalue = B[(pair * N) + col];

		element AShuffles[SQRT_TILE_WIDTH];
		element BShuffles[SQRT_TILE_WIDTH];
		for (int shuffleIndex = 0; shuffleIndex < SQRT_TILE_WIDTH; shuffleIndex++)
		{
			AShuffles[shuffleIndex] = __shfl_sync(-1, registerAvalue, ALanes[shuffleIndex]);
			BShuffles[shuffleIndex] = __shfl_sync(-1, registerBvalue, BLanes[shuffleIndex]);
		}

		#pragma unroll
		for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
		{
			sums[i] += AShuffles[i / SQRT_TILE_WIDTH] * BShuffles[i % SQRT_TILE_WIDTH];
		}
	}

	//write result
	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		C[((tileTop + i) * N) + col] = sums[i];
	}
}

__global__ void matrix_multiply_on_gpu_doublerotate_allwarps(element* ATranspose, element* B, element* C)
{
	//same as doublerotate_onewarp but the main loop is split into as many ranges as warps in a tile
	//each warp executes its part of the range to get an outer product, and warps add their results at the end
	if (threadIdx.x >= TILE_WIDTH_IN_ELEMENTS) { return; }

	element sums[TILE_WIDTH_IN_ELEMENTS];
	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		sums[i] = 0;
	}

	//lanes from which this thread will be receiving values during warp shuffles
	int ALanes[SQRT_TILE_WIDTH];
	int BLanes[SQRT_TILE_WIDTH];
	for (int i = 0; i < SQRT_TILE_WIDTH; i++)
	{
		ALanes[i] = (threadIdx.x + (SQRT_TILE_WIDTH * i)) % TILE_WIDTH_IN_ELEMENTS;	//rotate A left (receive from higher lanes)
		BLanes[i] = ((threadIdx.x - i) + TILE_WIDTH_IN_ELEMENTS) % TILE_WIDTH_IN_ELEMENTS;	//rotate B right (receive from lower lanes)
	}

	int tileTop = blockIdx.y * TILE_WIDTH_IN_ELEMENTS;
	int col = (blockIdx.x * TILE_WIDTH_IN_ELEMENTS) + threadIdx.x;

	//split the range into as many parts as there are warps working on tiles
	//have each warp compute an incomplete outer product, add all warps' parts to get final result at end
	for (int pair = threadIdx.y * (N / TILE_WIDTH_IN_ELEMENTS); pair < (threadIdx.y + 1) * (N / TILE_WIDTH_IN_ELEMENTS); pair++)
	//pairs consisting of a column vector from A (row vector from ATranspose), and a row vector of B
	{
		//element registerAvalue = A[((tileTop + threadIdx.x) * N) + pair];
		element registerAvalue = ATranspose[(pair * N) + tileTop + threadIdx.x];
		element registerBvalue = B[(pair * N) + col];

		element AShuffles[SQRT_TILE_WIDTH];
		element BShuffles[SQRT_TILE_WIDTH];
		for (int shuffleIndex = 0; shuffleIndex < SQRT_TILE_WIDTH; shuffleIndex++)
		{
			AShuffles[shuffleIndex] = __shfl_sync(-1, registerAvalue, ALanes[shuffleIndex]);
			BShuffles[shuffleIndex] = __shfl_sync(-1, registerBvalue, BLanes[shuffleIndex]);
		}

		for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
		{
			sums[i] += AShuffles[i / SQRT_TILE_WIDTH] * BShuffles[i % SQRT_TILE_WIDTH];
		}
	}

	//write result

	//have each thread add its incomplete outer product to memory row by row
	//synchronize threads to write all their rwos but at any instant to write a unique row
	//alternatively could be using atomic adds

	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		__syncthreads();
		int row = (threadIdx.y + i) % TILE_WIDTH_IN_ELEMENTS;
		C[((tileTop + row) * N) + col] += sums[row];
	}
}

__global__ void matrix_multiply_on_gpu_doublerotate_onewarp_prefetching(element* ATranspose, element* B, element* C)
{
	//like doublerotate_onewarp but with prefetching

	if (threadIdx.x >= TILE_WIDTH_IN_ELEMENTS) { return; }

	element sums[TILE_WIDTH_IN_ELEMENTS];
	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		sums[i] = 0;
	}

	//lanes from which this thread will be receiving values during warp shuffles
	int ALanes[SQRT_TILE_WIDTH];
	int BLanes[SQRT_TILE_WIDTH];
	for (int i = 0; i < SQRT_TILE_WIDTH; i++)
	{
		ALanes[i] = (threadIdx.x + (SQRT_TILE_WIDTH * i)) % TILE_WIDTH_IN_ELEMENTS;
		BLanes[i] = ((threadIdx.x - i) + TILE_WIDTH_IN_ELEMENTS) % TILE_WIDTH_IN_ELEMENTS;
	}

	int tileTop = blockIdx.y * TILE_WIDTH_IN_ELEMENTS;
	int col = (blockIdx.x * TILE_WIDTH_IN_ELEMENTS) + threadIdx.x;
	
	int pair = 0;
	element nextAvalue = ATranspose[(pair * N) + tileTop + threadIdx.x];
	element nextBvalue = B[(pair * N) + col];

	while(pair < N-1)	//pairs consisting of a column vector from A (row vector from ATranspose), and a row vector of B
	{
		//element registerAvalue = A[((tileTop + threadIdx.x) * N) + pair];
		element registerAvalue = nextAvalue;
		element registerBvalue = nextBvalue;

		nextAvalue = ATranspose[((pair + 1) * N) + tileTop + threadIdx.x];
		nextBvalue = B[((pair+1) * N) + col];

		element AShuffles[SQRT_TILE_WIDTH];
		element BShuffles[SQRT_TILE_WIDTH];
		for (int shuffleIndex = 0; shuffleIndex < SQRT_TILE_WIDTH; shuffleIndex++)
		{
			AShuffles[shuffleIndex] = __shfl_sync(-1, registerAvalue, ALanes[shuffleIndex]);
			BShuffles[shuffleIndex] = __shfl_sync(-1, registerBvalue, BLanes[shuffleIndex]);
		}

		#pragma unroll
		for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
		{
			sums[i] += AShuffles[i / SQRT_TILE_WIDTH] * BShuffles[i % SQRT_TILE_WIDTH];
		}

		pair++;
	}

	element AShuffles[SQRT_TILE_WIDTH];
	element BShuffles[SQRT_TILE_WIDTH];
	for (int shuffleIndex = 0; shuffleIndex < SQRT_TILE_WIDTH; shuffleIndex++)
	{
		AShuffles[shuffleIndex] = __shfl_sync(-1, nextAvalue, ALanes[shuffleIndex]);
		BShuffles[shuffleIndex] = __shfl_sync(-1, nextBvalue, BLanes[shuffleIndex]);
	}

	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		sums[i] += AShuffles[i / SQRT_TILE_WIDTH] * BShuffles[i % SQRT_TILE_WIDTH];
	}

	//write result
	for (int i = 0; i < TILE_WIDTH_IN_ELEMENTS; i++)
	{
		C[((tileTop + i) * N) + col] = sums[i];
	}
}

KernelTest AKernelTests[] = {
	KernelTest(matrix_multiply_on_gpu_naive, "naive", ALLWARP_TILE_DIMS, NO_REORDERING),
	KernelTest(matrix_multiply_on_gpu_broadcast, "broadcast", ALLWARP_TILE_DIMS, NO_REORDERING)
};

KernelTest ATransposeKernelTests[] = {
	KernelTest(matrix_multiply_on_gpu_singlerotate_onewarp, "single rotate one warp", ONEWARP_TILE_DIMS, SINGLE_ROTATE),
	KernelTest(matrix_multiply_on_gpu_doublerotate_onewarp, "double rotate one warp", ONEWARP_TILE_DIMS, DOUBLE_ROTATE),
	//KernelTest(matrix_multiply_on_gpu_doublerotate_allwarps, "double rotate all warps", ALLWARP_TILE_DIMS, DOUBLE_ROTATE),
	KernelTest(matrix_multiply_on_gpu_doublerotate_onewarp_prefetching, "double rotate one warp prefetching", ONEWARP_TILE_DIMS, DOUBLE_ROTATE)
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

#pragma region A_transpose_kernels
	host_a.transpose();
	if (cudaMemcpy(device_A, host_a.elements.get(), sizeof(element) * N * N, cudaMemcpyHostToDevice))
	{
		cout << "Error copying A transpose to device memory\n";
	}

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
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, device_A, N, device_B, N, &beta, device_C, N);
	cudaDeviceSynchronize();
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