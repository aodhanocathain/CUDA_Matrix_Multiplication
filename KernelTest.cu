#include <string>
using std::string;
#include <iostream>
using std::cout;

#include "KernelTest.cuh"
#include "FMatrix.hpp"
#include "Constants.hpp"

using std::endl;

KernelTest::KernelTest(void (*kernel)(element* device_A, element* device_B, element* device_C), string kernelName, dim3 tileDims, enum reorder postKernelReordering) :
	kernel(kernel), kernelName(kernelName), tileDims(tileDims), postKernelReordering(postKernelReordering) {}

void KernelTest::run(element* device_A, element* device_B, element* device_C, FMatrix& host_matrix_for_device_answer, FMatrix& host_answer)
{
	cout << "runnning " << this->kernelName << endl;

	//reset output matrix
	host_matrix_for_device_answer.reset();
	if (cudaMemcpy(device_C, host_matrix_for_device_answer.elements.get(), sizeof(float) * N * N, cudaMemcpyHostToDevice))
	{
		cout << "Error copying reset memory from host to device\n";
	}

	//time and run the kernel
	time_t start_time = clock();
	kernel << <GRID_DIMS, this->tileDims >> > (device_A, device_B, device_C);
	cudaDeviceSynchronize();
	time_t end_time = clock();

	cudaError_t cudaError = cudaGetLastError();
	if (cudaError) {
		cout << "CUDA error: " << cudaGetErrorString(cudaError) << endl;
	}

#ifdef PRINT_KERNEL_TIMES
	cout << ((float)(end_time - start_time)) / CLOCKS_PER_SEC << " seconds\n";
#endif

	//copy device results to host
	if (cudaMemcpy(host_matrix_for_device_answer.elements.get(), device_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost))
	{
		cout << "Error copying device memory to host memory\n";
	}

	//reorder the output matrix
	if (this->postKernelReordering == NO_REORDERING)
	{
		//no reordering necessary
	}
	else if (this->postKernelReordering == SINGLE_ROTATE)
	{
		host_matrix_for_device_answer.reorderSingleRotate();
	}
	else if (this->postKernelReordering == DOUBLE_ROTATE)
	{
		host_matrix_for_device_answer.reorderDoubleRotate();
	}
	

#ifdef PRINT_MATRICES
	host_matrix_for_device_answer.print(this->kernelName);
#endif

#ifdef CHECK_ANSWERS
	if (host_answer == host_matrix_for_device_answer)
	{
		cout << "results were equal\n";
	}
	else
	{
		cout << "results were not equal\n";
	}
#endif
}