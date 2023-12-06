
#pragma once

#include <string>
using std::string;

#include "cuda_runtime.h"

#include "FMatrix.hpp"

class KernelTest
{
public:

	void (*kernel)(element* device_A, element* device_B, element* device_C);
	string kernelName;
	dim3 tileDims;	//the tile dimensions for launching the kernel
	enum reorder postKernelReordering;	//indicates how to reorder the output matrix

	KernelTest(void (*kernel)(element* device_A, element* device_B, element* device_C), string kernelName, dim3 tileDims, enum reorder postKernelReordering);
	void run(element* device_A, element* device_B, element* device_C, FMatrix& host_matrix_for_device_answer, FMatrix& host_answer);
};