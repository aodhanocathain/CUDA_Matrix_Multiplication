#pragma once

#define WARP_SIZE 32	//how many threads in a warp in CUDA

//used for the tiling kernel
//ensure the block output tiles remain square and cleanly divide the matrix
#define FRAGS_PER_THREAD 4
#define REPLICATION 2
#define THREADS_ACROSS 4
#define THREADS_DOWN 8
#define WARPS_ACROSS 4
#define WARPS_DOWN 2
#define TILED_KERNEL_BLOCK_WIDTH (REPLICATION*FRAGS_PER_THREAD*THREADS_ACROSS*WARPS_ACROSS)
#define BLOCK_ITEMS_K ((WARP_SIZE*WARPS_DOWN*WARPS_ACROSS)/TILED_KERNEL_BLOCK_WIDTH)

#define ROTATION_VECTOR_WIDTH 32
#define SQRT_ROTATION_VECTOR_WIDTH 6	//rounded up to nearest integer, used for double-rotation method

#define MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS 256	//parameter for problem size
#define N (ROTATION_VECTOR_WIDTH*MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS)	//matrix width

#define NUM_TESTS 1

//before compiling, uncomment any of the below that are desirable
#define RANDOMIZE_MATRICES
//#define PRINT_MATRICES
//#define CHECK_ANSWERS
#define PRINT_KERNEL_TIMES

enum reorder { NO_REORDERING, SINGLE_ROTATE, DOUBLE_ROTATE };	//options for reordering matrices after certain kernels