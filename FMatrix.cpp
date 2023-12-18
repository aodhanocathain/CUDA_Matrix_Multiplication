#include <memory>
using std::unique_ptr;
#include <string>
using std::string;
#include <iostream>
using std::cout;
using std::endl;

#include <time.h>
#include <stdlib.h>
#include <immintrin.h>

#include "FMatrix.hpp"

FMatrix::FMatrix(int numRows, int numCols)
{
	this->elements = unique_ptr<element[]>(new element[numRows * numCols]);
	this->numRows = numRows;
	this->numCols = numCols;
}

FMatrix FMatrix::operator*(FMatrix& multiplier)
{
	FMatrix result(this->numRows, multiplier.numCols);

	//All elements in any row of the output matrix are dependant on the same row of inputs from this matrix
	//All elements in any row of the output matrix are dependant on unique columns of inputs from the multiplier matrix
	//Can iterate through multiple columns of the multiplier matrix in parallel to calculate their dot product with their common row of this matrix.

	for (int row = 0; row < result.numRows; row++)
	{
		//vectorize iterations through columns
		for (int col = 0; col < 8 * (result.numCols / 8); col += 8)
		{
			__m256 sums = _mm256_set1_ps(0);	//dot products from different columns
			for (int k = 0; k < this->numCols; k++)
			{
				int thisIndex = (row * this->numCols) + k;
				//load the common element from this matrix into all lanes
				__m256 broadcastElement = _mm256_set1_ps(this->elements[thisIndex]);
				int multiplierIndex = (k * multiplier.numRows) + col;
				//load the unique column elements into their corresponding lanes
				__m256 vectorElements = _mm256_loadu_ps(&(multiplier.elements[multiplierIndex]));
				//multiply lanes, then add results to the running dot products
				sums = _mm256_add_ps(sums, _mm256_mul_ps(broadcastElement, vectorElements));
			}
			int index = (row * result.numCols) + col;
			//store the vector of results
			_mm256_storeu_ps(&(result.elements[index]), sums);
		}

		//calculate remaining elements that would not fill an entire vector
		for (int col = 8 * (result.numCols / 8); col < result.numCols; col++)
		{
			float sum = 0;
			for (int k = 0; k < this->numCols; k++)
			{
				int firstIndex = (row * this->numCols) + k;
				int secondIndex = (k * multiplier.numRows) + col;
				sum += this->elements[firstIndex] * multiplier.elements[secondIndex];
			}
			int index = (row * result.numCols) + col;
			result.elements[index] = sum;
		}
		if ((row % 80) == 0)
		{
			cout << row << '/' << result.numRows << endl;
		}
	}
	return result;
}

bool FMatrix::operator==(FMatrix& mat)
{
	//dimensions assumed to be equal

	float error_tolerance = 0.05f;
	//check if any elements differ
	for (int row = 0; row < this->numRows; row++)
	{
		for (int col = 0; col < this->numCols; col++)
		{
			int index = (row * this->numCols) + col;
			float ratio = mat.elements[index] / this->elements[index];
			if (ratio < (1 - error_tolerance) || ratio >(1 + error_tolerance)) { return false; }
		}
	}

	//elements are all equal --> matrices are equal
	return true;
}

void FMatrix::reset()
{
	for (int i = 0; i < this->numRows * this->numCols; i++)
	{
		this->elements[i] = 0.0f;
	}
}

void FMatrix::transpose()
{
	for (int row = 0; row < this->numRows; row++)
	{
		for (int col = row + 1; col < this->numCols; col++)
		{
			element swap = this->elements[(row * this->numCols) + col];
			this->elements[(row * this->numCols) + col] = this->elements[(col * this->numCols) + row];
			this->elements[(col * this->numCols) + row] = swap;
		}
	}
}

void FMatrix::randomize()
{
	srand(clock());
	for (int row = 0; row < this->numRows; row++)
	{
		for (int col = 0; col < this->numCols; col++)
		{
			int index = (row * this->numCols) + col;
			this->elements[index] = rand() % 4;
		}
	}
}

void FMatrix::print(string name)
{
	cout << name << endl;
	for (int row = 0; row < this->numRows; row++)
	{
		for (int col = 0; col < this->numCols; col++)
		{
			int index = (row * this->numCols) + col;
			cout << this->elements[index] << '\t';
		}
		cout << endl;
	}
}

void FMatrix::reorderSingleRotate()
{
	//the kernel(s) using the single rotate method calculates the correct elements but orders them incorrectly
	//this function reorders the matrix by moving each element from where it went to where it should have gone

	element* correctOrdering = new element[this->numRows * this->numCols];
	//retrace the steps of each thread block to move results to where they should have gone
	for (int blockIdxY = 0; blockIdxY < MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS; blockIdxY++)
	{
		for (int blockIdxX = 0; blockIdxX < MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS; blockIdxX++)
		{
			//retrace the steps of the thread block at the given indices
			int tileTop = blockIdxY * ROTATION_VECTOR_WIDTH;
			int tileLeft = blockIdxX * ROTATION_VECTOR_WIDTH;
			for (int threadIdxX = 0; threadIdxX < ROTATION_VECTOR_WIDTH; threadIdxX++)
			{
				int col = tileLeft + threadIdxX;
				for (int row = 0; row < ROTATION_VECTOR_WIDTH; row++)
				{
					int oldRow = tileTop + row;
					int newRow = tileTop + ((row + col) % ROTATION_VECTOR_WIDTH);
					correctOrdering[(newRow * N) + col] = this->elements[(oldRow * N) + col];
				}
			}
		}
	}
	this->elements = unique_ptr<element[]>(correctOrdering);
}

void FMatrix::reorderDoubleRotate()
{
	//the kernel(s) using double rotation calculates the correct elements but orders them incorrectly
	//this function reorders the matrix by moving each element from where it went to where it should have gone

	element* correctOrdering = new element[this->numRows * this->numCols];
	//retrace the steps of each thread block to move results to where they should have gone
	for (int blockIdxY = 0; blockIdxY < MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS; blockIdxY++)
	{
		for (int blockIdxX = 0; blockIdxX < MATRIX_WIDTH_IN_ROTATION_VECTOR_LENGTHS; blockIdxX++)
		{
			//retrace the steps of the thread block at the given indices
			int tileTop = blockIdxY * ROTATION_VECTOR_WIDTH;
			int tileLeft = blockIdxX * ROTATION_VECTOR_WIDTH;
			for (int threadIdxX = 0; threadIdxX < ROTATION_VECTOR_WIDTH; threadIdxX++)
			{
				int oldCol = tileLeft + threadIdxX;	//going down the thread's column of results
				for (int ALeft = 0; ALeft < ROTATION_VECTOR_WIDTH; ALeft += SQRT_ROTATION_VECTOR_WIDTH)
				{
					for (int BRight = 0; BRight < SQRT_ROTATION_VECTOR_WIDTH; BRight++)
					{
						int i = ALeft + BRight;
						if (i >= ROTATION_VECTOR_WIDTH) { break; }

						int oldRow = tileTop + i;

						int ALane = (ROTATION_VECTOR_WIDTH + threadIdxX + ALeft) % ROTATION_VECTOR_WIDTH;
						int BLane = (ROTATION_VECTOR_WIDTH + threadIdxX - BRight) % ROTATION_VECTOR_WIDTH;

						int newRow = tileTop + ALane;
						int newCol = tileLeft + BLane;

						correctOrdering[(newRow * N) + newCol] = this->elements[(oldRow * N) + oldCol];
					}
				}
			}
		}
	}
	this->elements = unique_ptr<element[]>(correctOrdering);
}