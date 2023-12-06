#pragma once

#include <memory>
using std::unique_ptr;
#include <string>
using std::string;

#include "Constants.hpp"

//current code for the * operator and for cuBLAS test assumes single precision floating point elements 
typedef float element;

//a class for matrices of single precision floats
class FMatrix
{
	public:

		unique_ptr<element[]> elements;
		int numRows;
		int numCols;

		FMatrix(int numRows, int numCols);

		FMatrix operator*(FMatrix& mat);
		bool operator==(FMatrix& mat);

		void reset();

		void transpose();
		void randomize();

		void reorderSingleRotate();
		void reorderDoubleRotate();

		void print(string name);
};