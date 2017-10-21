#pragma once
extern "C" {

/*
	[NumData, k]       = size(result)
	[NumData, NumDims] = size(mat)
	[NumData, 1]       = size(label)
*/
void kernel(int * __restrict result, const int k, const float * __restrict mat, const int * __restrict label, const int NumData, const int NumDims);

}