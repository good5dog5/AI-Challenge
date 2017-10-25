#pragma once
extern "C" {

/*
	[NumData, k]       = size(result)
	[NumData, NumDims] = size(mat)
	[NumData, 1]       = size(label)
*/
void kernel(int * __restrict result, const int k, const float * __restrict mat, const int * __restrict label, const int NumData, const int NumDims);

/*
	[NumTarget, k]       = size(result)
	[NumTarget, NumDims] = size(targetMat)
	[NumSource, NumDims] = size(sourceMat)
	[NumSource, 1]       = size(label)
*/
void predict(int * __restrict result  , const int k,
    const float * __restrict targetMat, const int NumTarget,
    const float * __restrict sourceMat, const int NumSource,
    const int * __restrict label      , const int NumDims);
}
