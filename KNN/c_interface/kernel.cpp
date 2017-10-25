#include "kernel.h"
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>

class neighbor
{
public:

	int id = -1;
	int label = -1;
	float distance = std::numeric_limits<float>::max();
};

float lossfunc(const float * __restrict target, const float * __restrict source, const int NumDims)
{
	float result = 0;

	//#pragma omp parallel for reduction(+:result)
	for (int ii = 0; ii < NumDims; ii = ii + 1)
	{
		float diff = (target[ii] - source[ii]);
		result = result + diff * diff;
	}

	return result;
}

void kernel(int * __restrict result, const int k, const float * __restrict mat, const int * __restrict label, const int NumData, const int NumDims)
{
	std::vector<neighbor> ArrayNeighbor( NumData * (k + 1) );

	#pragma omp parallel for schedule(guided)
	for (int ii = 0; ii < NumData; ii = ii + 1)
	{
		for (int jj = 0; jj < NumData; jj = jj + 1)
		{
			if (ii == jj)
			{
				continue;
			}

			int index = ii * (k + 1);
			ArrayNeighbor[index + k].id       = jj;
			ArrayNeighbor[index + k].label    = label[jj];
			ArrayNeighbor[index + k].distance = lossfunc(&mat[jj * NumDims], &mat[ii * NumDims], NumDims);
			std::sort(ArrayNeighbor.begin() + index, ArrayNeighbor.begin() + index + (k + 1), [](const neighbor& lhs, const neighbor& rhs)
			{
				return lhs.distance < rhs.distance;
			});

		}

		for (int jj = 0; jj < k; jj = jj + 1)
		{
			result[ii * k + jj] = ArrayNeighbor[ii * (k + 1) + jj].label;
		}
	}

}

void predict(int * __restrict result  , const int k,
    const float * __restrict targetMat, const int NumTarget,
    const float * __restrict sourceMat, const int NumSource,
    const int * __restrict label      , const int NumDims)
{
        std::vector<neighbor> ArrayNeighbor( NumTarget * (k + 1) );

        #pragma omp parallel for schedule(guided)
        for (int ii = 0; ii < NumTarget; ii = ii + 1)
        {
		// computing loss
                for (int jj = 0; jj < NumSource; jj = jj + 1)
                {

                        int index = ii * (k + 1);
                        ArrayNeighbor[index + k].id       = jj;
                        ArrayNeighbor[index + k].label    = label[jj];
                        ArrayNeighbor[index + k].distance = lossfunc(&targetMat[ii * NumDims], &sourceMat[jj * NumDims], NumDims);
                        std::sort(ArrayNeighbor.begin() + index, ArrayNeighbor.begin() + index + (k + 1), [](const neighbor& lhs, const neighbor& rhs)
                        {
                                return lhs.distance < rhs.distance;
                        });

                }

		// output
                for (int jj = 0; jj < k; jj = jj + 1)
                {
                        result[ii * k + jj] = ArrayNeighbor[ii * (k + 1) + jj].label;
                }
        }



}
