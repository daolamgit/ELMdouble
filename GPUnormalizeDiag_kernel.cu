/********************************************************************
*  File: GPUnormalizeDiag.cu
*
*  Description: Force the entries of diagonal matrix to 1 to get a
*               left sided entity matrix.
*   
*	
*  Includes 1 Functions:
*    - normalizeDiag_kernel 
*  
*
*  Arguments: 
*	- double *diagMatrix    Input Matrix 1D 
*	- double *invMatrix     Invers Matrix 1D
*   - int size             Matrix dimension in size
*   - int row              The current Block offset of the row
*	
* Used custom Routines:
*	- 
*
*********************************************************************/
#ifndef NORMALIZEDIAG_CU
#define NORMALIZEDIAG_CU

#include "gpu_inverse.h"

__global__ void normalizeDiag_kernel (double *diagMatrix, double *invMatrix, int size,
                                      int row)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;

    int blockOffset = bx * BLOCKSIZE;
    __shared__ double diagEl[BLOCKSIZE];

    if (tx == ty)
    {
        diagEl[ty] = diagMatrix[row + ty * size + tx];
    }
    __syncthreads ();

    invMatrix[blockOffset + ty * size + tx] =
                            invMatrix[blockOffset + ty * size + tx] / diagEl[ty];
}

#endif
