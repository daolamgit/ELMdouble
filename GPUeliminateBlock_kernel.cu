/********************************************************************
*  File: GPUeliminateBlock_kernel.cu
*
*  Description: 
*   Calculate the Gaus-Seidel algorithm on a small BLOCK of the Inputmatrix. 
*   To benefit from the Shared Memory the BLOCK is sized to BLOCKSIZE = 16.
*	
*  Includes 2 Functions:
*    - eliminateBlockL_kernel
*    - eliminateBlockU_kernel   
*  
*
*  Arguments: 
*	- double *dInData      Input Matrix 1D, on the device
*   - int size            Matrix dimension in size
*	
* Used custom Routines:
*	- 
*
*********************************************************************/


#ifndef ELIMINATEBLOCK_H
#define ELIMINATEBLOCK_H

#include "gpu_inverse.h"

__global__ void eliminateBlockL_kernel (double *dInData, 
                                        int size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double triangleBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];

    triangleBlock[ty][tx] = dInData[ty * size + tx];
    __syncthreads ();

    
#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZEMINUS1
#endif
    //i equals the current row
    for (int i = 0; i < BLOCKSIZEMINUS1; i++)
    {
        // calculate the pivot element to get the current row i to zero
        double pivotEl = triangleBlock[ty][i] / triangleBlock[i][i];

        __syncthreads ();       // Each pivotEl have to be calculated and store in the registers

        if (ty > i)             // If all cols (ty) are below the current row (i)?
        {
            if (tx > i)         // The element is right to the current row, subtract the element
            {
                triangleBlock[ty][tx] -= pivotEl * triangleBlock[i][tx];
            }
            if (tx == i)        // Store the pivot element in the current row
            {
                triangleBlock[ty][tx] = pivotEl;
            }
        }
        __syncthreads ();       // Wait for each thread
    }

    dInData[ty * size + tx] = triangleBlock[ty][tx];       // Write the result back to memory
}


__global__ void eliminateBlockU_kernel (double *dInData, int size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double triangleBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];

    triangleBlock[ty][tx] = dInData[ty * size + tx];
    __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZEMINUS1
#endif    
    //i equals the current row
    for (int i = BLOCKSIZEMINUS1; i > 0; i--)
    {
        // calculate the pivot element to get the current row i to zero
        double pivotEl = triangleBlock[ty][i] / triangleBlock[i][i];

        __syncthreads ();       // Each pivotEl have to be calculated and store in the registers

        if (ty < i)             // If all rows (ty) are above the current row (i)?
        {
            if (tx < i)         // The element is left to the current row, subtract the element
            {
                triangleBlock[ty][tx] -= pivotEl * triangleBlock[i][tx];
            }
            if (tx == i)        // Store the pivot element in the current row
            {
                triangleBlock[ty][tx] = pivotEl;
            }
        }
        __syncthreads ();        // Wait for each thread
    }

    dInData[ty * size + tx] = triangleBlock[ty][tx];       //Write the result back to device memory
}

#endif
