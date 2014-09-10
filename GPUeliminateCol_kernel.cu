/********************************************************************
*  File: GPUeliminateCol_kernel.cu
*
*  Description: Calculate each pivot elements above/over (L/U) the block
*               from step 1. And fill The blocks with them
*   
*	
*  Includes 2 Functions:
*    - eliminateColL_kernel
*    - eliminateColU_kernel   
*  
*
*  Arguments: 
*	- double *dMatrixIn      Input Matrix 1D pointed to the current row, on the device 
*   - int size              Matrix dimension in size
*   - int diagEl            The adjusted blockoffset from step 1
*	
* Used custom Routines:
*	- 
*
*********************************************************************/
#ifndef ELIMINATE_COL_KERNEL
#define ELIMINATE_COL_KERNEL

#include "gpu_inverse.h"

__global__ void eliminateColL_kernel (double *dMatrixIn, int size, int diagEl)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //bx is used to adress the Blocks above the precalculated block from step 1
    int bx = blockIdx.x;    

    //only the blocks can enter which are above the precalculated block from step 1
    if (bx * BLOCKSIZE > (diagEl + 1))
    {
        int offset = diagEl * size;
        int blockOffset = bx * BLOCKSIZE *size;

        __shared__ double pivotBlock[BLOCKSIZE][BLOCKSIZE];
        __shared__ double inBlock[BLOCKSIZE][BLOCKSIZE];

        pivotBlock[ty][tx] = dMatrixIn[offset + ty * size + tx];   // The Block from step 1
        inBlock[ty][tx] = dMatrixIn[blockOffset + ty * size + tx]; // each Block which is above the pivotBlock

        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif
        //iterate through the block und calculate the pivot elements
        for (int i = 0; i < BLOCKSIZE; i++)
        {
            double pivotEl = inBlock[ty][i] / pivotBlock[i][i];

            __syncthreads ();

            //adjust all values right to the current interation step
            if (tx > i)
            {
                //substract the row
                inBlock[ty][tx] -= pivotBlock[i][tx] * pivotEl;
            }
            //store the pivot element in the col
            else
            {
                inBlock[ty][i] = pivotEl;
            }

            __syncthreads ();
        }

        dMatrixIn[blockOffset + ty * size + tx] = inBlock[ty][tx];
    }
}


__global__ void eliminateColU_kernel (double *dMatrixIn, int size, int diagEl)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;

    if (bx * BLOCKSIZE <diagEl)
    {
        int offset = diagEl * size;
        int blockOffset = bx * BLOCKSIZE *size;

        __shared__ double pivotBlock[BLOCKSIZE][BLOCKSIZE];
        __shared__ double inBlock[BLOCKSIZE][BLOCKSIZE];

        pivotBlock[ty][tx] = dMatrixIn[offset + ty * size + tx];
        inBlock[ty][tx] = dMatrixIn[blockOffset + ty * size + tx];

        __syncthreads ();

#ifdef USELOOPUNROLLING
#pragma unroll BLOCKSIZE
#endif

        for (int i = BLOCKSIZEMINUS1; i >= 0; i--)
        {
            double pivotEl = inBlock[ty][i] / pivotBlock[i][i];

            __syncthreads ();

            if (tx < i)
            {
                inBlock[ty][tx] -= pivotBlock[i][tx] * pivotEl;
            }
            else/* if (tx == i)*/
            {
                inBlock[ty][i] = pivotEl;
            }

            __syncthreads ();
        }

        dMatrixIn[blockOffset + ty * size + tx] = inBlock[ty][tx];
    }
}

#endif
