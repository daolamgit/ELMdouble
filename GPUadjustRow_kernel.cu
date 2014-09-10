/********************************************************************
*  File: GPUadjustRow_kernel.cu
*
*  Description: 
*   adjust the rest of row with the pre-calculated pivot elements from step 1.
*   Also adjust the inverse block with the identity matrix.
*	
*  Includes 2 Functions:
*    - adjustColL_kernel
*    - adjustColU_kernel   
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
#ifndef GPUADJUSTROW_KERNEL_H
#define GPUADJUSTROW_KERNEL_H

#include "gpu_inverse.h"

__global__ void adjustRowL_kernel (double *dMatrixIn, double *dMatrixInDiag,
                                   double *dMatrixInv, int width, int diagEl)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;

    __shared__ double pivotBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];
    __shared__ double inBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];
    __shared__ double invBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];

    /*
     * Adjust the rest blocks which are right from the prepared block of step 1
     * and adjust the inverse blocks
     */
    if (bx * BLOCKSIZE > (diagEl + 1))
    {
        pivotBlock[ty][tx] = dMatrixInDiag[ty * width + tx];
        inBlock[ty][tx] = dMatrixIn[ty * width + bx * BLOCKSIZE +tx];
        invBlock[ty][tx] = dMatrixInv[ty * width + bx * BLOCKSIZE +tx];

        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif
        //i equals the current row where the pivot elements are stored
        for (int i = 0; i < BLOCKSIZEMINUS1; i++)
        {
            // if the cols are below  
            if (ty > i)
            {
                double pivot = pivotBlock[ty][i];
                //Subtract the row
                inBlock[ty][tx] -= inBlock[i][tx] * pivot;
                invBlock[ty][tx] -= invBlock[i][tx] * pivot;
            }

            __syncthreads ();
        }
        //Store the results back in device memory
        dMatrixIn[ty * width + bx * BLOCKSIZE +tx] = inBlock[ty][tx];
        dMatrixInv[ty * width + bx * BLOCKSIZE +tx] = invBlock[ty][tx];
    }
    /*
     * Adjust the last blocks from the indentity matrix which are left 
     */
    else
    {
        pivotBlock[ty][tx] = dMatrixInDiag[ty * width + tx];
        invBlock[ty][tx] = dMatrixInv[ty * width + bx * BLOCKSIZE +tx];

        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif
        for (int i = 0; i < BLOCKSIZEMINUS1; i++)//last changed
        {
            if (ty > i)
            {
                double pivot = pivotBlock[ty][i];

                invBlock[ty][tx] -= invBlock[i][tx] * pivot;
            }

            __syncthreads ();
        }
        dMatrixInv[ty * width + bx * BLOCKSIZE + tx] = invBlock[ty][tx];
    }
}



__global__ void adjustRowU_kernel (double *dMatrixIn, double *dMatrixInv, int width,
                                   int diagEl)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;

    __shared__ double pivotBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];
    __shared__ double invBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];

    pivotBlock[ty][tx] = dMatrixIn[ty * width + tx];
    invBlock[ty][tx] = dMatrixInv[ty * width + bx * BLOCKSIZE +tx];

    __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif
    for (int i = BLOCKSIZEMINUS1; i > 0; i--)
    {
        if (ty < i)
        {
            double pivot = pivotBlock[ty][i];

            invBlock[ty][tx] -= invBlock[i][tx] * pivot;
        }

        __syncthreads ();
    }

    dMatrixInv[ty * width + bx * BLOCKSIZE +tx] = invBlock[ty][tx];
}

#endif
