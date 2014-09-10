/********************************************************************
*  File: GPUeliminateRest_kernel.cu
*
*  Description: Adjust the rest of the entire inv-/matrix with the precalculated
*               pivot elements from step 3. 
*   
*	
*  Includes 2 Functions:
*    - eliminateRestL_kernel
*    - eliminateRestU_kernel   
*  
*
*  Arguments: 
*	- double *dMatrixIn      Input Matrix 1D 
*	- double *dMatrixInv     Invers Matrix 1D
*   - int size              Matrix dimension in size
*   - int diagEl            The adjusted blockoffset from step 1
*	
* Used custom Routines:
*	- 
*
*********************************************************************/
#ifndef ELIMINATE_REST_KERNEL
#define ELIMINATE_REST_KERNEL

#include "gpu_inverse.h"

__global__ void eliminateRestL_kernel (double *dMatrixIn, double *dMatrixInv, int size,
                                       int diagEl)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ double pivEl[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];
    __shared__ double pivBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];
    __shared__ double inBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];

    //rest of the unadjusted Matrix which is right above the diagEl
    if (bx * BLOCKSIZE > (diagEl + 1) && by * BLOCKSIZE > (diagEl + 1))
    {
        int blockOffset = by * BLOCKSIZE * size + bx * BLOCKSIZE;
        int blockPivElOffset = by * BLOCKSIZE * size + diagEl;
        int blockPivOffset = diagEl * size + bx * BLOCKSIZE;

        inBlock[ty][tx] = dMatrixIn[blockOffset + ty * size + tx];
        pivEl[ty][tx] = dMatrixIn[blockPivElOffset + ty * size + tx];
        pivBlock[ty][tx] = dMatrixIn[blockPivOffset + ty * size + tx];
        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif
        //Subtract each row from the input Matrix =>dMatrixIn
        for (int i = 0; i < BLOCKSIZE; i++)
        {
            inBlock[ty][tx] -= pivEl[ty][i] * pivBlock[i][tx];
        }
        __syncthreads ();
        dMatrixIn[blockOffset + ty * size + tx] = inBlock[ty][tx];
        __syncthreads ();

        inBlock[ty][tx] = dMatrixInv[blockOffset + ty * size + tx];
        pivBlock[ty][tx] = dMatrixInv[blockPivOffset + ty * size + tx];

        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif
        //Subtract each row from the invers Matrix =>dMatrixInv
        for (int i = 0; i < BLOCKSIZE; i++)
        {
            inBlock[ty][tx] -= pivEl[ty][i] * pivBlock[i][tx];
        }

        __syncthreads ();
        dMatrixInv[blockOffset + ty * size + tx] = inBlock[ty][tx];
    }
    //Adjust the left Blocks from the invers matrix which are left from the diagEl
    else if (by * BLOCKSIZE > (diagEl + 1))
    {
        int blockOffset = by * BLOCKSIZE * size + bx * BLOCKSIZE;
        int blockPivElOffset = by * BLOCKSIZE * size + diagEl;
        int blockPivOffset = diagEl * size + bx * BLOCKSIZE;

        pivEl[ty][tx] = dMatrixIn[blockPivElOffset + ty * size + tx];
        inBlock[ty][tx] = dMatrixInv[blockOffset + ty * size + tx];
        pivBlock[ty][tx] = dMatrixInv[blockPivOffset + ty * size + tx];
        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif

        for (int i = 0; i < BLOCKSIZE; i++)
        {
            inBlock[ty][tx] -= pivEl[ty][i] * pivBlock[i][tx];
        }

        __syncthreads ();
        dMatrixInv[blockOffset + ty * size + tx] = inBlock[ty][tx];
    }
}


__global__ void eliminateRestU_kernel (double *dMatrixIn, double *dMatrixInv, int size,
                                       int diagEl)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ double pivEl[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];
    __shared__ double pivBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];
    __shared__ double inBlock[BLOCKSIZE][BLOCKSIZE+AVOIDBANKCONFLICTS];

    //rest der unbearbeiteten Matrix bearbeiten
    if ((bx * BLOCKSIZE + 1) <diagEl && (by * BLOCKSIZE +1) <diagEl)     //linke seite von in; 0-pivblock
    {

        int blockOffset = by * BLOCKSIZE * size + bx * BLOCKSIZE;
        int blockPivElOffset = by * BLOCKSIZE * size + diagEl;
        int blockPivOffset = diagEl * size + bx * BLOCKSIZE;

        inBlock[ty][tx] = dMatrixIn[blockOffset + ty * size + tx];
        pivEl[ty][tx] = dMatrixIn[blockPivElOffset + ty * size + tx];
        pivBlock[ty][tx] = dMatrixIn[blockPivOffset + ty * size + tx];
        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif

        for (int i = BLOCKSIZEMINUS1; i >= 0; i--)
        {
            inBlock[ty][tx] -= pivEl[ty][i] * pivBlock[i][tx];
        }
        __syncthreads ();
        dMatrixIn[blockOffset + ty * size + tx] = inBlock[ty][tx];
        __syncthreads ();

        inBlock[ty][tx] = dMatrixInv[blockOffset + ty * size + tx];
        pivBlock[ty][tx] = dMatrixInv[blockPivOffset + ty * size + tx];

        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif

        for (int i = BLOCKSIZEMINUS1; i >= 0; i--)
        {
            inBlock[ty][tx] -= pivEl[ty][i] * pivBlock[i][tx];
        }

        __syncthreads ();
        dMatrixInv[blockOffset + ty * size + tx] = inBlock[ty][tx];
    }
    else if (by * BLOCKSIZE <(diagEl))
    {
        int blockOffset = by * BLOCKSIZE *size + bx * BLOCKSIZE;
        int blockPivElOffset = by * BLOCKSIZE *size + diagEl;
        int blockPivOffset = diagEl * size + bx * BLOCKSIZE;


        pivEl[ty][tx] = dMatrixIn[blockPivElOffset + ty * size + tx];
        inBlock[ty][tx] = dMatrixInv[blockOffset + ty * size + tx];
        pivBlock[ty][tx] = dMatrixInv[blockPivOffset + ty * size + tx];
        __syncthreads ();

#ifdef USELOOPUNROLLING
    #pragma unroll BLOCKSIZE
#endif

        for (int i = BLOCKSIZEMINUS1; i >= 0; i--)
        {
            inBlock[ty][tx] -= pivEl[ty][i] * pivBlock[i][tx];
        }

        __syncthreads ();
        dMatrixInv[blockOffset + ty * size + tx] = inBlock[ty][tx];
    }
}

#endif
