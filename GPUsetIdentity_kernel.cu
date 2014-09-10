/********************************************************************
*  File: GPUnormalizeDiag.cu
*
*  Description: Generate an identitymatrix
*   
*	
*  Includes 1 Functions:
*    - GPUsetIdentity 
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

#ifndef GPUSETIDENTITY_KERNEL_H
#define GPUSETIDENTITY_KERNEL_H

#include "gpu_inverse.h"

__global__ void GPUsetIdentity (double* matrix,
                                int width)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int offset = bx * BLOCKSIZE + tx;
    matrix[offset * width + offset] = 1;
}

#endif
