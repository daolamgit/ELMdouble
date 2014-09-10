/********************************************************************
*  File: GPUGausSeidel.c
*
*  Description:
*	
*	Mainfunction to compute an inverse Matrix from a positive definite 
*   Matrix on the GPU. The Routine is using the Gaus Seidel Matrix invertion 
*   algorithm. 
*   
*   1   2   1   |  1  0   0               1   0   0  |  -2.5   1.5   0.5
*   2   3   1   |  0  1   0       =>      0   1   0  |   1.5  -0.5  -0.5 
*   1   1   2   |  0  0   1               0   0   1  |   0.5  -0.5   0.5 
*   Inputmatrix       E                       E          Inverse Matrix
*
*  Arguments: 
*	- double *hDataIn      Input Matrix 1D, no data changes
*   - double *hDataOut     Output Matrix 1D, the inverse datamatrix  
*   - int size            Matrix dimension in size, width = height = size
*	
*  Used custom kernels rutines:
*	- GPUsetIdentity          
*   - eliminateBlockL_kernel    
*   - adjustColL_kernel         
*   - eliminateColL_kernel      
*   - eliminateRestL_kernel     
*
*   - eliminateBlockU_kernel    
*   - adjustColU_kernel         
*   - eliminateColU_kernel      
*   - eliminateRestU_kernel     
*
*********************************************************************/

//#include <cutil.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "GPUGausSeidel.h"
#include "gpu_inverse.h"
#include "helper_cuda.h"
#include <stdio.h>





int GPUGausSeidel (double *hDataIn, double *hDataOut, int size)
{
    double *dDataIn;
    double *dDataInv;
    int i;
    int size2InBytes = size * size * sizeof (double);
  
    //Allocating memory for the datamatrix and identity matrix (Einheitsmatrix)
    cudaMalloc ((void **) &dDataIn, size2InBytes);
    cudaMalloc ((void **) &dDataInv, size2InBytes);
    
    //Prepare the calculation of the identitymatrix
    cudaMemset ((void *) dDataInv, 0, size2InBytes);
    //Transfair the matrix from host to device
    cudaMemcpy ((void *) dDataIn, (void *) hDataIn, size2InBytes,
                cudaMemcpyHostToDevice);

    //Used SP/MP for calculations
    dim3 idyThreads (BLOCKSIZE);    
    dim3 idyBlocks (size / BLOCKSIZE);
    dim3 nThreads (BLOCKSIZE, BLOCKSIZE);   
    dim3 nBlocks (size / BLOCKSIZE);        
    dim3 nBlocksRest (size / BLOCKSIZE, size / BLOCKSIZE);

    //Calculate the Identitymatrix 
    GPUsetIdentity <<< idyBlocks, idyThreads >>> (dDataInv, size);
    cudaDeviceSynchronize ();

    //calculate the right diagonal Matrix (L)
    for (i = 0; i < size; i += BLOCKSIZE)
    {
        int offset = i * size + i;

        /* step 1:
         *  calculate the triangle matrix
         *  store the pivot elements to left part of the triangel
         */

        eliminateBlockL_kernel <<< 1, nThreads >>> (dDataIn + offset, size);
        cudaDeviceSynchronize ();

        /* step 2:
         *  calculate the rest of the rows with the pivot elements from step 1
         *  
         */
        adjustRowL_kernel <<< nBlocks, nThreads >>> (dDataIn + i * size, dDataIn + offset,
                                                     dDataInv + i * size, size, i);
        cudaDeviceSynchronize ();


        /* step 3:
         *Fill the colls below the block with the pivot elements they are used
         *    to get the colls to zero and multiply with the row
         */
        eliminateColL_kernel <<< nBlocks, nThreads >>> (dDataIn + i, size, i);
        cudaDeviceSynchronize ();

        /* step 4:
         *  Adjust the rest of the Matrix with the calculated pivot Elements
         *  El_new_0 -= (p0+p1+p2..+p15) * El_piv_0
         */
        eliminateRestL_kernel <<< nBlocksRest, nThreads >>> (dDataIn, dDataInv, size, i);
        cudaDeviceSynchronize ();
    }

    //Set the left lower diagonalmatrix to zero (async?)
    for (i = 1; i < size; i++)
    {
        int offset = i * size;
        cudaMemset ((void *) (dDataIn + offset), 0, i * sizeof (double));
    }
    cudaDeviceSynchronize ();


    //calculate the right diagonal Matrix (U)
    for (i = (size - BLOCKSIZE); i >= 0; i -= BLOCKSIZE)
    {
        int offset = i * size + i;

        /* step 1:
         *  calculate the triangle matrix
         *  store the pivot elements to left part of the triangel
         */
        eliminateBlockU_kernel <<< 1, nThreads >>> (dDataIn + offset, size);
        cudaDeviceSynchronize ();

        /* step 2:
         *  calculate the rest of the rows with the pivot elements from step 1
         *  
         */
        adjustRowU_kernel <<< nBlocks, nThreads >>> (dDataIn + offset,
                                                     dDataInv + i * size, size, i);
        cudaDeviceSynchronize ();

        /* step 3:
         *  Fill the colls below the block with the pivot elements they are used
         *      to get the colls to zero and multiply with the row
         */
        eliminateColU_kernel <<< nBlocks, nThreads >>> (dDataIn + i, size, i);
        cudaDeviceSynchronize ();

        /* step 4:
         *  Adjust the rest of the Matrix with the calculated pivot Elements
         *  El_new_0 -= (p0+p1+p2..+p15) * El_piv_0
         */
        eliminateRestU_kernel <<< nBlocksRest, nThreads >>> (dDataIn, dDataInv, size, i);
        cudaDeviceSynchronize ();
    }
    
    /*
     * force the diagonal entries to 1
     */
    for (i = 0; i < size; i += BLOCKSIZE)
    {
        int rowOffset = i * size;
        normalizeDiag_kernel <<< nBlocks, nThreads >>> (dDataIn + rowOffset,
                                                        dDataInv + rowOffset, size, i);
        cudaDeviceSynchronize ();
    }


    cudaMemcpy ((void *) hDataOut, (void *) dDataInv, size2InBytes,
                cudaMemcpyDeviceToHost);

    cudaFree (dDataIn);
    cudaFree (dDataInv);

    return 0;
}

//an copy of GPUGausSeidel but doesn't have the memory transfer
//all memory reside in GPU but its not a kernel
int device_GPUGausSeidel (double *dDataIn, double *dDataInv, int size)
{
//    double *dDataIn;
//    double *dDataInv;
    int i;
    int size2InBytes = size * size * sizeof (double);
  
    //Allocating memory for the datamatrix and identity matrix (Einheitsmatrix)
    //cudaMalloc ((void **) &dDataIn, size2InBytes);
    //cudaMalloc ((void **) &dDataInv, size2InBytes);
    
    //Prepare the calculation of the identitymatrix
    cudaMemset ((void *) dDataInv, 0, size2InBytes);
    //Transfair the matrix from host to device
    //cudaMemcpy ((void *) dDataIn, (void *) hDataIn, size2InBytes,
    //            cudaMemcpyHostToDevice);

    //Used SP/MP for calculations
    dim3 idyThreads (BLOCKSIZE);    
    dim3 idyBlocks (size / BLOCKSIZE);
    dim3 nThreads (BLOCKSIZE, BLOCKSIZE);   
    dim3 nBlocks (size / BLOCKSIZE);        
    dim3 nBlocksRest (size / BLOCKSIZE, size / BLOCKSIZE);

    //Calculate the Identitymatrix 
    GPUsetIdentity <<< idyBlocks, idyThreads >>> (dDataInv, size);
    cudaDeviceSynchronize ();

    //calculate the right diagonal Matrix (L)
    for (i = 0; i < size; i += BLOCKSIZE)
    {
        int offset = i * size + i;

        /* step 1:
         *  calculate the triangle matrix
         *  store the pivot elements to left part of the triangel
         */

        eliminateBlockL_kernel <<< 1, nThreads >>> (dDataIn + offset, size);
        cudaDeviceSynchronize ();

        /* step 2:
         *  calculate the rest of the rows with the pivot elements from step 1
         *  
         */
        adjustRowL_kernel <<< nBlocks, nThreads >>> (dDataIn + i * size, dDataIn + offset,
                                                     dDataInv + i * size, size, i);
        cudaDeviceSynchronize ();


        /* step 3:
         *Fill the colls below the block with the pivot elements they are used
         *    to get the colls to zero and multiply with the row
         */
        eliminateColL_kernel <<< nBlocks, nThreads >>> (dDataIn + i, size, i);
        cudaDeviceSynchronize ();

        /* step 4:
         *  Adjust the rest of the Matrix with the calculated pivot Elements
         *  El_new_0 -= (p0+p1+p2..+p15) * El_piv_0
         */
        eliminateRestL_kernel <<< nBlocksRest, nThreads >>> (dDataIn, dDataInv, size, i);
        cudaDeviceSynchronize ();
    }

    //Set the left lower diagonalmatrix to zero (async?)
    for (i = 1; i < size; i++)
    {
        int offset = i * size;
        cudaMemset ((void *) (dDataIn + offset), 0, i * sizeof (double));
    }
    cudaDeviceSynchronize ();


    //calculate the right diagonal Matrix (U)
    for (i = (size - BLOCKSIZE); i >= 0; i -= BLOCKSIZE)
    {
        int offset = i * size + i;

        /* step 1:
         *  calculate the triangle matrix
         *  store the pivot elements to left part of the triangel
         */
        eliminateBlockU_kernel <<< 1, nThreads >>> (dDataIn + offset, size);
        cudaDeviceSynchronize ();

        /* step 2:
         *  calculate the rest of the rows with the pivot elements from step 1
         *  
         */
        adjustRowU_kernel <<< nBlocks, nThreads >>> (dDataIn + offset,
                                                     dDataInv + i * size, size, i);
        cudaDeviceSynchronize ();

        /* step 3:
         *  Fill the colls below the block with the pivot elements they are used
         *      to get the colls to zero and multiply with the row
         */
        eliminateColU_kernel <<< nBlocks, nThreads >>> (dDataIn + i, size, i);
        cudaDeviceSynchronize ();

        /* step 4:
         *  Adjust the rest of the Matrix with the calculated pivot Elements
         *  El_new_0 -= (p0+p1+p2..+p15) * El_piv_0
         */
        eliminateRestU_kernel <<< nBlocksRest, nThreads >>> (dDataIn, dDataInv, size, i);
        cudaDeviceSynchronize ();
    }
    
    /*
     * force the diagonal entries to 1
     */
    for (i = 0; i < size; i += BLOCKSIZE)
    {
        int rowOffset = i * size;
        normalizeDiag_kernel <<< nBlocks, nThreads >>> (dDataIn + rowOffset,
                                                        dDataInv + rowOffset, size, i);
        cudaDeviceSynchronize ();
    }


    //cudaMemcpy ((void *) hDataOut, (void *) dDataInv, size2InBytes,
    //            cudaMemcpyDeviceToHost);

    //cudaFree (dDataIn);
    //cudaFree (dDataInv);

    return 0;
}


//More general purpose of cuda Mult
double * mult2(const double *A, const double *B,
			int m, int n, int k, int a, int b, int c,
			int T1, int T2){
				return NULL;

}

