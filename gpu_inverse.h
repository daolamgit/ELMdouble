#include <cublas_v2.h>
#include <stdio.h>
#include "helper_cuda.h"

#ifndef gpu_inverse_H
#define gpu_inverse_H




//for Mul
//#define BLOCK_SIZE 10

//#define BLOCKSIZE 4
//#define BLOCKSIZEMINUS1 3

//#define USELOOPUNROLLING 1  
//#define AVOIDBANKCONFLICTS 0    //this just runs faster :X

//global variable just for this CUBLAS
//I'LL write another function for multiplying
//extern cublasHandle_t handle;
//extern cublasStatus_t status;




//#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)
//
//    inline void __checkCudaErrors( cudaError_t err, const char *file, const int line )
//    {
//        if( cudaSuccess != err) {
//		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
//                    file, line, (int)err, cudaGetErrorString( err ) );
//            exit(-1);
//        }
//    }

	#define checkCublasErrors(err)           __checkCublasErrors (err, __FILE__, __LINE__)

    inline void __checkCublasErrors( cublasStatus_t err, const char *file, const int line )
    {
		if( CUBLAS_STATUS_SUCCESS != err) {
		    fprintf(stderr, "%s(%i) : CUBLAS Runtime API error %d: %s.\n",
                    file, line, (int)err, _cudaGetErrorEnum( err ) );
            exit(-1);
        }
    }


inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUGFILE
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

#endif
