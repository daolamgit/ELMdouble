#ifndef GPUGAUSSEIDEL_H
#define GPUGAUSSEIDEL_H

//#ifdef __cplusplus
//   extern "C" {
//#endif





double * mult2(const double *A, const double *B,
			int m, int n, int k, int a, int b, int c,
			int T1, int T2);


void mult(double* C, const double *A, const double *B, int WA, int WB, int HC);

//void mult(double*, const double *A, const double *B,
//			int m, int n, int k,
//			int T1, int T2);

double * pinv(double *matrix, int m, int n); 

int device_GPUGausSeidel (double* dmatrix, 
                   double* doutput, 
                   int size);

int GPUGausSeidel (double* matrix, 
                   double* output, 
                   int size);



//void check_device_memory(double *d_p, int r, int c);
//void check_device_memory(int *d_p, int r, int c);
//void check_device_memory(double *d_p, int r, int c, const char *filename);


__global__ void adjustRowL_kernel (double *dMatrixIn, double *dMatrixInDiag,
                                   double *dMatrixInv, int width, int diagEl);
__global__ void adjustRowU_kernel (double *dMatrixIn, double *dMatrixInv, int width,
                                   int diagEl);

__global__ void eliminateBlockL_kernel (double *dInData, 
                                        int size);
__global__ void eliminateBlockU_kernel (double *dInData, int size);

__global__ void eliminateColL_kernel (double *dMatrixIn, int size, int diagEl);
__global__ void eliminateColU_kernel (double *dMatrixIn, int size, int diagEl);

__global__ void eliminateRestL_kernel (double *dMatrixIn, double *dMatrixInv, int size,
                                       int diagEl);
__global__ void eliminateRestU_kernel (double *dMatrixIn, double *dMatrixInv, int size,
                                       int diagEl);

__global__ void normalizeDiag_kernel (double *diagMatrix, double *invMatrix, int size,
                                      int row);
__global__ void GPUsetIdentity (double* matrix,
                                int width);


//#ifdef __cplusplus
//   }
//#endif

#endif
