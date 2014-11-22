#ifndef MATRIX_H
#define MATRIX_H 

//the difficulty is I don't know where to start
#include <cublas_v2.h>
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"
#include "helper_string.h"

//#define SCALE 1e8
//#define COEFREG 1e-3 

//for transpose
#define TILE_DIM    BLOCKSIZE
#define BLOCK_ROWS  BLOCKSIZE
#define BLOCK_DIM BLOCKSIZE

#define BLOCKX 32
#define BLOCKY 8

typedef struct{ 
	int Width;
	int Height;
	double * Elements;
} Matrix;

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef DEBUGFILE
#define DEBUGFILE 0
#endif

#if (DEBUGFILE)
#define DBGF(_x) ((void)(_x))
#else
#define DBGF(_x) ((void)0)
#endif

#if (DEBUG)
#define DBG(_x) ((void)(_x))
#else
#define DBG(_x) ((void)0)
#endif


//# define BLOCK_SIZE 16

#define IDX2R1(i,j,ld) ((((i)-1)*(ld))+ ((j)-1))
#define IDX2C1(i,j,ld) ((((j)-1)*(ld))+ ((i)-1))
#define IDX2R0(i,j,ld) ((((i))*(ld))+ ((j)))
#define IDX2C0(i,j,ld) ((((j))*(ld))+ ((i)))
#define IDX2R(i,j,ld) (i*ld + j) //ld: length of row
#define IDX2C(i,j,ld) (i + j*ld)


//void compute_Wo( d_OutputWeight, d_H, d_T); //least mean square by GaussSeidel

//void LoadMatrix(Matrix & A);
void CublasMult(Matrix &, cublasHandle_t handle, const Matrix, const Matrix, bool, bool); //row dom mult
void MatrixMultCublas(double *pC, cublasHandle_t handle,  double *pA,  double *pB,
				const int WA, const int HA, const int WB, const int HB, const int WC, const int HC,
				bool T1, bool T2);

void CublasMultCol(Matrix &, cublasHandle_t handle, const Matrix, const Matrix, bool, bool); //col dom mult
void MatrixMultCublasCol(double *pC, cublasHandle_t handle,  double *pA,  double *pB,
				const int HA, const int WA, const int HB, const int WB, const int HC, const int WC,
				bool T1, bool T2);


void getLU(double *d_L, double *d_U, const double *A , const int N);
__global__ void getLUKernel(double *L, double *U, const double *A, const int N);

__global__ void matrixMulKernel( double* C, double* A, double* B, int wA, int wB, int HC);
__global__ void max_matrix(const double* Y, int * I, int R, int C);
void transpose(double *HT, const double* H, int size_x, int size_y);
__global__ void activate_fun(const char* fun, double * H, int N);
__global__ void activate_fun_matrix(const char* fun, double * A, int W, int H);
__global__ void kernelRegularize( double * A, int N, double *C);

__global__ void kernelInverseQuadricRBF( double * C, const double *A, const double *B,
							const int ld, const int nA, const int nB);
__global__ void kernelQuadricRBF( double * C, const double *A, const double *B,
							const int ld, const int nA, const int nB);


__global__ void kernelQuadricRBFFast( double * C, const double *A, const double *B,
							const int ld, const int nA, const int nB);
__device__ void samb2( double a, double *b, double *c);
							
__global__ void kernelGaussianRBF( double * C, const double *A, const double *B,
							const int ld, const int nA, const int nB);							

__global__ void vectorAddMatrix(double *C, double *A, double *B, int W, int H);
__global__ void vectorAdd(double *C, const double *A, const double * B, int N);
__global__ void vectorMinus(double *C, const double *A, const double * B, int N);

void check_device_memory(const double *d_p, int r, int c, const char *filename);
void check_host_memory(const double *d_p, int r, int c, const char *filename);
void check_device_memory(const double *d_p, int r, int c);
void check_device_memory(const int *d_p, int r, int c);

__global__ void transposeDiagonal (double *odata, const double *idata, int width, int height, int nreps);



__global__ void kernelScaleDown( double *A, int W, int H, double *);
__global__ void kernelScaleUp( double *A, int W, int H);
#endif
