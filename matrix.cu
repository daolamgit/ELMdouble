#include "matrix.h"

//in C I use Row to store memory unless specify by the library
//and start 0 
//void LoadMatrix(Matrix & A){
// int H = A.Height;
// int W = A.Width;
// double* data = A.Elements;
//
// for (int r=0;r<H;r++)
//  for (int c=0;c<W;c++)
//   data[IDX2R0(r,c,W)] = 2* ((double)rand() /RAND_MAX) -1;
//	//data[IDX2R0(r,c,W)] = 1;
//
//}

//--------------------------------------------------------
void CublasMult(Matrix &C, cublasHandle_t handle, const Matrix A, const Matrix B, bool T1, bool T2){
//Edit on Apr. 6th: lengthy explain below can be summarized in 1 sentence: with the cublas matrix representation, is it what you want?
	//If not, transpose it (them).
	//Remember the memory is fixed, only the representation changes.
//Input:
//A: device matrix 
//B: device matrix 
//A, B is stored as row dominant, just like  math representation
//However: cublas is Col dominant so we must transpose before 
//using cublas
//Consider: A, B must be transpose so that they can multiply
//AFter they are transposed they are translated into Cublas with consideration that it will be transposed when cublas read the memory. So the OPERAND is used to retain to the original order.
//The result is also transpose due to cublas, so actually we compute the transpose of the multiplication.
//The matrix in the parameter is the matrix at their C represenatation NOT CUBLAS REPRESENATION.
cublasStatus_t status;
double alpha = 1.0;
double beta = 0.0;



//mxk kxn 
//mxk nxk
//kxm kxn
//kxm nxk

//tom lai la cublas la transpose cua math representation nen cu transpose het la duoc
//Edit April 6: The thought direction of C mem storage of row make me think in like above. Its
//not wrong but it complicates the problem. If I think C mem matrix storage as a linked list with fixed segment, that
//will help me to go straight to the problem. Remember 2 things: get data the right order and place in the right order
//e.g: 4x4 X 4x4 can be formated to 2x8 X 8x2, lda= 4 but m= 2. Although I don't think of any app for this flexiblity, but it's good
//to have
if (T1)
              	if (T2)
				//wrong:math: mxk kxn qua cublas ->math: kxm nxk. Do do phai transpose
				//m,n, k: number of row of op(A), number of columns of op(B), number of col of op(A)
				
				//A = kxm B = nxk
				//C = AT*BT mxn
				//CT = B*A
				//mult happen in cublas: nxk kxm
                        status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                B.Height, A.Width , A.Height, &alpha,
                                                B.Elements, B.Width,
                                                A.Elements, A.Width, &beta,
                                                C.Elements, B.Height);
                else
			//kxm kxn
			//C = AT*B
			//CT = BT*A nxk kxm
			//mult in cublas: 
                        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                B.Width, A.Width , A.Height, &alpha,
                                                B.Elements, B.Width,
                                                A.Elements, A.Width, &beta,
                                                C.Elements, B.Width);
        else
                if (T2)
			//mxk nxk
			//C = A*BT
			//CT = B*AT nxk kxm
			//mult in cublas: 
                        status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                B.Height, A.Height , A.Width, &alpha,
                                                B.Elements, B.Width,
                                                A.Elements, A.Width, &beta,
                                                C.Elements, B.Height);
                else
				
				//This is the start
				//C = AB: mxk kxn
				//CT = (AB)T = BT*AT
				//mult happen in cublas: nxk kxm
				//all of them auto transpose because of cublas

                        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                B.Width, A.Height , A.Width, &alpha,
                                                B.Elements, B.Width,
                                                A.Elements, A.Width, &beta,
                                                C.Elements, B.Width);

if (status!= CUBLAS_STATUS_SUCCESS){
	printf("Error in CUBLAS Multiplying Matrix %s \n", _cudaGetErrorEnum(status));
	exit(EXIT_FAILURE);
}
cudaDeviceSynchronize();

}

void MatrixMultCublas(double *pC, cublasHandle_t handle,  double *pA,  double *pB,
				const int WA, const int HA, const int WB, const int HB, const int WC, const int HC,
				bool T1, bool T2){
	//A remdedy for Matrix arrangement nuisance before Cublas Mult
	Matrix MC, MA, MB;

	MA.Width = WA;
	MA.Height = HA;
	MA.Elements = pA;

	MB.Width = WB;
	MB.Height = HB;
	MB.Elements = pB;

	MC.Width =WC;
	MC.Height =HC;
	MC.Elements = pC;

	CublasMult(MC, handle, MA, MB, T1, T2);
}

//------------------------------------------------------------------------------------------------
void CublasMultCol(Matrix &C, cublasHandle_t handle, const Matrix A, const Matrix B, bool T1, bool T2){
//Edit on Apr. 6th: lengthy explain below can be summarized in 1 sentence: with the cublas matrix representation, is it what you want?
	//If not, transpose it (them).
	//Remember the memory is fixed, only the representation changes.
//Input:
//A: device matrix
//B: device matrix
//A, B is stored as row dominant, just like  math representation
//However: cublas is Col dominant so we must transpose before
//using cublas
//Consider: A, B must be transpose so that they can multiply
//AFter they are transposed they are translated into Cublas with consideration that it will be transposed when cublas read the memory. So the OPERAND is used to retain to the original order.
//The result is also transpose due to cublas, so actually we compute the transpose of the multiplication.
//The matrix in the parameter is the matrix at their C represenatation NOT CUBLAS REPRESENATION.
cublasStatus_t status;
double alpha = 1.0;
double beta = 0.0;



//mxk kxn
//mxk nxk
//kxm kxn
//kxm nxk

//tom lai la cublas la transpose cua math representation nen cu transpose het la duoc
//Edit April 6: The thought direction of C mem storage of row make me think in like above. Its
//not wrong but it complicates the problem. If I think C mem matrix storage as a linked list with fixed segment, that
//will help me to go straight to the problem. Remember 2 things: get data the right order and place in the right order
//e.g: 4x4 X 4x4 can be formated to 2x8 X 8x2, lda= 4 but m= 2. Although I don't think of any app for this flexiblity, but it's good
//to have
if (T1)
              	if (T2)

				//A = kxm B = nxk
				//C = AT*BT mxn
				//CT = B*A
				//mult happen in cublas: nxk kxm
                        status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                A.Width, B.Width , A.Height, &alpha,
                                                A.Elements, A.Height,
                                                B.Elements, B.Height, &beta,
                                                C.Elements, C.Height);
                else
			//kxm kxn
			//C = AT*B
			//CT = BT*A nxk kxm
			//mult in cublas:
                        status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                A.Width, B.Width , A.Height, &alpha,
                                                A.Elements, A.Height,
                                                B.Elements, B.Height, &beta,
                                                C.Elements, C.Height);
        else
                if (T2)
			//mxk nxk
			//C = A*BT
			//CT = B*AT nxk kxm
			//mult in cublas:
                        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                A.Height, B.Height , A.Width, &alpha,
                                                A.Elements, A.Height,
                                                B.Elements, B.Height, &beta,
                                                C.Elements, C.Height);
                else

				//This is the start
				//C = AB: mxk kxn
				//all of them auto transpose because of cublas

                        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,

                                                A.Height, B.Width , A.Width, &alpha,
                                                A.Elements, A.Height,
                                                B.Elements, B.Height, &beta,
                                                C.Elements, C.Height);

if (status!= CUBLAS_STATUS_SUCCESS){
	printf("Error in CUBLAS Multiplying Matrix %s \n", _cudaGetErrorEnum(status));
	exit(EXIT_FAILURE);
}
cudaDeviceSynchronize();

}


void MatrixMultCublasCol( double *pC, cublasHandle_t handle,  double *pA,  double *pB,
				const int HA, const int WA, const int HB, const int WB, const int HC, const int WC,
				bool T1, bool T2){
	//A remdedy for Matrix arrangement nuisance before Cublas Mult
	Matrix MC, MA, MB;

	MA.Width = WA;
	MA.Height = HA;
	MA.Elements = pA;

	MB.Width = WB;
	MB.Height = HB;
	MB.Elements = pB;

	MC.Width =WC;
	MC.Height =HC;
	MC.Elements = pC;

	CublasMultCol(MC, handle, MA, MB, T1, T2);
}

//--------------------------------------------------------------------------------
__global__ void matrixMulKernel( double* C, const double* A, const double* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
 
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
 
    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = wA * BLOCK_SIZE * by;
 
    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + wA - 1;
 
    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = BLOCK_SIZE * bx;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;
 
	double Csub = 0;
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
    {

        // Declaration of the shared memory array As 
        // used to store the sub-matrix of A
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
 
        // Declaration of the shared memory array Bs 
        // used to store the sub-matrix of B
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
 
        // Load the matrices from global memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
 
        // Synchronize to make sure the matrices 
        // are loaded
        __syncthreads();
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

}
//-------------------------------------------------------------------------------
void transpose(double *HT, const double* H, int size_x, int size_y){
	dim3 grid(size_x / BLOCK_DIM, size_y / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
	transposeDiagonal<<< grid, threads >>>(HT, H, size_x, size_y,1);
}
//------------------------------------------------------------------------
__global__ void max_matrix(const double* Y, int * I, int R, int C){
	//Input: Matrix Y: RxC
	//Output: I: index of the max along the column
	int id;

	id = blockIdx.x * blockDim.x + threadIdx.x;
	//find max for this column id
	if (id < C){
		double M = -1e9;
		for (int k = 0; k<R; k++)
			if (M<Y[id*R+k]){
				M = Y[id*R+k];
				I[id] = k;
			}
	}


}
//-------------------------------------------------------------------
void mult(double* C, const double *A, const double *B, int WA, int WB, int HC){
	//Input:
	//C, A, B: all device matrix
	//A: HA x WA; B = HBx WB
/*
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	int WC = WB;
	dim3 grid(WC / threads.x, HC / threads.y);
	matrixMul<<< grid, threads >>>(C, A, 
                                  B, WA, WB);

								  */
}


void getLU(double *d_L, double *d_U, const double *A , const int N){
	dim3 Block(BLOCKSIZE,BLOCKSIZE);
	dim3 Grid(N/Block.x, N/Block.y);

	getLUKernel <<<Grid,Block>>> (d_L, d_U, A, N);
	//check_cuda_errors(__FILE__, __LINE__);
}

//kernel cal for getLU
__global__ void getLUKernel(double *L, double *U, const double *A, const int N){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if (row > col) {//for L
		int aIndex = col*N + row; //index in A matrix
		int lIndex  = N*col - col*( col +1)/2 + row ; //index in L vector
		L[ lIndex] 	= A[ aIndex];
		return;
	}

	if (row <col) {//for U
		int aIndex = col*N + row; //index in A matrix
		/// wrong ful idea: int uIndex = N* row -row*(row+1)/2 + col;
		int uIndex = row + col*( col+1)/2; //it goes col first,
		U[ uIndex] = A[ aIndex];
		return;
	}

	//row = col
	int aIndex = col*N + row; //index in A matrix
	int lIndex  = N*col - col*( col +1)/2 + row ; //index in L vector
	int uIndex = row + col*( col+1)/2; //it goes col first
	L[ lIndex] 	= A[ aIndex];
	U[ uIndex]	= 0;

}

//-----------------------------------------------------------------------------------------
__global__ void activate_fun_matrix(const char* fun, double * A, int W, int H){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int index = row * W + col;

	if (col< W && row <H)
		A[index] = 1/(1+exp(-A[index]));;
}

__global__ void kernelScaleDown( double *A, int W, int H, double * Scale)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	
	int index = row * W + col;
	if (col < W && row < H)
		A[index]  	/= *Scale;
} 

__global__ void kernelScaleUp( double *A, int W, int H) //never use
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	
	int index = row * W + col;
	if (col < W && row < H)
		A[index]  	*= 1;
} 

__global__ void activate_fun(const char* fun, double * H, int N){
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < N)
			//logsig
			//H[i] = 1/(1+exp(-H[i]));

			//tansig
			H[i] = 2/(1+exp(-2*H[i])) -1;
}


__global__ void kernelRegularize (double * A, int N, double *COEFREG)
{
	int ThreadsperSM = blockDim.x;
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
	if ( ThreadsperSM* bx+ tx < N)
		A[ IDX2R0( ThreadsperSM * bx+ tx, ThreadsperSM*bx+ tx, N)] +=  *COEFREG;//* A[0]; //add .1% of A[0]0] instead

}

//--------------------------------------------------------------------------------------
__global__ void kernelGaussianRBF( double * C, const double *A, const double *B,
							const int ld, const int nA, const int nB)
{
int ty	= threadIdx.y;
int tx 	= threadIdx.x;

int r 	= blockIdx.y* BLOCKSIZE+ threadIdx.y;
int c 	= blockIdx.x* BLOCKSIZE+ threadIdx.x;
double 	p = 0;

//int NB 	= ld/ BLOCKSIZE;

__shared__ double Sr[BLOCKSIZE][BLOCKSIZE]; //fix ty, move tx
__shared__ double Sc[BLOCKSIZE][BLOCKSIZE]; //fix tx, move ty

for (int b=0; b< (BLOCKSIZE+ ld -1)/BLOCKSIZE; b++)//each block of matrix data
	{
	if (b*BLOCKSIZE+ tx <ld && r < nA)
		Sr[ty][tx]	= A[ IDX2R( r, b*BLOCKSIZE+ tx, ld) ];
	else
		Sr[ty][tx] 	= 0;
		
	if (b*BLOCKSIZE+ ty< ld && c < nB)		
		Sc[tx][ty] 	= B[ IDX2R( c, b*BLOCKSIZE+ ty, ld)];
	else 
		Sc[tx][ty] 	= 0;	
		
		__syncthreads();		
		
	for (int k=0; k< BLOCKSIZE; k++) //each element in the block
			//p 	+= Sr[tx][k]* Sc[ty][k];
			p 	+= (Sr[ty][k] - Sc[tx][k])* (Sr[ty][k] - Sc[tx][k]) ;//bvector[ b*BLOCKSIZE+ k]*bvector[ b*BLOCKSIZE+ k];
			
		__syncthreads();
		
	}
if (r < nA && c < nB)		
	C[IDX2R( r, c, nB )] 	= exp( -p);									
}

//---------------------------------------------------------------------------------------

__global__ void kernelInverseQuadricRBF( double * C, const double *A, const double *B,
							const int ld, const int nA, const int nB)
{
int ty	= threadIdx.y;
int tx 	= threadIdx.x;

int r 	= blockIdx.y* BLOCKSIZE+ threadIdx.y;
int c 	= blockIdx.x* BLOCKSIZE+ threadIdx.x;
double 	p = 0;

//int NB 	= ld/ BLOCKSIZE;

__shared__ double Sr[BLOCKSIZE][BLOCKSIZE]; //fix ty, move tx
__shared__ double Sc[BLOCKSIZE][BLOCKSIZE]; //fix tx, move ty

for (int b=0; b< (BLOCKSIZE+ ld -1)/BLOCKSIZE; b++)//each block of matrix data
	{
	if (b*BLOCKSIZE+ tx <ld && r < nA)
		Sr[ty][tx]	= A[ IDX2R( r, b*BLOCKSIZE+ tx, ld) ];
	else
		Sr[ty][tx] 	= 0;
		
	if (b*BLOCKSIZE+ ty< ld && c < nB)		
		Sc[tx][ty] 	= B[ IDX2R( c, b*BLOCKSIZE+ ty, ld)];
	else 
		Sc[tx][ty] 	= 0;	
		
		__syncthreads();		
		
	for (int k=0; k< BLOCKSIZE; k++) //each element in the block
			//p 	+= Sr[tx][k]* Sc[ty][k];
			p 	+= (Sr[ty][k] - Sc[tx][k])* (Sr[ty][k] - Sc[tx][k]) ;//bvector[ b*BLOCKSIZE+ k]*bvector[ b*BLOCKSIZE+ k];
			
		__syncthreads();
		
	}
if (r < nA && c < nB)		
	C[IDX2R( r, c, nB )] 	= 1/sqrt( p);	//can optimize in the futre								
}

__global__ void kernelQuadricRBF( double * C, const double *A, const double *B,
							const int ld, const int nA, const int nB)
{
int ty	= threadIdx.y;
int tx 	= threadIdx.x;

int r 	= blockIdx.y* BLOCKSIZE+ threadIdx.y;
int c 	= blockIdx.x* BLOCKSIZE+ threadIdx.x;
double 	p = 0;

//int NB 	= ld/ BLOCKSIZE;

__shared__ double Sr[BLOCKSIZE][BLOCKSIZE]; //fix ty, move tx
__shared__ double Sc[BLOCKSIZE][BLOCKSIZE]; //fix tx, move ty

for (int b=0; b< (BLOCKSIZE+ ld -1)/BLOCKSIZE; b++)//each block of matrix data
	{
	if (b*BLOCKSIZE+ tx <ld && r < nA)
		Sr[ty][tx]	= A[ IDX2R( r, b*BLOCKSIZE+ tx, ld) ];
	else
		Sr[ty][tx] 	= 0;
		
	if (b*BLOCKSIZE+ ty< ld && c < nB)		
		Sc[tx][ty] 	= B[ IDX2R( c, b*BLOCKSIZE+ ty, ld)];
	else 
		Sc[tx][ty] 	= 0;	
		
		__syncthreads();		
		
	for (int k=0; k< BLOCKSIZE; k++) //each element in the block
			//p 	+= Sr[tx][k]* Sc[ty][k];
			p 	+= (Sr[ty][k] - Sc[tx][k])* (Sr[ty][k] - Sc[tx][k]) ;//bvector[ b*BLOCKSIZE+ k]*bvector[ b*BLOCKSIZE+ k];
			
		__syncthreads();
		
	}
if (r < nA && c < nB)		
	C[IDX2R( r, c, nB )] 	= sqrt( p);	//can optimize in the futre								
}

__global__ void vectorAddMatrix(double *C, double *A, double *B, int W, int H)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int index = row * W + col;

	if (col< W && row <H)
		C[index] = A[index]+ B[index];
}

__global__ void vectorMinus(double *C, const double *A, const double * B, int N){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i <N)
	{
        C[i] = A[i] - B[i];
    }
}

__global__ void vectorAdd(double *C, const double *A, const double * B, int N){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}
//-----------------------------------------------------------------------------------
void check_device_memory(const double *d_p, int r, int c, const char *filename){
	size_t SizeMem;

	FILE * fp;

	fp = fopen(filename, "w");

	if (fp==NULL){
		printf("Can open file %s to write \n",filename);
		exit(-1);
	}

	//THERE IS A RISK SO I REWREITE THIS

	//col  IS THE DOMINANT DIMENSION, which is r a leading
	SizeMem = r*c* sizeof(double);
	double *h_p = (double *)malloc(SizeMem);
	
	//cublasGetVector(Size,sizeof(double),d_p,1,h_p,1); //spacing storage is useless
	checkCudaErrors(cudaMemcpy(h_p, d_p, SizeMem,cudaMemcpyDeviceToHost));
	//c = 1;

	//printf("content :\n");

	for (int i=0;i<r;i++){

			for (int j=0;j<c;j++)

			fprintf(fp,"%2.4f ",h_p[IDX2C0(i,j,r)]);

			fprintf(fp,"\n");

		}

	free(h_p);

	fclose(fp);

}
//-------------------------------------------------------
void check_device_memory(const int *d_p, int r, int c){
	//c = 1;
	//THERE IS A RISK SO I REWREITE THIS
	//col  IS THE DOMINANT DIMENSION, which is r a leading
	int *h_p = (int *)malloc(r*c*sizeof(int));
	cublasGetVector(r*c,sizeof(int),d_p,1,h_p,1); //spacing storage is useless

	printf("content :\n");
	for (int j=0;j<c;j++){
			for (int i=0;i<r;i++)
			DBG(printf("%d ",h_p[IDX2C(i,j,r)]));
			DBG(printf("\n"));
		}
	free(h_p);
}
//-----------------------------------------------------------------
void check_device_memory(const double *d_p, int r, int c){

	//THERE IS A RISK SO I REWREITE THIS
	//col  IS THE DOMINANT DIMENSION, which is r a leading
	double *h_p = (double *)malloc(r*c*sizeof(double));
	cublasGetVector(r*c,sizeof(double),d_p,1,h_p,1); //spacing storage is useless
	//c = 1;
	printf("content :\n");
	for (int i=0;i<r;i++){
			for (int j=0;j<c;j++)
			DBG( printf("%f ",h_p[IDX2R(i,j,c)]));
			DBG(printf("\n"));
		}
	free(h_p);
}

//--------------------------------------------------------------------------------------------
void check_host_memory(const double *d_p, int r, int c, const char *filename){
	

	FILE * fp;

	fp = fopen(filename, "w");

	if (fp==NULL){

		printf("Can open file %s to write \n",filename);

		exit(-1);

	}



	//row  IS THE DOMINANT DIMENSION, which is r a leading

	
	for (int i=0;i<r;i++){

			for (int j=0;j<c;j++)

			fprintf(fp,"%2.4f ",d_p[IDX2C0(i,j,r)]);

			fprintf(fp,"\n");

		}

	

	fclose(fp);

}

//----------------------------------------------------------------------
__global__ void transposeDiagonal(double *odata, const double *idata, int width, int height, int nreps)
{
  __shared__ double tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }    

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;  
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
  
    __syncthreads();
  
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}

