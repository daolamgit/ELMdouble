#include 	"elm.h"
#include 	<stdio.h>
#include 	<string>
#include 	<algorithm>

#include 	"matrix.h"
#include 	"cuda.h"
#include 	"cuda_runtime.h"
#include 	<cublas_v2.h>
#include 	"helper_cuda.h"
#include 	"device_launch_parameters.h"

#include 	"GPUGausSeidel.h"
#include 	"gpu_inverse.h"

//cuda related function

//global variables
cublasHandle_t handle;
//cublasStatus_t status;


//---------------------------------------------------------------------
CElm::CElm( const char * f1, const char *f2, int C, int F, int Tr, int Te, int NHidden, const char* type){
printf("Object Initialized \n");
NoFeature = F;
NoTrain = Tr;
NoTest = Te;

NoClass = C;

//neural network type
strcpy( nnType, type); 
bRegularize 	= false; 	
COEFREG 	= 0;//1e-3; //fraction of 1;
checkCudaErrors (cudaMalloc( (void**)&dSCALE, sizeof( double) )); //for scale
checkCudaErrors (cudaMalloc( (void**)&dCOEFREG, sizeof( double) )); //for regular
checkCudaErrors (cudaMemcpy( dCOEFREG, &COEFREG, sizeof( double), cudaMemcpyHostToDevice));

int 		nDevices;

cudaGetDeviceCount( &nDevices) ;
for (int i=0; i< nDevices; i++){
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, i)!= cudaSuccess){
	printf("Device Error \n");
	break;
	}
	printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);


ThreadsperBlock = prop.maxThreadsPerBlock;
ThreadsperSM 	= prop.maxThreadsPerMultiProcessor;
	printf("Max thread per block :%d \n", ThreadsperBlock);
	printf("Max thread per MP: %d \n", ThreadsperSM);
}

size_t 		size_mem;

///////////malloc Train feature
//for feature
size_mem 			= (NoFeature+1) * NoTrain * sizeof(double); //include bias as feature 1
train = (double*) malloc(size_mem);
//check malloc
if (train == NULL){
	printf( "Host mem alloc fail Train \n");
	exit(-1);
}

//for label
size_mem 			= NoTrain*sizeof(double);//for
train_label 		= (double*) malloc(size_mem);
//check label
if (train_label == NULL){
	printf("Host mem malloc fail Train \n");
	exit(-1);
}
//read from file
load(f1, train_label, train, NoFeature, NoTrain);


/////////similarly for Test feature//////////////////
//for feature
size_mem 			= (NoFeature+1) *NoTest*sizeof(double);
test 				= (double*) malloc(size_mem);
//check malloc
if (test == NULL){
	printf("Host mem alloc fail Test \n");
	exit(-1);
}
//for label
size_mem 			= NoTest*sizeof(double);//for
test_label = (double*) malloc(size_mem);
//check label
if (test_label == NULL){
	printf("Host mem malloc fail Test \n");
	exit(-1);
}
//read from file
load(f2, test_label, test, NoFeature, NoTest);


printf("Class label: \n");
for (int i=0; i<NoTrain;i++)
	DBG( printf( "%lf ", train_label[i]));

//create the cublas handle that is used throught the program
cublasCreate(&handle);

size_mem =  NoTrain * NoClass *sizeof(double);
train_label_matrix = (double *) malloc(size_mem);
if (train_label_matrix == NULL){
		printf("label matrix mem maloc fail\n");
		exit(-1);
	}
	convert(train_label_matrix, train_label, NoTrain, NoClass);

}

//----------------------------------------------------------------------
CElm::~CElm(){

printf("Object deleted \n");
//cublasDestroy(handle);
}

//----------------------------------------------------------------------
void CElm::load(const char * filename, double *label, double * feature, const int NoFeature, const int NoSample){
//double pointer here is just for reference call, not for matrix manipulation

	double temp;//1 feature
	//int NoFeature;
	FILE 		*fid	= fopen(filename,"r");
	if (fid == NULL) {
		perror("Error: ");
		return ;
	}

	//value of pointer feature is the address of the malloced memory
	//feature = (double*) malloc(NoFeature*NoSample*sizeof(double));
	//label = (int*) malloc(NoSample*sizeof(int));
	printf( "Begin loading file \n");
	for (int i=0;i<NoSample;i++) {
		//read the label
		if (fscanf(fid,"%lf ",&temp) <= 0 )
			{
			printf ("Unexpected end of file at %d \n", i);
			exit( -1);
			}
		//change the value at the right memory
		*(label+i)	 = temp;
		//then read the features
		for (int j=0;j<NoFeature;j++)
			//change the value at the right memory
			{
			fscanf(fid,"%lf ", &temp);	
			*(feature+(i* (NoFeature +1) +j)) 	= temp;
			//fscanf(fid,"%lf ",feature+(i* (NoFeature +1) +j));
			}

		//add the feature 1
		if (strstr( nnType, "sig")!=NULL) //summation NN type
			*(feature+ (i+1) *(NoFeature +1) -1) 	= 1.0;
		else if 	(strstr( nnType, "rbf")!=NULL ) //rbf NN type
			*(feature+ (i+1) *(NoFeature +1) -1) 	= 0.0;
			else
				{
					printf("Wrong Neural network type \n");
					exit(-1);
				}
	}

	printf( "<< End of loading file \n");
	fclose(fid);
}

void CElm::convert(double * label_matrix, const double * label, const int No, const int NoClass){
	//input: vector label: 0->n
	//output: matrix label: -1 and 1 if classification, else the same as label
	//cublasHandle_t handle;
	//cublasStatus_t status;

	//if NoClass = 1 ->regression; just copy the label into label_matrix
	if (NoClass ==1){
		for (int i=0; i<No;i++)
			label_matrix[i] = (double)label[i];
		return;
	}

	for (int i=0;i<NoClass*No;i++)
		label_matrix[i] 	= -1;


	//set the right class =1
	int right_class;
	for (int i=0;i<No;i++){
		right_class 		= (int)*(label+i);
		label_matrix[IDX2C(right_class,i,NoClass)] 		= 1;
	}
	//just see Idx as number of leading dimenson + the position in that leading dim


	for (int i=0;i<No;i++){
	for (int j=0;j<NoClass;j++)
		//printf("%lf ",*(*label+IDX2C(i,j,No))); //to test the index
		DBG( printf("%lf ",label_matrix[IDX2R(i,j,NoClass)]));
	DBG( printf("\n"));
	}

	//free label vector
	//free(label_double);

}


/////////////////Function of Gauss//////////////////////////////////////
void CElm :: compute_Wo( double *d_W,  double *d_H,  double *d_T){
	//Purpose: solve for W from WH = T
	//H*Ht* Wot = H* Tt;

	///////compute H*Ht i.e.//square matrix
	double		*d_HHt;
	size_t		size_mem;
	size_mem			=	NoHidden * NoHidden *sizeof(double);
	checkCudaErrors( cudaMalloc( (void**)& d_HHt, size_mem));
	 
	MatrixMultCublasCol( d_HHt, handle, d_H, d_H,
						 NoHidden, NoTrain, 
						 NoHidden, NoTrain,
						 NoHidden, NoHidden,
						 0, 1);
	//check error of HHt multiply
	check_cuda_errors(__FILE__, __LINE__);
	DBGF( check_device_memory( d_HHt, NoHidden, NoHidden,"ckc.HHt"));


	////////compute H*Tt////////////////
	double			*d_HTt;
	size_mem		=	NoHidden * NoClass * sizeof(double);
	checkCudaErrors( cudaMalloc( ( void**)& d_HTt, size_mem)); //malloc device mem

	printf( "Compute HTt \n");
	//multiply now
	MatrixMultCublasCol( d_HTt, handle, d_H, d_T,
						NoHidden, NoTrain,
						NoClass, NoTrain,
						NoHidden, NoClass,
						0, 1);
	//check error
	check_cuda_errors(__FILE__, __LINE__);
	DBGF( check_device_memory( d_HTt, NoHidden, NoClass,"ckc.HTt"));


	//just the 2 of us
	///////////////////scale the 2 matrix to make it converge faster
	/////find max
	int iMax;
	double Max;
	cublasIdamax( handle, NoHidden* NoHidden, d_HHt, 1, &iMax);
	check_cuda_errors(__FILE__, __LINE__);

	//get the value to host
	checkCudaErrors( cudaMemcpy(&Max, d_HHt+ iMax, sizeof(double), cudaMemcpyDeviceToHost));
	Max 			= 1.0/Max;

	//check Max
	printf("Core dump %d %lf: \n", iMax, Max) ;
	//then scale A
	cublasDscal (handle, NoHidden* NoHidden, &Max, d_HHt, 1);
	check_cuda_errors(__FILE__, __LINE__);
	//then scale B
	cublasDscal (handle, NoHidden* NoClass, &Max, d_HTt, 1);
	check_cuda_errors(__FILE__, __LINE__);

	//check_device_memory( d_HHt, NoHidden, NoHidden,"ckc.HHt");
	//check_device_memory( d_HTt, NoClass, NoHidden,"ckc.HTt");

	////////solve HHt* Wot = HTt;/////////////////
	//Solve for c=1; i.e. regression//
	//solve(W1, A, b); //with W1 is hx1 col of Wot or 1xh row of Wo
	//and b is hx1 col of HTt

	//////////compute L* and U
	double 	*d_L, *d_U;
	size_t 	memSize;
	memSize 		= NoHidden*(NoHidden+1)/2* sizeof(double);
	checkCudaErrors( cudaMalloc( (void**)&d_L, memSize ));
	checkCudaErrors( cudaMalloc( (void**)&d_U, memSize ));
	getLU(d_L, d_U, d_HHt , NoHidden);
	//check memory
	DBGF( check_device_memory( d_L, 1, NoHidden*(NoHidden+1)/2,"ckc.L"));
	DBGF( check_device_memory( d_U, 1, NoHidden*(NoHidden+1)/2,"ckc.U"));

	////////////////////////
	cudaFree( d_HHt); //free for more mem
	////////////////////////
	double *x, *b; //for Ax =b

	////loop through the number of class
	//Span through multipl GPUs
	for (int i=0; i< NoClass; i++){
		printf( "Computing OutputWeight of class %d: \n", i);
		x		= d_W + i* NoHidden; //so Wo will be store with NoHidden vectors,
		//remember to transpose Wo when multiply
		b		= d_HTt + i* NoHidden;
		solveGaussSeidel(x, d_L, d_U, b, NoHidden);

		//check x for W0

		//copy to Wo
		//checkCudaErrors( cudaMemcpy() )
	}
	////end of loop

}

void CElm :: solveGaussSeidel(double *x, double *d_L, double *d_U, double *b, const int N){ //don't put const becuase of matrix mult
	//Self explain: A matrix NxN, x is result
	//format to Gauss Seidel
	//A = L + U;

	double 	tor 	= 1;
	int 	iter 	= 0;
	//transform into Blas form
	//L* x = b-Ux;

	double *x0; //old x
	checkCudaErrors( cudaMalloc( (void**)&x0, N*sizeof (double) ));

	//initialize x by b
	checkCudaErrors( cudaMemcpy(x, b, N* sizeof(double), cudaMemcpyDeviceToDevice));
	DBGF( check_device_memory( x, 1, N,"ckc.b"));

	/////////////////////repeat until converge///////////////////////////////////////
	do{
		iter 	+= 1;
		//copy x new to x
		checkCudaErrors( cudaMemcpy(x0, x, N* sizeof(double), cudaMemcpyDeviceToDevice));
		DBGF( check_device_memory( x, 1, N,"ckc.x"));

		//compute Ux;
		cublasDtpmv (handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
					 N, d_U, x, 1); //result is saved to x
		//check error
		DBGF( check_device_memory( x, 1, N,"ckc.Ux"));

		//compute b- Ux
		int BlocksperGrid = (N + ThreadsperBlock -1)/ThreadsperBlock;
		vectorMinus <<<BlocksperGrid, ThreadsperBlock>>> (x, b, x, N); //x1 = b-Ux
		check_cuda_errors(__FILE__, __LINE__);
		//check error
		DBGF( check_device_memory( x, 1, N,"ckc.b-Ux"));

		/////////////////solve/////////////////////////
		cublasDtpsv( handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
					N, d_L, x, 1); //solve L*x = b-Ux
		check_cuda_errors(__FILE__, __LINE__);

		//check error
		DBGF( check_device_memory( x, 1, N,"ckc.x1"));

		///////check convergence
		vectorMinus <<<BlocksperGrid, ThreadsperBlock>>> (x0, x, x0, N); //x = x1 -x
		DBGF( check_device_memory( b, 1, N,"ckc.x1-x"));

		double err, norm;
		//computer error
		cublasDnrm2 (handle, N, x0, 1, &err);
		cublasDnrm2 (handle, N, x, 1, &norm);
		check_cuda_errors(__FILE__, __LINE__);

		if (!(iter % 1000))
				printf("Error at iteration %d: %lf \n", iter, err );

		//for check error
		//DBGF( check_device_memory( b, 1, N,"ckc.b"));

//		if (iter ==2)
//			break;

		tor = err/ norm; //toreant of error
	} while (tor >1e-4);

	///////////////////////////////until no change
	//DBGF( check_device_memory( x, 1, NoHidden,"ckc.solveGauss"));
	//release
	cudaFree(x0);
}
////////////////End of Gauss


void CElm::run_train_GaussSeidel(){
	//convert from vector label to matrix label
	size_t size_mem;
	size_t avail, total;

	//////////For computing time//////////////////////

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);



	////////////////Training/////////////////////////////////
	//compute H for training
	size_mem = NoHidden* NoTrain* sizeof(double);
	H = (double *)malloc(size_mem);
	if (H==NULL){
		perror("Fail to malloc H %n: \n");
		exit(-1);
	}
	checkCudaErrors(cudaMalloc( (void**)&d_H, size_mem ));

	//compute H
	compute_H(d_H, train,NoTrain);
	//copy d_h to RAM H, to use later when compute Wo*H
	checkCudaErrors(cudaMemcpy(H, d_H, size_mem, cudaMemcpyDeviceToHost ));
	printf("TempH activation\n");
	DBGF(check_device_memory(d_H,NoHidden,NoTrain,"ckc.H"));


	//////////for OutputWeight//////////////////////////
	//Outputweight = dpinvH*T;
	//for label
	double 	*d_T;
	size_mem 		= NoTrain * NoClass * sizeof(double);
	checkCudaErrors(cudaMalloc( (void**)&d_T, size_mem));
	checkCudaErrors(cudaMemcpy (d_T, train_label_matrix, size_mem, cudaMemcpyHostToDevice));
	//check d_T
	//DBGF( check_device_memory(d_T, NoTrain, NoClass,"ckc.T"));

	//for OutputWeigth device memory
	size_mem 		= NoHidden * NoClass * sizeof(double);
	cudaMemGetInfo(&avail, &total);
	printf("Before malloc d_OutputWeight: Avail mem: %ld, total mem: %ld \n", avail,total);
	checkCudaErrors(cudaMalloc( (void**)&d_OutputWeight, size_mem));
	cudaMemGetInfo(&avail, &total);
	printf("After malloc d_OutputWeight: Avail mem: %ld, total mem: %ld \n", avail,total);

	////compute WoT/////////////////
	d_OutputWeightT  = d_OutputWeight;
	compute_Wo( d_OutputWeightT, d_H, d_T); //i.e. solve for WoH = T
	printf("OutputWeight \n");
	//Sadly, d_outputweight is indeed the transpose, because of the way the problem is format
	DBGF(check_device_memory(d_OutputWeightT,  NoHidden, NoClass, "ckc.OutputWeightT"));

	/////////////////////REcord time //////////////////////////////////////
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float fTrT;
	cudaEventElapsedTime(&fTrT,start,stop);
	TrT 	= (double)fTrT;
	cudaEventDestroy(start);
	cudaEventDestroy(stop) ;


	/////////////////////train output////////////////////////////
	//malloc H again
	//size_mem = NoHidden* NoTrain* sizeof(double);

	//cudaMemGetInfo(&avail, &total);
	//printf("Avail mem before malloc H to compute T %ld\n",avail);
	//checkCudaErrors(cudaMalloc( (void**)&d_H, size_mem ));
	//checkCudaErrors( cudaMemcpy(d_H, H, size_mem, cudaMemcpyHostToDevice));

	//mult(d_T, d_H, d_OutputWeight, NoHidden, NoClass, NoTrain);

	//change this back to working
	MatrixMultCublasCol( d_T, handle, d_OutputWeightT, d_H,
						 NoHidden, NoClass,
						 NoHidden, NoTrain,
						 NoClass, NoTrain,
						 1, 0);
	check_cuda_errors(__FILE__, __LINE__);
	//train_output_matrix = mult2(d_OutputWeight, H, NoClass, NoTrain ,NoHidden, 1, 0, 0, 1,0);
	//check output
	printf("Train Label Output:\n");
	DBGF(check_device_memory(d_T, NoClass, NoTrain,"ckc.TrainLabelcomputed"));
	cudaFree(d_H);

	//check accuray of training
	TrA 	= compute_accuracy_device(d_T, train_label, NoClass, NoTrain);
	printf("Accuracy training %lf\n ",TrA);

	//free train memory
	cudaFree(d_T);
	free(H);
	
}


//--------------------------------------------------------------------------
void CElm::run_train(){
	//convert from vector label to matrix label
	size_t size_mem;
	size_t avail, total;

	//////////For computing time//////////////////////

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);


	////////////////Training/////////////////////////////////
	//compute H for training
	size_mem = NoHidden* NoTrain* sizeof(double);
	H = (double *)malloc(size_mem);
	if (H==NULL){
		perror("Fail to malloc H %n: \n");
		exit(-1);
	}
	checkCudaErrors(cudaMalloc( (void**)&d_H, size_mem ));

	//compute H
	compute_H(d_H, train,NoTrain);

	//copy d_h to RAM H
	checkCudaErrors(cudaMemcpy(H, d_H, size_mem, cudaMemcpyDeviceToHost ));
	printf("TempHtrain activation\n");
	DBGF(check_device_memory(d_H,NoHidden,NoTrain,"ckc.H"));

	//square matrix
	double *d_square;
	size_mem = NoHidden * NoHidden *sizeof(double);
	checkCudaErrors(cudaMalloc( (void**)&d_square, size_mem ));
	 
	//multiply
	printf("Compute HTH \n");
	//mult(d_square, d_HT, d_H, NoTrain, NoHidden ,NoHidden);
	Matrix MSquare, MH;

	MSquare.Width = NoHidden; MSquare.Height = NoHidden; MSquare.Elements = d_square;

	MH.Width = NoHidden; MH.Height = NoTrain; MH.Elements = d_H;

	//MInputWeight.Width = NoHidden; MInputWeight.Height = NoFeature; MInputWeight.Elements = d_InputWeight;
	CublasMult(MSquare, handle, MH, MH,1,0); 

	check_cuda_errors(__FILE__, __LINE__);
	//setup executions params
	//check error
	DBGF(check_device_memory(d_square, NoHidden, NoHidden,"ckc.HTH"));

	
	if (bRegularize)///SINCE i USE DOUBLE, DON'T NEED TO SCALE DOWN
		{
		//scale down due to large element of HTH
		////////////////////////////////////////////
		checkCudaErrors( cudaMemcpy( dSCALE, d_square, sizeof( double), cudaMemcpyDeviceToDevice)); //cpy to kernel
		int Wid = NoHidden;
		int Hei = NoHidden;
		dim3 Block(BLOCKSIZE,BLOCKSIZE);
		//dim3 Block(1, ThreadsperBlock); //to maximize the number of block
		dim3 Grid( (BLOCKSIZE-1+ Wid)/Block.x, (BLOCKSIZE-1+ Hei)/Block.y);
		kernelScaleDown<<<Grid,Block>>>(d_square, Wid, Hei, dSCALE);
		check_cuda_errors(__FILE__, __LINE__);	
		//////////////////////////////////////////////	
		//end of scale


		//regularlization
		////////////////////////
		dim3 block 	= ThreadsperBlock;
		dim3 grid 	= NoHidden/ThreadsperBlock+ 1;
		
		kernelRegularize <<<grid, block >>>( d_square, NoHidden, dCOEFREG );//coefReg just a legacy, no need
		check_cuda_errors(__FILE__, __LINE__);
		DBGF(check_device_memory(d_square, NoHidden, NoHidden,"ckc.HTHScale+C"));
		//end of regularlization
		}
	
	//inverse
	double *d_square_inv;
	//see how much memory avail
	cudaMemGetInfo(&avail, &total);
	printf("Before d_square_inv: Avail mem: %ld, total mem: %ld \n", avail,total);

	//malloc invser matrix
	checkCudaErrors (cudaMalloc((void**)&d_square_inv,size_mem));
	printf("Inverting HTH \n");
	if (device_GPUGausSeidel(d_square, d_square_inv, NoHidden)){
		check_cuda_errors(__FILE__, __LINE__);
		printf("Error inverse AA' \n");
		exit(-1);
	}

	if (bRegularize)
		{
		int Wid = NoHidden;
		int Hei = NoHidden;
		dim3 Block(BLOCKSIZE,BLOCKSIZE);
		//dim3 Block(1, ThreadsperBlock); //to maximize the number of block
		dim3 Grid( (BLOCKSIZE-1+ Wid)/Block.x, (BLOCKSIZE-1+ Hei)/Block.y);
		////scale up again after inversion
		kernelScaleDown<<<Grid,Block>>>(d_square_inv, Wid, Hei, dSCALE); //still scale DOWN, tricky
		check_cuda_errors(__FILE__, __LINE__);	
		/////////////////////////////////////////////////////
		}

	//free square
	DBGF(check_device_memory(d_square_inv, NoHidden, NoHidden,"ckc.HH1"));
	cudaFree(d_square);

	//malloc d_pinv
	double *d_pinvH;
	size_mem 			= NoHidden* NoTrain* sizeof(double);
	checkCudaErrors(cudaMalloc( (void**)&d_pinvH, size_mem ));

	//mult to get d_pinv
	//mult(d_pinvH, d_H, d_square_inv, NoTrain,NoHidden, NoHidden, 1, 0);
	//multiply
	printf("Compute (HTH)-1*HT \n");
	//mult(d_square, d_H, d_HT, NoHidden, NoHidden ,NoTrain, 0 ,1);
	//mult(d_pinvH, d_square_inv, d_HT , NoHidden, NoTrain, NoHidden);
	Matrix MPinvH, MSquareInv;
	MSquareInv.Height 	= NoHidden; MSquareInv.Width = NoHidden; MSquareInv.Elements = d_square_inv;
	MPinvH.Width 		= NoTrain; MPinvH.Height = NoHidden; MPinvH.Elements = d_pinvH;

	CublasMult(MPinvH, handle, MSquareInv, MH, 0, 1);
	check_cuda_errors(__FILE__, __LINE__);
	//check
	DBGF( check_device_memory(d_pinvH, NoTrain, NoHidden, "ckc.pinvH"));

	//free
	//cudaFree(d_HT);
	cudaFree(d_square_inv);
	//////////////End of Pinv //////////////////////////////////

	//////////for OutputWeight//////////////////////////
	//Outputweight = dpinvH*T;
	//for label
	double *d_T;
	size_mem = NoTrain * NoClass * sizeof(double);
	checkCudaErrors(cudaMalloc( (void**)&d_T, size_mem));
	checkCudaErrors(cudaMemcpy (d_T, train_label_matrix, size_mem, cudaMemcpyHostToDevice));
	//check d_T
	DBGF( check_device_memory(d_T, NoClass, NoTrain,"ckc.T"));

	//for OutputWeigth
	size_mem = NoHidden * NoClass * sizeof(double);
	cudaMemGetInfo(&avail, &total);
	printf("Before malloc d_OutputWeight: Avail mem: %ld, total mem: %ld \n", avail,total);
	checkCudaErrors(cudaMalloc( (void**)&d_OutputWeight, size_mem));
	cudaMemGetInfo(&avail, &total);
	printf("After malloc d_OutputWeight: Avail mem: %ld, total mem: %ld \n", avail,total);
	//d_OutputWeight = compute_weight(d_pinvH, train_label_matrix); //I should make this matrix in device
	//result is a transpose of d_Ouput

	//int BLOCKCLASS = 2;
	//mult(d_OutputWeight, d_pinvH, d_T, NoTrain, NoClass, NoHidden);
	Matrix MOutputWeight, MT;
	MT.Width = NoClass; MT.Height = NoTrain; MT.Elements = d_T;
	MOutputWeight.Width = NoClass; MOutputWeight.Height = NoHidden; MOutputWeight.Elements = d_OutputWeight;

	CublasMult( MOutputWeight, handle, MPinvH, MT, 0, 0);

	check_cuda_errors(__FILE__, __LINE__);
	//CHECK
	printf("OutputWeight \n");
	//DBG(check_device_memory(d_OutputWeight, NoHidden, NoClass));
	DBGF(check_device_memory(d_OutputWeight, NoClass, NoHidden, "ckc.OutputWeight"));
	//free pinvH
	cudaFree(d_pinvH);


	/////////////////////REcord time //////////////////////////////////////
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float fTrT;
	cudaEventElapsedTime(&fTrT,start,stop);
	TrT 		= (double)fTrT;
	cudaEventDestroy(start);
	cudaEventDestroy(stop) ;


	/////////////////////train output////////////////////////////
	//malloc H again
	//size_mem = NoHidden* NoTrain* sizeof(double);

	//cudaMemGetInfo(&avail, &total);
	//printf("Avail mem before malloc H to compute T %ld\n",avail);
	//checkCudaErrors(cudaMalloc( (void**)&d_H, size_mem ));
	//checkCudaErrors( cudaMemcpy(d_H, H, size_mem, cudaMemcpyHostToDevice));

	//mult(d_T, d_H, d_OutputWeight, NoHidden, NoClass, NoTrain);
	CublasMult(MT, handle, MH, MOutputWeight, 0, 0);
	check_cuda_errors(__FILE__, __LINE__);
	//train_output_matrix = mult2(d_OutputWeight, H, NoClass, NoTrain ,NoHidden, 1, 0, 0, 1,0);
	//check output
	printf("Train Label Output:\n");
	DBGF(check_device_memory(d_T, NoClass, NoTrain,"ckc.TrainLabelcomputed"));
	cudaFree(d_H);

	//check accuray of training
	TrA = compute_accuracy_device(d_T, train_label, NoClass, NoTrain);
	printf("Accuracy training %lf\n ",TrA);

	//free train memory
	cudaFree(d_T);
	free(H);
	
}

///////////////////////////////////////////////////////////////////////////////////////////////

void CElm::run_test(){

	size_t avail, total, size_mem;

	//////////////////////////For Timing//////////////////////////////////
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);


	//malloc H for test
	cudaMemGetInfo(&avail, &total);
	printf("Avail mem at the begin of testing %ld\n",avail);
	size_mem = NoTest  * NoHidden *sizeof(double);
	checkCudaErrors(cudaMalloc( (void**)&d_H, size_mem ));
	compute_H(d_H, test, NoTest);;

//	checkCudaErrors(cudaMemcpy(H, d_H, size_mem, cudaMemcpyDeviceToHost ));
	printf("TempHtest activation\n");
	DBGF(check_device_memory(d_H, NoHidden, NoTest,"ckc.Htest"));


	//malloc for Test output label
	double *d_T;
	size_mem = NoTest * NoClass * sizeof(double);
	checkCudaErrors(cudaMalloc( (void**)&d_T, size_mem));

	//test_output_matrix = mult2(d_OutputWeight,H, NoClass, NoTest, NoHidden , 1, 0, 0,1,0);
	//mult(d_T, d_H, d_OutputWeight, NoHidden, NoClass, NoTest);
	//MatrixMultCublas(d_T, handle, d_H, d_OutputWeightT, NoHidden, NoTest, NoHidden, NoClass, NoClass, NoTest, 0, 1);
	MatrixMultCublasCol( d_T, handle, d_OutputWeight, d_H,
						 NoClass, NoHidden,
						 NoHidden, NoTest,
						 NoClass, NoTest,
						 0, 0);
	
	
	check_cuda_errors(__FILE__, __LINE__);

	//Check result
	printf("Test Label Output:\n");
	//DBGF(check_device_memory(d_T, NoTest, NoClass,"ckc.TestLabelcomputed"));
	//write result
	check_device_memory(d_T, NoClass, NoTest,"ckc.TestLabelcomputed");

	cudaFree(d_H);
	cudaFree(d_OutputWeight);
	//check accuray of training
	TeA = compute_accuracy_device(d_T, test_label, NoClass, NoTest);
	printf("Accurarcy testing %lf\n ",TeA);


	/////////////////////REcord time //////////////////////////////////////
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float fTeT;
	cudaEventElapsedTime(&fTeT,start,stop); 
	TeT 	= (double)fTeT;
	cudaEventDestroy(start);
	cudaEventDestroy(stop) ;
	//free label output
	cudaFree(d_T);


}

double CElm::compute_accuracy(const double *output_matrix, const int *label, int R, int C){
	//Input:
	//output_matrix: computed RAM outmatrix label
	//label: round truth
	//R, C: number/ of rows and columns
	//compute training accuracy
	int * label_computed;
	cudaMalloc((void**)&label_computed, C * sizeof(double));

	//malloc memory so that max is run on device
	double * d_output_matrix;
	cudaMalloc((void**)&d_output_matrix, R*C* sizeof(double));
	cudaMemcpy(d_output_matrix, output_matrix, R*C* sizeof(double), cudaMemcpyHostToDevice);

	//Init the kernel call
	int ThreadsperBlock = 256;
	int BlocksperGrid = (C + ThreadsperBlock -1)/ThreadsperBlock;
	max_matrix<<<BlocksperGrid, ThreadsperBlock>>>(d_output_matrix, label_computed, R, C);
	//checkmatrix
	check_device_memory(label_computed, C, 1);

	//compute the difference between real label and computed label
	int misclass = 0;
	int * h_label_computed;
	h_label_computed = (int*)malloc(C*sizeof(int));
	cudaMemcpy((void*)h_label_computed, label_computed, C*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i=0;i<C; i++)
		if (label[i]!=h_label_computed[i])
			misclass++;
	
	cudaFree(d_output_matrix);
	cudaFree(label_computed);
	free(h_label_computed);

	return 1-(double)misclass/C;
}

double CElm::compute_accuracy_device(const double *d_output_matrix, const double *label, int R, int C){
	//Input:
	//output_matrix: computed RAM outmatrix label
	//label: round truth
	//R, C: number/ of rows and columns
	//compute training accuracy
	if (R==1){ //Regression

		//copy from the device to host
		double * h_label_computed;
		h_label_computed = (double*)malloc(C*sizeof(double));
		checkCudaErrors(cudaMemcpy((void*)h_label_computed, d_output_matrix, C*sizeof(double),cudaMemcpyDeviceToHost)) ;

		//compute the mean square error
		double msr = 0;
		for (int i=0;i<C;i++)
			msr = msr + (h_label_computed[i] - label[i])*(h_label_computed[i] - label[i]);
		msr = sqrt(msr/C);

		//releae mem
		free(h_label_computed);

		return msr;
	}

	//else
	//Classification
	int * label_computed;
	checkCudaErrors( cudaMalloc((void**)&label_computed, C * sizeof(int)));


	//Init the kernel call
	int BlocksperGrid = (C + ThreadsperBlock -1)/ThreadsperBlock;
	max_matrix<<<BlocksperGrid, ThreadsperBlock>>>(d_output_matrix, label_computed, R, C);
	check_cuda_errors(__FILE__, __LINE__);
	//checkmatrix
	//check_device_memory(label_computed, C, 1);

	//compute the difference between real label and computed label
	int misclass = 0;
	int * h_label_computed;
	int right_label;
	h_label_computed = (int*)malloc(C*sizeof(int));
	checkCudaErrors(cudaMemcpy((void*)h_label_computed, label_computed, C*sizeof(int),cudaMemcpyDeviceToHost));
	for (int i=0;i<C; i++){
		right_label = (int)label[i] ;
		if (right_label!=h_label_computed[i])
			misclass++;
	}
	cudaFree(label_computed);
	free(h_label_computed);

	return 1-(double)misclass/C;
}

int CElm::CountFeature(char * line){
	char * pch;
	int No = 0;

	pch = strtok(line," ");

	//count the No of features
	while (pch!=NULL)
	{
		pch = strtok(NULL, "  ");
		No++;
	}
	
	printf("%d features\n",No);
	return No;
}

void CElm::neuralnet_init(int NHidden){

	//init dCOEFREG
	checkCudaErrors (cudaMemcpy( dCOEFREG, &COEFREG, sizeof( double), cudaMemcpyHostToDevice));

	NoHidden = NHidden;

	//the difficult in CUDA is the differentiate between host mem
	//and the device mem
	//free( InputWeight); //free first when repeat running
	InputWeight = (double*) malloc(NoHidden* (NoFeature+1) * sizeof(double)); //NoHIdden rows, NoFeature cols, NoFeature stride
	if (InputWeight ==NULL){
		perror ("Error in InputWeight :\n");
		exit(-1);
	}
	for (int i = 0; i < NoHidden*( NoFeature+1); i++)
    {
        InputWeight[i] = ((double)rand() / RAND_MAX)*2 - 1;
    }

}

void CElm::neuralnet_destroy(){
	
	//free everything related to Neuron and CUDA
	free(InputWeight);
	free(BiasHiddenNeurons);

	//writ result to file
	//Write file log
	fpointer = fopen("Result.log","a");
	fprintf(fpointer,"NoHidden :%d :\n",NoHidden);
	fprintf(fpointer,"Train Acc :%lf :\n",TrA);
	fprintf(fpointer,"Test Acc :%lf :\n",TeA);
	fprintf(fpointer,"Train Time :%lf :\n",TrT/1000.f);
	fprintf(fpointer,"Test Time :%lf :\n",TeT/1000.f);
	fprintf(fpointer,"\n");
	fclose(fpointer);
}

//----------------------------------------------------------
void CElm::compute_H(double* d_tempH, const double* Feature,int NoSample){
	//Input: 
	//Feature: NoSample vectors of length F
	//Output: Matrix H to compute the output

	//double * d_tempH;	//NoHidden x NoSample
	double * d_Feature;  //NoFeature x NoSample
	size_t size_mem;
	
	//d_Feature
	size_mem = (NoFeature+1) *NoSample *sizeof(double);
	checkCudaErrors (cudaMalloc( (void**)&d_Feature, size_mem ));
	//checkCudaErrors (cudaMalloc( (void**)&d_FeatureT, size_mem ));
	//cublasSetVector(NoFeature*NoSample,sizeof(double),Feature,1,d_Feature,1); //why 1 incre ->1 for col, leading dim for row?
	//DBGF(check_host_memory(Feature,NoSample, NoFeature+1, "ckc.Feature1"));
	checkCudaErrors (cudaMemcpy(d_Feature, Feature, size_mem, cudaMemcpyHostToDevice));

	//InputWeight
	size_mem = NoHidden *(NoFeature+1) *sizeof(double);
	checkCudaErrors (cudaMalloc((void **)&d_InputWeight,size_mem));	
	//cublasSetVector(NoHidden*NoFeature,sizeof(double),InputWeight,1,d_InputWeight,1);
	checkCudaErrors (cudaMemcpy(d_InputWeight, InputWeight, size_mem, cudaMemcpyHostToDevice));

	//check before mult
	printf("d_inputweigth \n");
	DBGF(check_device_memory(d_InputWeight,   NoHidden, NoFeature+1, "ckc.InputWeight")); //leading dimension is noColumns
	printf("d_Feature \n");
	//sept 4th: wrong order DBGF(check_device_memory(d_Feature, NoFeature+1 ,NoSample, "ckc.Feature"));

	//hiddenOutput( d_tempH, d_InputWeight, 
	//////////////activation type
	//char *strCmp;
	if ( strstr( nnType, "sig")!=NULL) //sums sig activation
		sumSig( d_tempH, d_Feature, d_InputWeight, NoSample);
		
	else if ( strstr( nnType, "rbf")!=NULL)// rbf kernel	
		rbfNN( d_tempH, d_Feature, d_InputWeight, NoSample);
		
	//release cuda memory immediately
	cudaFree(d_InputWeight);
	cudaFree(d_Feature);	
	
}

/*
void CElm::gaussianRbf(double * d_tempH, const double *d_Feature, const double* d_InputWeight, int NoSample)
{
int BlockY 	= (BLOCKSIZE- 1 + NoSample)/ BLOCKSIZE;	
int BlockX 	= (BLOCKSIZE- 1 + NoHidden)/ BLOCKSIZE;

dim3 grids( BlockX, BlockY);
dim3 blocks( BLOCKSIZE, BLOCKSIZE);

kernelGaussianRBF <<<grids, blocks>>> (d_tempH, d_Feature, d_InputWeight,
								NoFeature+ 1, NoSample, NoHidden); //NoFeature+1: include vector b in INputWeight
check_cuda_errors( __FILE__, __LINE__);
printf("H = exp( d_Feature - d_InputWeight) \n");

}
*/

void CElm::rbfNN(double * d_tempH, const double *d_Feature, const double* d_InputWeight, int NoSample)
{

int BlockY 	= (BLOCKSIZE- 1 + NoSample)/ BLOCKSIZE;	
int BlockX 	= (BLOCKSIZE- 1 + NoHidden)/ BLOCKSIZE;

dim3 grids( BlockX, BlockY);
dim3 blocks( BLOCKSIZE, BLOCKSIZE);

printf("H = d_Feature - d_InputWeight \n");
if (strcmp( nnType, "multiquadricrbf")==0)
	{
	kernelQuadricRBF <<<grids, blocks>>> (d_tempH, d_Feature, d_InputWeight,
								NoFeature+ 1, NoSample, NoHidden); //NoFeature+1: include vector b in INputWeight
	check_cuda_errors( __FILE__, __LINE__);	
	return;						
	}
	
if (strcmp(nnType, "inversemultiquadricrbf")==0)
	{
	kernelInverseQuadricRBF <<<grids, blocks>>> (d_tempH, d_Feature, d_InputWeight,
								NoFeature+ 1, NoSample, NoHidden); //NoFeature+1: include vector b in INputWeight
	check_cuda_errors( __FILE__, __LINE__);	
	return;						
	}							
if (strcmp(nnType, "gaussianrbf")==0)
	{
	kernelGaussianRBF <<<grids, blocks>>> (d_tempH, d_Feature, d_InputWeight,
								NoFeature+ 1, NoSample, NoHidden); //NoFeature+1: include vector b in INputWeight
	check_cuda_errors( __FILE__, __LINE__);		
	return;																					
	}							

//wrong rbf kernel
printf(" << Wrong RBF kernel \n");
exit(-1);
}

void CElm::sumSig( double* d_tempH, double* d_Feature,  double* d_InputWeight,   int NoSample)
{

	MatrixMultCublasCol( d_tempH, handle, d_InputWeight, d_Feature, 
						NoHidden, NoFeature +1,
						NoFeature +1, NoSample,
						NoHidden, NoSample,
						0, 0);
	DBGF(check_device_memory(d_tempH,  NoHidden, NoSample, "ckc.tempHnew"));
	check_cuda_errors(__FILE__, __LINE__);

	//check after mult
	//printf("tempH = d_inputweight* d_Feature \n");
	printf("tempH = d_Feature* d_InputWeight \n");

	//NoHidden*NoSample is too large to fit into the kernel
	//must be divide into smaller part

	//int BlocksperGrid = (NoHidden*NoSample + ThreadsperBlock -1)/ThreadsperBlock;
	int W = NoHidden;
	int H = NoSample;
	dim3 Block(BLOCKSIZE,BLOCKSIZE);
	//dim3 Block(1, ThreadsperBlock); //to maximize the number of block
	dim3 Grid( (BLOCKSIZE-1+ W)/Block.x, (BLOCKSIZE-1+ H)/Block.y);

	
	//ACTIVATION
	activate_fun_matrix<<<Grid,Block>>>("sig", d_tempH, W, H);
	check_cuda_errors(__FILE__, __LINE__);

	//DBGF(check_device_memory(d_tempH, NoSample, NoHidden,"ckc.H"));
	////Have to check every command because this is new
	////compute H = 1/(1+exp(-tempH) by kernel function	
}

double* CElm::compute_weight(double* dpinvH, double* T){
	//dpinv: NoTrain x NoHidden
	//cublasHandle_t handle;
	double * d_OutputWeight; //NoHidden x NoClass
	double * d_T;//NoClass x NoTrain
	//double alpha = 1.0f; 
	//double beta = 0.0f;


	int SizeinBytesOut = NoHidden * NoClass *sizeof(double);
	int SizeinBytesLabel = NoTrain * NoClass *sizeof(double);
	

	//cublasCreate(&handle);

	size_t avai, total;
	cudaMemGetInfo(&avai, &total);
	printf("Avail me: %ld, total mem: %ld \n", avai,total);

	cudaMalloc((void**)&d_OutputWeight, SizeinBytesOut);
	cudaMalloc((void**)&d_T,SizeinBytesLabel);
	cublasSetVector(NoTrain * NoClass, sizeof(double), T, 1, d_T, 1);

	//check input
	printf("pinvH \n");
	check_device_memory(dpinvH, NoTrain, NoHidden);
	printf("T label \n");
	check_device_memory(d_T, NoClass, NoTrain);
	
	//mult: 
	//exit
	mult(d_OutputWeight, d_T, dpinvH, NoTrain, NoHidden, NoClass);

	/*cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
		NoHidden, NoClass, NoTrain, &alpha,
				dpinvH, NoTrain,
				d_T, NoClass, &beta,
				d_OutputWeight, NoHidden);*/

	cudaDeviceSynchronize();	
//free
	cudaFree(d_T);
	//cublasDestroy(handle);

	return d_OutputWeight;


}

//---------------------------------------------------------------------------------------------------
