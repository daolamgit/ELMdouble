#include	"cuda.h"
#include 	"cuda_runtime.h"
#include 	<cublas_v2.h>
#include 	"helper_cuda.h"
#include 	"device_launch_parameters.h"

#include 	"elm.h"
#include 	<stdio.h>
#include 	<stdlib.h>
#include 	<time.h>
#include 	<iostream>
//#include "common/book.h"

using namespace std;

//__global__ void add(){
//
//}

int 	main(int argc, char ** argv){

	
	if (argc < 9){
		printf("Syntax: train test NoClass NoFeatures NoTrain NoTest NoHidden(must be BLOCKSIZE) Activat \n");
		exit(-1);
	}
	
	CElm 	*elm = new CElm(argv[1], argv[2],
						atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), 
						argv[8]);

//test, train, NoClass, NoFeature, NoTest, NoTrain, NoHidden, sig						
//CElm *elm = new CElm("C:\\temp\\train.txt","C:\\temp\\test.txt",10, 3200, 6000, 1,3200,"sig") ;   
	//2 8 576 192 16 sig for diabetes
//elm->load("train.txt","test.txt");

//CElm *elm = new CElm("diabetes_train","diabetes_test",2, 8, 10, 6,8,"sig") ;

//find the cuda device
int 		dev = findCudaDevice(argc, (const char**) argv);
if (dev==-1) { 
	printf("No gpu \n");
	return 0;
}

int 		Iter = 1;
FILE* 		fid;

int NoHidden = 1024;
do
	{
	for (int i=1;i<=1;i++)
		{
		//int NoHidden = 4096*i;
		printf("---------------------NoHidden = %d------------------\n",NoHidden);

		fid = fopen("TestRun","a");
		if (fid==NULL)
			{
			perror("No file to write \n");
			return 0;
			}

		//use to store the iteration result
		double 		T_A 		= 0;
		double 		TY_A		= 0;
		double 		T_time	= 0;
		double 		TY_time	= 0;

		////////////////////////////////////////////
		//generate different random weight
		//srand( time(NULL));
		////////////////////////////////////////////
		
		

		for (int iter=0; iter<Iter; iter++)
			{
			srand(1); //same random every run to tune up the param	
			elm->neuralnet_init(NoHidden);

			//elm->run_train_GaussSeidel(); %an attemp for GaussSeidel not work well

			elm->run_train();
			elm->run_test();

			//get the result
			T_A 		+= elm->TrA;
			TY_A		+= elm->TeA;
			T_time		+= elm->TrT;
			TY_time 	+= elm->TeT;

			elm->neuralnet_destroy();
			}

		fprintf(fid,"%4.2f \t %4.2f \t %4.2f \t %4.2f \n", T_A/Iter, TY_A/Iter, T_time/Iter/1000, TY_time/Iter/1000);
		fclose( fid);

		//return, for nvpp check
		//return 1;
		}
		
	printf( "No Train :\n");
	int noTrain;
	scanf( "%d", &noTrain);
	elm->NoTrain = noTrain;
	
	printf( "No Test :\n");
	int noTest;
	scanf( "%d", &noTest);
	elm->NoTest 	= noTest;
	
	printf( "No Hidden :\n");
	//int noTrain;
	scanf( "%d", &NoHidden);
	
	printf( "Coef Reg: \n");
	double coefReg;
	scanf( "%lf", &coefReg);
	elm->COEFREG 		= coefReg;
	
	printf( "act type:\n");
	char actType[32];
	scanf( "%s", actType);
	strcpy( elm->nnType,  actType);
	}
while (1); //until key break	


/*
int NoTrain = elm->NoTrain;
int NoTest = elm->NoTest;
int NoClass = elm->NoClass;
printf("\n");
for (int i=0;i<NoClass;i++){
	for (int j=0;j<NoTrain;j++)
		printf("%d ",elm->train_label[IDX2C(i,j,NoTrain)]);
		printf("\n");
	}

printf("\n");
for (int i=0;i<NoClass;i++){
	for (int j=0;j<NoTest;j++)
		printf("%d ",elm->test_label[IDX2C(i,j,NoTest)]);
		printf("\n");
	}
*/



return 0;
}
