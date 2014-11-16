#ifndef ELM_H
#define ELM_H

#include "cuda.h"
#include <stdio.h>
#include <cublas_v2.h>


//Revision Feb. 15 13
//Change the label to double so this ELM can work with regression
//
//

class CElm {
public:
	CElm( const char*, const char*, int C, int F, int Tr, int Te, int N, const char*);
	//CElm(train file, test file, NoClass, NoFeature, NoTrain, NoTest, NoHidden, Type)
	~CElm();
	////void func();

	//for multiGPU
	int 			gpuid[100]; //100 GPUs max

	int 			ThreadsperBlock;
	int 			ThreadsperSM;
	int 			BlocksperGrid;
	
	//for scale in case of single precision
	bool 			bRegularize; //Regularize or Not
	double 			COEFREG; //regularize coeff
	double 			SCALE;
	double 			*dSCALE;
	double 			*dCOEFREG;
	
	FILE* 			fpointer;

	double 			TeA,TrA; //test acc and train acc
	double 			TeT, TrT; //test time and train time
	double 			*train;   //train data 2D matrix
					//row is sample
					//column is feature
					//NoSample x NoFeature: it is this guy that make my thought crooked
	//change
	//int *train_label;	//first 1D vector
	double 			*train_label;

	double 			*train_label_matrix;//later, 2D matrix
						//row is number of Class
						//col is number of NoTrain	
	double 			*train_output_matrix;

	int 			NoTrain;
	int 			NoTest;
	int 			NoFeature;

	double 			*test;

	char 			nnType[32]; //neuron network type	
		
	//change
	//int *test_label;
	double 			*test_label;
	//double *test_label_matrix;	//Have to specify the dimension of the matrix
	double 			*test_output_matrix;

	//for classification
	//int 			type;//activation func, different from nnType
	//Number of hidden Neurons
	int 			NoHidden;
	//number of class = No of Output Neurons
	int 			NoClass; //1 mean regression

	//the matrix to compute classification
	double 			*H; //NoHidden x NoSample
	double 			*d_H; //for cuda mem
	//NoNeed to store in elm because H is different from train and test

	//never can store device memory because they are expensive
	double  			*d_InputWeight, *InputWeight; //NoHidden x NoFeature
	double  			*BiasHiddenNeurons; //NoHidden x 1
	//after I add 1 to each feature, no need BiasHidden add but I need chang
	//2 things: read feature file and add one more dim in W matrix
	double 	 		*d_OutputWeight,  *d_OutputWeightT;//NoHIdden x NoClass

	//Solv min by GaussSeidel
	void 		compute_Wo( double *W,  double *H,  double *T); //WH = T
	void 		solveGaussSeidel(double *x, double *d_L, double *d_U, double *b, const int N);

	//Method
	void 		run_train(); //run the classificatyion
	void 		run_train_GaussSeidel();
	void 		run_test(); //test
	//load the data into memory
	void 		load(const char * file, double * label, double * feature,const int NoFeature, const int NoSample);

	//Initialize neuron net
	void 		neuralnet_init(int NoHidden);
	void 		neuralnet_destroy();

	//compute the H matrix for using pinvers in training
	void 		compute_H(double*, const double * Feature, int NoSample); //used in testing
	void 		sumSig( double* ,  double* Feature,  double* Input, int );
	void 		rbfNN( double*, const double* Feature, const double* Input, int);
	//void 		gaussianRbf( double*, const double* Feature, const double* Input, int);
	
	//this double * is a device memory 

	double* 		compute_weight(double* dpinvH, double* T);

	//classify the test data
	void 		classify();
	void 		func(int *p);
	//count the number of feature
	int 		CountFeature(char * line);//from a line count features

	//convert from class to binary class -1 1
	void 		convert(double*, const double * label, const int No, const int NoClass);

	void 		print_matrix(double *A, int r, int c);

	//cuda relate function
	//__device__ void activate_fun(const char* fun, double * H);
	double 		compute_accuracy(const double *output_matrix, const int *label, int R, int C);

	double 		compute_accuracy_device(const double *output_matrix, const double *label, int R, int C);
};



#endif
