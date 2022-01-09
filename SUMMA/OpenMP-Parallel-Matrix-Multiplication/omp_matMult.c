/******
Script for Matrix Multiplication with Sequential ijk Algorithm vs.
Parallel Algorithm with (standard) OpenMP and an optimized OpenMP version
******/
// Compilation MacBook: gcc -openmp -g -Wall -std=c99 omp_matMult.c -o omp_matMult -lm
// Compilation Cluster: gcc -fopenmp -g -Wall -std=c99 omp_matMult.c -o omp_matMult -lm
// Run: ./omp_matMult <No. of Iterations> ----> Number of Iterations i.e. 10, 20, 100...
/*** Standard Parameter:
	- MAX_DIM = 2000*2000
	- MAX_VAL = 10
	- MIN_VAL = 1
	- THRESHOLD = 0.001
	- iteration = 10
	- dimension = 16 to 2048 (iterated in for()-loop !!)
	- numThreads = 16 to 1024 (iterated in for()-loop !!)
***/

// Headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <unistd.h> // Fix implicit declaration of function 'getpid' is invalid in C99

// Parameters
#define MIN_VAL 1
#define MAX_VAL 10

// Constants
const double THRESHOLD = 0.001; // Threshold to check for accuracy of different calculations
const int MIN_DIM = 16;
const int MAX_DIM = 16384;
const int MIN_THREAD = 2;
const int MAX_THREAD = 128;

// matrix creation method
unsigned long mix(unsigned long a, unsigned long b, unsigned long c);
double* randomMatrix(int dimension);
double* zeroMatrix(int dimension);
//void displayMatrix(double* matrix, int dimension);

// matrix multiplication methods
double sequentialMultiply(double* matrixA, double* matrixB, double* matrixC, int dimension);
double parallelMultiply_omp_static(double* matrixA, double* matrixB, double* matrixC, int dimension, int numThreads);
double parallelMultiply_omp_dynamic(double* matrixA, double* matrixB, double* matrixC, int dimension, int numThreads);

// test cases
void sequentialMultiplyTest(int dimension, int iterations);
void parallelMultiplyTest_omp_static(int dimension, int iterations, int numThreads);
void parallelMultiplyTest_omp_dynamic(int dimension, int iterations, int numThreads);

// verify multiplication
int verifyMultiplication(double* matrixA, double* matrixB, double* result, int dimension);


/********** Main Script *************/
int main(int argc, char* argv[]) {

	int numThreads, iterations, dimension;

	if(argc != 2)
	{
		printf("Usage: %s <iterations>\n", argv[0]);
		return -1;
	}
	iterations = strtol(argv[1], NULL, 10);

	// Generate Necessary files
	// Create Sequential Multiply test log
	FILE* fp;
	fp = fopen("sequentialMultiplyTest.txt", "w+");
	fclose(fp);

	// Create Parallel Multiply OMP static test log
	fp = fopen("parallelMultiplyTest_omp_static.txt", "w+");
	fclose(fp);

	// Create Parallel Multiply OMP guided test log
	fp = fopen("parallelMultiplyTest_omp_dynamic.txt", "w+");
	fclose(fp);

	for(dimension=MIN_DIM; dimension<=MAX_DIM; dimension=dimension*2) {
		for(numThreads=MIN_THREAD; numThreads<=MAX_THREAD; numThreads=numThreads*2) {
			parallelMultiplyTest_omp_static(dimension, iterations, numThreads);
		}
	}

	for(dimension=MIN_DIM; dimension<=MAX_DIM; dimension=dimension*2) {
		for(numThreads=MIN_THREAD; numThreads<=MAX_THREAD; numThreads=numThreads*2) {
			parallelMultiplyTest_omp_dynamic(dimension, iterations, numThreads);
		}
	}

	for(dimension=MIN_DIM; dimension<=MAX_DIM; dimension=dimension*2){
		sequentialMultiplyTest(dimension, iterations);
	}
	// return successfully
	return 0;
}


/**************** Functions to call in Main-script or Test-scripts *************/
// set seet for random numbers generator
unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
	a=a-b;  a=a-c;  a=a^(c >> 13);
	b=b-c;  b=b-a;  b=b^(a << 8);
	c=c-a;  c=c-b;  c=c^(b >> 13);
	a=a-b;  a=a-c;  a=a^(c >> 12);
	b=b-c;  b=b-a;  b=b^(a << 16);
	c=c-a;  c=c-b;  c=c^(b >> 5);
	a=a-b;  a=a-c;  a=a^(c >> 3);
	b=b-c;  b=b-a;  b=b^(a << 10);
	c=c-a;  c=c-b;  c=c^(b >> 15);
	return c;
 }

 /************ create square matrizes A & B with random values ************/
 double* randomMatrix(const int dimension) {
	 	// generate 2-dimensional random matrix with doubles
		double* matrix = (double *) calloc(dimension*dimension, sizeof(double));
 		// set seed with mix()-function
    unsigned long seed = mix(clock(), time(NULL), getpid());
    srand(seed);
 		// initialize the matrix
    int j, i;
 		double rnd = 0.0;
		#pragma omp parallel for
    for (i=0; i<dimension; i++) {
        for (j=0; j<dimension; j++) {
            rnd = rand() % MAX_VAL + MIN_VAL;
            matrix[i*dimension + j] = rnd;
         }
     }
		 return matrix;
 }

 /************ create square matrizes C with zeros ************/
 double* zeroMatrix(const int dimension) {
	 	// generate 2-dimensional random matrix with doubles
		double* matrix = (double *) calloc(dimension*dimension, sizeof(double));
 		// initialize the matrix
    int j, i;
		#pragma omp parallel for
    for (i=0; i<dimension; i++) {
        for (j=0; j<dimension; j++) {
            matrix[i*dimension + j] = 0.0;
         }
     }
		 return matrix;
 }

/************* Display the squared matrix **************/
/*void displayMatrix(double* matrix, const int dimension) {
	int i, j; // Iterators
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			printf("%f\t", matrix[i][j]);
		}
		printf("\n");
	}
}*/

/*********** Sequential Matrix Multiplication ***********/
double sequentialMultiply(double* matrixA, double* matrixB, double* matrixC, const int dimension) {
// Sequentiall multiply given input matrices and return resultant matrix
	int i, j, k; // Iterators
	double elapsed; // time difference

	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	/* Head */
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			matrixC[i*dimension + j] = 0.0;
			for(k=0; k<dimension; k++) {
				matrixC[i*dimension + j] += matrixA[i*dimension + k] * matrixB[k*dimension + j];
			}
		}
	}
	/* Tail */
	gettimeofday(&t1, 0);
	elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

/*********** parallel Matrix Multiplication with OpenMP and static schedule **********/
double parallelMultiply_omp_static(double* matrixA, double* matrixB, double* matrixC, int dimension, int numThreads) {
// Parallel multiply given input matrices using optimal methods and return resultant matrix
	int i, j, k; // Iterators
	double elapsed; // time difference
	// take the time
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	/* Head */
	#pragma omp parallel private(i, j, k) num_threads(numThreads)
	{
		#pragma omp for schedule(static)
		for(i=0; i<dimension; i++) {
			for(j=0; j<dimension; j++) {
				matrixC[i*dimension + j] = 0.0;
				for(k=0; k<dimension; k++) {
					matrixC[i*dimension + j] += matrixA[i*dimension + k] * matrixB[k*dimension + j];
				}
			}
		}
	}
	/* Tail */
	gettimeofday(&t1, 0);
	elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

/*********** parallel Matrix Multiplication with OpenMP and static schedule **********/
double parallelMultiply_omp_dynamic(double* matrixA, double* matrixB, double* matrixC, int dimension, int numThreads) {
// Parallel multiply given input matrices using optimal methods and return resultant matrix
	int i, j, k;
	double elapsed; // time difference
	// take the time
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	/* Head */
	#pragma omp parallel private(i, j, k) num_threads(numThreads)
	{
		#pragma omp for schedule(dynamic)
		for(i=0; i<dimension; i++) {
			for(j=0; j<dimension; j++) {
				matrixC[i*dimension + j] = 0.0;
				for(k=0; k<dimension; k++) {
					matrixC[i*dimension + j] += matrixA[i*dimension + k] * matrixB[k*dimension + j];
				}
			}
		}
	}
	/* Tail */
	gettimeofday(&t1, 0);
	elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

/********** Check the Results of the different Multiplications against Sequantial Calculation ***********/
int verifyMultiplication(double* matrixA, double* matrixB, double* result, int dimension) {

	int i, j, k;
	double tot, sumErrors = 0.0;

	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			tot = 0.0;
			for(k=0; k<dimension; k++) {
				tot += matrixA[i*dimension + k] * matrixB[k*dimension + j];
			}
			sumErrors += fabs( tot - result[i*dimension] + j);
		}
	}
	sumErrors = fabs( (sumErrors/(dimension*dimension)) - THRESHOLD );
	return sumErrors;
}

/************* Test Scripts for Multiplications ***************/
// Sequential Test Script
void sequentialMultiplyTest(int dimension, int iterations) {
	// prepare file to write
	FILE* fp;
	fp = fopen("sequentialMultiplyTest.txt", "a+");
	// Console write
	printf("----------------------------------\n");
	printf("Test : Sequential Multiply        \n");
	printf("----------------------------------\n");
	printf("Dimension: %d\n", dimension);
	printf("..................................\n");
	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : Sequential Multiply        \n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension : %d\n", dimension);
	fprintf(fp, "..................................\n");

	// declare matrices and variables
	double* matrixA = NULL;
	double* matrixB = NULL;
	double* matrixResult = NULL;
	double* opmLatency = NULL;
	// allocate space for variables
	matrixA = (double *) calloc(dimension*dimension, sizeof(double));
	matrixB = (double *) calloc(dimension*dimension, sizeof(double));
	matrixResult = (double *) calloc(dimension*dimension, sizeof(double));
	opmLatency = malloc(iterations * sizeof(double));
	// create random matrices A and B
	matrixA = randomMatrix(dimension);
	matrixB = randomMatrix(dimension);

	// Iterate and measure performance
	int i;
	for(i=0; i<iterations; i++) {
		matrixResult = zeroMatrix(dimension);
		opmLatency[i] = sequentialMultiply(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		// Console write
		printf("%d.\t%f\n", i+1, opmLatency[i]);
		// File write
		fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
	}

	// Console write
	printf("\n");
	printf("----------------------------------\n");
	printf("Analyze Measurements              \n");
	printf("----------------------------------\n");
	// File write
	fprintf(fp, "\n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Analyze Measurements              \n");
	fprintf(fp, "----------------------------------\n");

	// Start statistical analysis
	double sum = 0.0;
	double sumSquared = 0.0;

	for(i=0; i<iterations; i++) {
		sum += opmLatency[i];
		sumSquared += pow(opmLatency[i], 2.0);
	}

	double mean = sum / iterations;
	double squareMean = sumSquared / iterations;
	double standardDeviation = sqrt(squareMean - pow(mean, 2.0));

	// Console write
	printf("Mean               : %f\n", mean);
	printf("Standard Deviation : %f\n", standardDeviation);
	printf("----------------------------------\n");
	//File write
	fprintf(fp, "Mean               : %f\n", mean);
	fprintf(fp, "Standard Deviation : %f\n", standardDeviation);
	fprintf(fp, "----------------------------------\n");

	// Releasing memory
	fclose(fp);
	free(opmLatency);
	free(matrixA);
	free(matrixB);
}

// Parallel OpenMP Test Script
void parallelMultiplyTest_omp_static(int dimension, int iterations, int numThreads) {
	FILE* fp;
	fp = fopen("parallelMultiplyTest_omp_static.txt", "a+");

	// Console write
	printf("----------------------------------\n");
	printf("Test : Parallel Multiply OMP static        \n");
	printf("----------------------------------\n");
	printf("Dimension: %d\tNo. of Threads: %d\n", dimension, numThreads);
	printf("..................................\n");

	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : Parallel Multiply OMP static         \n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension: %d\tNo. of Threads: %d\n", dimension, numThreads);
	fprintf(fp, "..................................\n");

	// declare matrices and variables
	double* matrixA = NULL;
	double* matrixB = NULL;
	double* matrixResult = NULL;
	double* opmLatency = NULL;
	// allocate space for variables
	matrixA = (double *) calloc(dimension*dimension, sizeof(double));
	matrixB = (double *) calloc(dimension*dimension, sizeof(double));
	matrixResult = (double *) calloc(dimension*dimension, sizeof(double));
	opmLatency = malloc(iterations * sizeof(double));
	// create random matrices A and B
	matrixA = randomMatrix(dimension);
	matrixB = randomMatrix(dimension);

	// Iterate and measure performance
	int i;
	for(i=0; i<iterations; i++) {
		matrixResult = zeroMatrix(dimension);
		opmLatency[i] = parallelMultiply_omp_static(matrixA, matrixB, matrixResult, dimension, numThreads);
		// check the result of the simple OpenMP parallel Multiplication
		//matrixCheck[i] = verifyMultiplication_omp(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		// Console write
		printf("%d.\t%f\n", i+1, opmLatency[i]);
		// File write
		fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
	}

	// Console write
	printf("\n");
	printf("----------------------------------\n");
	printf("Analyze Measurements              \n");
	printf("----------------------------------\n");

	// File write
	fprintf(fp, "\n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Analyze Measurements              \n");
	fprintf(fp, "----------------------------------\n");

	double sum = 0.0;
	double sumSquared = 0.0;

	// Statistical analyze
	for(i=0; i<iterations; i++) {
		sum += opmLatency[i];
		sumSquared += pow(opmLatency[i], 2.0);
	}

	double mean = sum / iterations;
	double squareMean = sumSquared / iterations;
	double standardDeviation = sqrt(squareMean - pow(mean, 2.0));

	// Console write
	printf("Mean               : %f\n", mean);
	printf("Standard Deviation : %f\n", standardDeviation);
	printf("----------------------------------\n");

	//File write
	fprintf(fp, "Mean               : %f\n", mean);
	fprintf(fp, "Standard Deviation : %f\n", standardDeviation);
	fprintf(fp, "----------------------------------\n");

	// Releasing memory
	fclose(fp);
	free(opmLatency);
	//free(matrixCheck);
	free(matrixA);
	free(matrixB);
}

// optimized parallel OpenMP Test Script
void parallelMultiplyTest_omp_dynamic(int dimension, int iterations, int numThreads) {
	FILE* fp;
	fp = fopen("parallelMultiplyTest_omp_dynamic.txt", "a+");

	// Console write
	printf("----------------------------------\n");
	printf("Test : Parallel Multiply OMP dynamic\n");
	printf("----------------------------------\n");
	printf("Dimension: %d\tNo. of Threads: %d\n", dimension, numThreads);
	printf("..................................\n");

	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : Parallel Multiply OMP dynamicn");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension: %d\tNo. of Threads: %d\n", dimension, numThreads);
	fprintf(fp, "..................................\n");

	// declare matrices and variables
	double* matrixA = NULL;
	double* matrixB = NULL;
	double* matrixResult = NULL;
	double* opmLatency = NULL;
	// allocate space for variables
	matrixA = (double *) calloc(dimension*dimension, sizeof(double));
	matrixB = (double *) calloc(dimension*dimension, sizeof(double));
	matrixResult = (double *) calloc(dimension*dimension, sizeof(double));
	opmLatency = malloc(iterations * sizeof(double));
	// create random matrices A and B
	matrixA = randomMatrix(dimension);
	matrixB = randomMatrix(dimension);

	// Iterate and measure performance
	int i;
	for(i=0; i<iterations; i++) {
		matrixResult = zeroMatrix(dimension);
		opmLatency[i] = parallelMultiply_omp_dynamic(matrixA, matrixB, matrixResult, dimension, numThreads);
		// check the result of the simple OpenMP parallel Multiplication
		//matrixCheck[i] = verifyMultiplication_omp_opti(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		// Console write
		printf("%d.\t%f\n", i+1, opmLatency[i]);
		// File write
		fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
	}

	// Console write
	printf("\n");
	printf("----------------------------------\n");
	printf("Analyze Measurements              \n");
	printf("----------------------------------\n");

	// File write
	fprintf(fp, "\n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Analyze Measurements              \n");
	fprintf(fp, "----------------------------------\n");

	double sum = 0.0;
	double sumSquared = 0.0;

	// Statistical analyze
	for(i=0; i<iterations; i++) {
		sum += opmLatency[i];
		sumSquared += pow(opmLatency[i], 2.0);
	}

	double mean = sum / iterations;
	double squareMean = sumSquared / iterations;
	double standardDeviation = sqrt(squareMean - pow(mean, 2.0));

	// Console write
	printf("Mean               : %f\n", mean);
	printf("Standard Deviation : %f\n", standardDeviation);
	printf("----------------------------------\n");

	//File write
	fprintf(fp, "Mean               : %f\n", mean);
	fprintf(fp, "Standard Deviation : %f\n", standardDeviation);
	fprintf(fp, "----------------------------------\n");

	// Releasing memory
	fclose(fp);
	free(opmLatency);
	//free(matrixCheck);
	free(matrixA);
	free(matrixB);
}
