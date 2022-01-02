/******
Script for Matrix Multiplication with Sequential ijk Algorithm vs.
Parallel Algorithm with (standard) OpenMP and an optimized OpenMP version
******/
// Compilation: gcc-11 -openmp matMult_omp.c -o matMult_omp
// Run: ./matMult_omp <No. of Iterations> ----> Number of Iterations i.e. 10, 20, 100...
/*** Standard Parameter:
	- MAX_DIM = 2000*2000
	- MAX_VAL = 10
	- MIN_VAL = 1
	- THRESHOLD = 0.001
	- iteration = 10
	- dimension = 200 to 2000 (iterated in for()-loop !!)
	- numThreads = 16 to 1024 (iterated in for()-loop !!)
***/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>

#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

const double THRESHOLD = 0.001; // Threshold to check for accuracy of different calculations

// Method signatures
double** randomSquareMatrix(int dimension);
double** zeroSquareMatrix(int dimension);
void displaySquareMatrix(double** matrix, int dimension);
void convert(double** matrixA, double** matrixB, int dimension);

// Matrix multiplication methods
double sequentialMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension);
double parallelMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension);
double optimizedParallelMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension);

// Test cases
void sequentialMultiplyTest(int dimension, int iterations, int numThreads);
void parallelMultiplyTest(int dimension, int iterations, int numThreads);
void optimizedParallelMultiplyTest(int dimension, int iterations, int numThreads);

// 1 Dimensional matrix on stack
TYPE flatA[MAX_DIM];
TYPE flatB[MAX_DIM];

// Verify multiplication
void verifyMultiplication(double** matrixA, double** matrixB, double** result, int dimension);

/********** Main Script *************/
int main(int argc, char* argv[]) {

	int numThreads;
	int iterations, dimension;

	if(argc != 2)
	{
		printf("Usage: %s <iterations>\n", argv[0]);
		return -1;
	}
	iterations = strtol(argv[1], NULL, 10);

	// Generate Necessary files
	// Create Sequential Multiply test log
	FILE* fp;
	fp = fopen("SequentialMultiplyTest.txt", "w+");
	fclose(fp);

	// Create Parallel Multiply test log
	fp = fopen("ParallelMultiplyTest.txt", "w+");
	fclose(fp);

	// Create Optimized Parallel Multiply test log
	fp = fopen("OptimizedParallelMultiplyTest.txt", "w+");
	fclose(fp);

	for(dimension=200; dimension<=2000; dimension+=200) {
		for(numThreads=16; numThreads<=1024; numThreads=numThreads*2) {
			optimizedParallelMultiplyTest(dimension, iterations, numThreads);
		}
	}

	for(dimension=200; dimension<=2000; dimension+=200) {
		for(numThreads=16; numThreads<=1024; numThreads=numThreads*2) {
			parallelMultiplyTest(dimension, iterations, numThreads);
		}
	}

	for(dimension=200; dimension<=2000; dimension+=200){
		sequentialMultiplyTest(dimension, iterations);
	}
	return 0;
}


/**************** Functions to call in Main-script or Test-scrpts *************/
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

/******** create a random square matrix ************/
double** randomSquareMatrix(int dimension) {
// Generate 2 dimensional random TYPE matrix
	double** matrix = malloc(dimension * sizeof(double*));
	int i,j; // Iterators
	for(i=0; i<dimension; i++) {
		matrix[i] = malloc(dimension * sizeof(double));
	}
// set Random seed
 	unsigned long seed = mix(clock(), time(NULL), getpid()); // call mix-function
 	srand(seed);

	#pragma omp parallel for
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
		}
	}
	return matrix;
}

/************ create a square matrix from zeros ************/
double** zeroSquareMatrix(int dimension) {
// Generate 2 dimensional zero double matrix
	double** matrix = malloc(dimension * sizeof(double*));
	int i, j; // Iterators
	for(i=0; i<dimension; i++) {
		matrix[i] = malloc(dimension * sizeof(double));
	}
// set Random seed
 	unsigned long seed = mix(clock(), time(NULL), getpid()); // call mix-function
 	srand(seed);

	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			matrix[i][j] = 0;
		}
	}
	return matrix;
}

/************* Display the squared matrix **************/
void displaySquareMatrix(double** matrix, int dimension) {
	int i, j; // Iterators
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			printf("%f\t", matrix[i][j]);
		}
		printf("\n");
	}
}

/*********** Sequential Matrix Multiplication ***********/
double sequentialMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension) {
// Sequentiall multiply given input matrices and return resultant matrix
	int i,j,k; // Iterators

	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	/* Head */
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			for(k=0; k<dimension; k++) {
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
	/* Tail */
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

/*********** parallel Matrix Multiplication with OpenMP **********/
double parallelMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension, int numThreads) {
// Parallel multiply given input matrices and return resultant matrix
	int i,j,k;

	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	/* Head */
	#pragma omp parallel num_threads(numThreads)
	{
		#pragma omp for
		for(i=0; i<dimension; i++) {
			for(j=0; j<dimension; j++) {
				for(k=0; k<dimension; k++) {
					matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
				}
			}
		}
	}
	/* Tail */
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

double optimizedParallelMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension, int numThreads) {
// Parallel multiply given input matrices using optimal methods and return resultant matrix

	int i, j, k, iOff, jOff;
	double tot;

	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	/* Head */
	convert(matrixA, matrixB, dimension);
	#pragma omp parallel shared(matrixC) private(i, j, k, iOff, jOff, tot) num_threads(numThreads)
	{
		#pragma omp for schedule(static)
		for(i=0; i<dimension; i++) {
			iOff = i * dimension;
			for(j=0; j<dimension; j++) {
				jOff = j * dimension;
				tot = 0;
				for(k=0; k<dimension; k++) {
					tot += flatA[iOff + k] * flatB[jOff + k];
				}
				matrixC[i][j] = tot;
			}
		}
	}
	/* Tail */
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}
// convert-function for optimized matrix multiplication with OpenMP
void convert(double** matrixA, double** matrixB, int dimension) {
	int i,j;
	#pragma omp parallel for
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			flatA[i * dimension + j] = matrixA[i][j];
			flatB[j * dimension + i] = matrixB[i][j];
		}
	}
}

/********** Check the Results of the different Multiplications against Sequantial Calculation ***********/
void verifyMultiplication(double** matrixA, double** matrixB, double** result, int dimension) {

	int i,j,k;
	double tot, sumErrors;
	char verification;
	for(i=0; i<dimension; i++) {
		for(j=0; j<dimension; j++) {
			tot = 0.0;
			for(k=0; k<dimension; k++) {
				tot += matrixA[i][k] * matrixB[k][j];
			}
			sumErrors += fabs(tot - result[i][j]);
			/*if(fabs(tot-result[i][j]) > THRESHOLD) {
				printf("Result is incorrect!\n");
				return; */
			}
		}
	}
	sumErrors = sumErrors / (dimension*dimension);
	if(sumErrors > THRESHOLD) {
		verification = "Matrix-Result not correct, Sum of Errors > Threshold!";
	}
	else {
		verification = "Matrix-Result correct, Sum of Errors < Threshold!";
	}
	//printf("Result is correct!\n");
	return verification;
}

/************* Test Scripts for Multiplications ***************/
// Sequential Test Script
void sequentialMultiplyTest(int dimension, int iterations) {
	FILE* fp;
	fp = fopen("SequentialMultiplyTest.txt", "a+");

	// Console write
	printf("----------------------------------\n");
	printf("Test : Sequential Multiply        \n");
	printf("----------------------------------\n");
	printf("Dimension : %d\n", dimension);
	printf("..................................\n");

	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : Sequential Multiply        \n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension : %d\n", dimension);
	fprintf(fp, "..................................\n");

	double* opmLatency = malloc(iterations * sizeof(double));
	double** matrixA = randomSquareMatrix(dimension);
	double** matrixB = randomSquareMatrix(dimension);

	// Iterate and measure performance
	int i;
	for(i=0; i<iterations; i++) {
		double** matrixResult = zeroSquareMatrix(dimension);
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

	double sum = 0.0;
	double sumSquared = 0.0;

	// Statistical analyze
	int i;
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
void parallelMultiplyTest(int dimension, int iterations, int numThreads) {
	FILE* fp;
	fp = fopen("ParallelMultiplyTest.txt", "a+");

	// Console write
	printf("----------------------------------\n");
	printf("Test : Parallel Multiply          \n");
	printf("----------------------------------\n");
	printf("Dimension : %d\n", dimension);
	printf("..................................\n");

	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : Parallel Multiply          \n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension : %d\n", dimension);
	fprintf(fp, "..................................\n");

	double* opmLatency = malloc(iterations * sizeof(double));
	double* matrixCheck = malloc(iterations * sizeof(double));
	double** matrixA = randomSquareMatrix(dimension);
	double** matrixB = randomSquareMatrix(dimension);

	// Iterate and measure performance
	int i;
	for(i=0; i<iterations; i++) {
		double** matrixResult = zeroSquareMatrix(dimension);
		opmLatency[i] = parallelMultiply(matrixA, matrixB, matrixResult, dimension, numThreads);
		// check the result of the simple OpenMP parallel Multiplication
		matrixCheck[i] = verifyMultiplication_omp(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		// Console write
		printf("%d.\t%f\t%f\n", i+1, opmLatency[i], matrixCheck[i]);
		// File write
		fprintf(fp, "%d.\t%f\t%f\n", i+1, opmLatency[i], matrixCheck[i]);
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
	int i;
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
	free(matrixCheck);
	free(matrixA);
	free(matrixB);
}

// optimized parallel OpenMP Test Script
void optimizedParallelMultiplyTest(int dimension, int iterations, int numThreads) {
	FILE* fp;
	fp = fopen("OptimizedParallelMultiplyTest.txt", "a+");

	// Console write
	printf("----------------------------------\n");
	printf("Test : Optimized Parallel Multiply\n");
	printf("----------------------------------\n");
	printf("Dimension : %d\n", dimension);
	printf("..................................\n");

	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : Optimized Parallel Multiply\n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension : %d\n", dimension);
	fprintf(fp, "..................................\n");

	double* opmLatency = malloc(iterations * sizeof(double));
	double* matrixCheck = malloc(iterations * sizeof(double));
	double** matrixA = randomSquareMatrix(dimension);
	double** matrixB = randomSquareMatrix(dimension);

	// Iterate and measure performance
	int i;
	for(i=0; i<iterations; i++) {
		double** matrixResult = zeroSquareMatrix(dimension);
		opmLatency[i] = optimizedParallelMultiply(matrixA, matrixB, matrixResult, dimension, numThreads);
		// check the result of the simple OpenMP parallel Multiplication
		matrixCheck[i] = verifyMultiplication_omp_opti(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		// Console write
		printf("%d.\t%f\t%f\n", i+1, opmLatency[i], matrixCheck[i]);
		// File write
		fprintf(fp, "%d.\t%f\t%f\n", i+1, opmLatency[i], matrixCheck[i]);
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
	int i;
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
	free(matrixCheck);
	free(matrixA);
	free(matrixB);
}
