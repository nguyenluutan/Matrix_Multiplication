/*
originally from: https://gist.github.com/metallurgix/0dfafc03215ce89fc595
Code especially for OpenMP !!!
*/
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define N 50

double A[N][N];
double B[N][N];
double C[N][N];

int main() {
	int i, j, k;
	double elapsed;
  struct timeval tv1, tv2;
  struct timezone tz;

  omp_set_num_threads(omp_get_num_procs());

	srand(time(0));

	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			A[i][j] = (double)rand() / (double)((unsigned)RAND_MAX);
			B[i][j] = (double)rand() / (double)((unsigned)RAND_MAX);
		}
	}

	gettimeofday(&tv1, &tz);

  #pragma omp parallel for private(i, j, k) shared(A, B, C)
  for (i=0; i<N; ++i) {
		for (j=0; j<N; ++j) {
			for (k=0; k<N; ++k) {
				C[i][j] += A[i][k] * B[k][j];
      }
    }
  }


  gettimeofday(&tv2, &tz);

  elapsed = (double)(tv2.tv_sec-tv1.tv_sec) + (double)(tv2.tv_usec-tv1.tv_usec) * 1.e-6;
  printf("elapsed time = %f seconds.\n", elapsed);

  for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			printf("%.2f\t", C[i][j]);
    }
    printf("\n");
  }
}
