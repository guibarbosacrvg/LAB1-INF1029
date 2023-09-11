#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include "timer.h"
#include "matrix_lib.h"

float scalar_value = 0.0f;

struct matrix matrixA, matrixB, matrixC;

int store_matrix(struct matrix *matrix, char *filename) {

}

int load_matrix(struct matrix *matrix, char *filename) {

}

int initialize_matrix(struct matrix *matrix, float value, float inc) {

}

int print_matrix(struct matrix *matrix) {

}

int check_errors(struct matrix *matrix, float scalar_value) {

}

int main(int argc, char *argv[]) {
  unsigned long int DimA_M, DimA_N, DimB_M, DimB_N;
  char *matrixA_filename, *matrixB_filename, *result1_filename, *result2_filename;
  char *eptr = NULL;
  struct timeval start, stop, overall_t1, overall_t2;

  // Mark overall start time
  gettimeofday(&overall_t1, NULL);

  // Check arguments
  if (argc != 10) {
        printf("Usage: %s <scalar_value> <DimA_M> <DimA_N> <DimB_M> <DimB_N> <matrixA_filename> <matrixB_filename> <result1_filename> <result2_filename>\n", argv[0]);
        return 0;
  }

  // Convert arguments

  /* Allocate the arrays of the four matrixes */

  /* Initialize the three matrixes */

  /* Scalar product of matrix A */
  printf("Executing scalar_matrix_mult(%5.1f, matrixA)...\n",scalar_value);
  gettimeofday(&start, NULL);
  if (!scalar_matrix_mult(scalar_value, &matrixA)) {
	printf("%s: scalar_matrix_mult problem.", argv[0]);
	return 1;
  }
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Print matrix */
  printf("---------- Matrix A ----------\n");
  print_matrix(&matrixA);

  /* Write first result */
  printf("Writing first result: %s...\n", result1_filename);
  if (!store_matrix(&matrixA, result1_filename)) {
	printf("%s: failed to write first result to file.", argv[0]);
	return 1;
  }

  /* Calculate the product between matrix A and matrix B */
  printf("Executing matrix_matrix_mult(matrixA, mattrixB, matrixC)...\n");
  gettimeofday(&start, NULL);
  if (!matrix_matrix_mult(&matrixA, &matrixB, &matrixC)) {
	printf("%s: matrix_matrix_mult problem.", argv[0]);
	return 1;
  }
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Print matrix */
  printf("---------- Matrix C ----------\n");
  print_matrix(&matrixC);

  /* Write second result */
  printf("Writing second result: %s...\n", result2_filename);
  if (!store_matrix(&matrixC, result2_filename)) {
	printf("%s: failed to write second result to file.", argv[0]);
	return 1;
  }

  /* Check foor errors */
  printf("Checking matrixC for errors...\n");
  gettimeofday(&start, NULL);
  check_errors(&matrixC, 10240.0f);
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Mark overall stop time
  gettimeofday(&overall_t2, NULL);

  // Show elapsed overall time
  printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));

  return 0;
}