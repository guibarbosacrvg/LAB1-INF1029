#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "timer.h"
#include "matrix_lib.h"

void print_matrix(Matrix* matrix){
    for(int i = 0; i < matrix->height; i++){
        for(int j = 0; j < matrix->width; j++){
            printf("%.2f ", matrix->rows[i * matrix->width + j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]){
    unsigned long int DimA_M, DimA_N, DimB_M, DimB_N;
    char* matrixA_filename, *matrixB_filename, *result_filename, *result2_filename;
    char* eptr = NULL;
    struct timeval start, stop, overall_t1, overall_t2;

    if(argc != 10){
        printf("Usage: %s <matrixA_filename> <matrixA_M> <matrixA_N> <matrixB_filename> <matrixB_M> <matrixB_N> <result_filename> <result2_filename> <scalar_value>\n", argv[0]);
        return 0;
    }

    // Allocating memory for the matrices
    Matrix* matrixA = (Matrix*) aligned_alloc(32, sizeof(Matrix));
    Matrix* matrixB = (Matrix*) aligned_alloc(32, sizeof(Matrix));
    Matrix* matrixC = (Matrix*) aligned_alloc(32, sizeof(Matrix));

    
}