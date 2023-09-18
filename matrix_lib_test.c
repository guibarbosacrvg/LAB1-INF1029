#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cpuid.h>
#include <immintrin.h>
#include <unistd.h>
#include "timer.h"
#include "matrix_lib.h"

float scalar_value = 0.0f;

void print_matrix(Matrix* matrix){
    int cont = 0;
    for(int i = 0; i < matrix->height; i++){
        for(int j = 0; j < matrix->width; j++){
            if(cont == 256){
                printf("\nOutput too large, printing first 256 values\n");
                return;
            }
            printf("%.2f ", matrix->rows[i * matrix->width + j]);
            cont++;
        }
        printf("\n");
    }
}

int read_matrix_from_binary(char* file, Matrix* target, unsigned long int height, unsigned long int width){
    FILE* fp = fopen(file, "rb");
    if(fp == NULL){
        printf("Error: Could not open file location %s\n", file);
        return 0;
    }

    // reading lines from the .dat file and storing them in the target matrix
    fread(target->rows, sizeof(float), height * width, fp);
    return 1;
}

Matrix *alloc_matrix(unsigned long int height, unsigned long int width) {
    Matrix *target = (Matrix*) aligned_alloc(32, sizeof(Matrix));
    target->height = height;
    target->width = width;
    target->rows = (float*) aligned_alloc(32, sizeof(float) * height * width);
    return target;
}

int store_matrix(Matrix* target, char* file){
    FILE* fp = fopen(file, "wb");
    if(fp == NULL){
        printf("Error: Could not open file location %s\n", file);
        return 0;
    }

    // writing lines from the target matrix to the .dat file
    fwrite(target->rows, sizeof(float), target->height * target->width, fp);
    fclose(fp);
    return 1;
}

int main(int argc, char* argv[]){
    unsigned long int DimA_M, DimA_N, DimB_M, DimB_N;
    char* matrixA_filename, *matrixB_filename, *result_filename, *result2_filename;
    char* eptr = NULL;
    struct timeval start, stop, overall_t1, overall_t2;
    
    // Mark overall start time
    gettimeofday(&overall_t1, NULL);
    
    if (argc != 10) {
        printf("Usage: %s <scalar_value> <DimA_M> <DimA_N> <DimB_M> <DimB_N> <matrixA_filename> <matrixB_filename> <result1_filename> <result2_filename>\n", argv[0]);
        return 0;
    }

    DimA_M = strtoul(argv[2], &eptr, 10);
    printf("DimA_M: %lu\n", DimA_M);
    DimA_N = strtoul(argv[3], &eptr, 10);
    printf("DimA_N: %lu\n", DimA_N);
    DimB_M = strtoul(argv[4], &eptr, 10);
    printf("DimB_M: %lu\n", DimB_M);
    DimB_N = strtoul(argv[5], &eptr, 10);
    printf("DimB_N: %lu\n", DimB_N);
    matrixA_filename = argv[6];
    printf("matrixA_filename: %s\n", matrixA_filename);
    matrixB_filename = argv[7];
    printf("matrixB_filename: %s\n", matrixB_filename);

    scalar_value = strtof(argv[1], &eptr);

    result_filename = argv[8];

    result2_filename = argv[9];

    // Allocating memory for the matrices
    
    Matrix *matrixA = alloc_matrix(DimA_M, DimA_N);
    Matrix *matrixB = alloc_matrix(DimB_M, DimB_N);
    Matrix *matrixC = alloc_matrix(DimA_M, DimB_N);

    // Initialize the three matrixes
    printf("Reading matrix A from file: %s...\n", matrixA_filename);
    if (!read_matrix_from_binary(matrixA_filename, matrixA, DimA_M, DimA_N)) {
        printf("%s: failed to read matrix A from file.", argv[0]);
        return 1;
    }

    printf("Reading matrix B from file: %s...\n", matrixB_filename);
    if (!read_matrix_from_binary(matrixB_filename, matrixB, DimB_M, DimB_N)) {
        printf("%s: failed to read matrix B from file.", argv[0]);
        return 1;
    }    

    /* Scalar product of matrix A */
    printf("Executing scalar_matrix_mult(%5.1f, matrixA)...\n",scalar_value);
    gettimeofday(&start, NULL);
    if (!scalar_matrix_mult(scalar_value, matrixA)) {
	    printf("%s: scalar_matrix_mult problem.", argv[0]);
	    return 1;
    }
    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    /* Print matrix */
    printf("---------- Matrix A ----------\n");
    print_matrix(matrixA);

    printf("Writing first result: %s...\n", result_filename);
    if (!store_matrix(matrixA, result_filename)) {
        printf("%s: failed to write first result to file.", argv[0]);
        return 1;
    }

    /* Calculate the product between matrix A and matrix B */
    printf("Executing matrix_matrix_mult(matrixA, mattrixB, matrixC)...\n");
    gettimeofday(&start, NULL);
    if (!matrix_matrix_mult(matrixA, matrixB, matrixC)) {
        printf("%s: matrix_matrix_mult problem.", argv[0]);
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    /* Print matrix */
    printf("---------- Matrix C ----------\n");
    print_matrix(matrixC);

    /* Write second result */
    printf("Writing second result: %s...\n", result2_filename);
    if (!store_matrix(matrixC, result2_filename)) {
        printf("%s: failed to write second result to file.", argv[0]);
        return 1;
    }
        
    // Mark overall stop time
    gettimeofday(&overall_t2, NULL);

    // Show elapsed overall time
    printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));
    
    // Showing CPU Model using lscpu 
    execve("/bin/lscpu", NULL, NULL);
    return 0;
}