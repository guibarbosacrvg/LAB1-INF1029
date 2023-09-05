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

void write_matrix(char* file, Matrix* target){
    FILE* fp = fopen(file, "wb");
    if(fp == NULL){
        printf("Error: Could not open file location %s\n", file);
        return;
    }

    // writing lines from the target matrix to the .dat file
    fwrite(target->rows, sizeof(float), target->height * target->width, fp);
    fclose(fp);
}

int main(int argc, char* argv[]){
    unsigned long int DimA_M, DimA_N, DimB_M, DimB_N;
    char* matrixA_filename, *matrixB_filename, *result_filename, *result2_filename;
    char* eptr = NULL;
    struct timeval start, stop, overall_t1, overall_t2;

    /*if(argc != 10){
        printf("Usage: %s <matrixA_filename> <matrixA_M> <matrixA_N> <matrixB_filename> <matrixB_M> <matrixB_N> <result_filename> <result2_filename> <scalar_value>\n", argv[0]);
        return 0;
    }*/

    // Allocating memory for the matrices
    
    Matrix *matrixA = alloc_matrix(256, 256);
    Matrix *matrixB = alloc_matrix(256, 256);
    Matrix *matrixC = alloc_matrix(256, 256);

    read_matrix_from_binary("data/matrix_inputA_256.dat", matrixA, 256, 256);
    read_matrix_from_binary("data/matrix_inputB_256.dat", matrixB, 256, 256);
    //print_matrix(matrixA);
    matrix_matrix_mult(matrixA, matrixB, matrixC);
    write_matrix("data/matrix_result1.dat", matrixC);
    read_matrix_from_binary("data/matrix_result1.dat", matrixC, 256, 256);
    print_matrix(matrixC);
    return 0;
}