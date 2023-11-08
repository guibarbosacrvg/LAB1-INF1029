#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "matrix_lib.h"
#include <math.h>

float scalar_value = 0.0f;

// Function to evaluate the result of the matrix multiplication
int evaluate_matrix_matrix_mult(Matrix* matrixA, Matrix* matrixB, Matrix* matrixC){
    if(matrixA != NULL && matrixB != NULL && matrixC != NULL){
        for(int i = 0; i < matrixC->height; i++){
            for(int j = 0; j < matrixC->width; j++){
                float result = 0;
                for(int k = 0; k < matrixA->width; k++){
                    result += matrixA->h_rows[i * matrixA->width + k] * matrixB->h_rows[k * matrixB->width + j];
                }
                if(matrixC->h_rows[i * matrixC->width + j] != result){
                    return 0;
                }
            }
        }
        return 1;
    }
    return 0;
}

// Function to evaluate the result of the scalar matrix multiplication
int evaluate_scalar_matrix_mult(float scalar_value, Matrix* matrix){
    if(matrix != NULL){
        for(int i = 0; i < matrix->height; i++){
            for(int j = 0; j < matrix->width; j++){
                if(matrix->h_rows[i * matrix->width + j] != scalar_value * matrix->h_rows[i * matrix->width + j]){
                    return 0;
                }
            }
        }
        return 1;
    }
    return 0;
}

int read_matrix_from_binary(char* file, Matrix* target, size_t height, size_t width){
    FILE* fp = fopen(file, "rb");
    if(fp == NULL){
        printf("Error: Could not open file location %s\n", file);
        return 0;
    }

    // reading lines from the .dat file and storing them in the target matrix
    fread(target->h_rows, sizeof(float), height * width, fp);
    return 1;
}

// Function to print the matrix
void print_matrix(Matrix* matrix){
    int cont = 0;
    for(int i = 0; i < matrix->height; i++){
        for(int j = 0; j < matrix->width; j++){
            if(cont == 256){
                printf("\nOutput too large, printing first 256 values\n");
                return;
            }
            printf("%.2f ", matrix->h_rows[i * matrix->width + j]);
            cont++;
        }
        printf("\n");
    }
}

Matrix* alloc_matrix(size_t height, size_t width){
    cudaError_t cuda_error;
    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
    matrix->height = height;
    matrix->width = width;
    matrix->h_rows = (float*) malloc(sizeof(float) * height * width);
    cudaMalloc((void**) &matrix->d_rows, sizeof(float) * height * width);
    cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(cuda_error));
        return NULL;
    }
    return matrix;
}

void free_matrix(Matrix* matrix){
    cudaError_t cuda_error;
    free(matrix->h_rows);
    cudaFree(matrix->d_rows);
    cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(cuda_error));
        return;
    }
    free(matrix);
}

int main(int argc, char* argv[]){
    size_t DimA_M, DimA_N, DimB_M, DimB_N, num_threads, num_blocks, max_memory;
    char* matrixA_filename, *matrixB_filename, *result_filename, *result2_filename;
    char* eptr = NULL;
    struct timeval start, stop, overall_t1, overall_t2;

    FILE* output_file = fopen("output.txt", "w");

    if (output_file == NULL){
        perror("Error: Could not open output file");
        return -1;
    }

    // Marking the overall time used
    gettimeofday(&overall_t1, NULL);

    // Checking if the number of arguments is correct
    if(argc != 12){
        perror("Error: Invalid number of arguments: ");
        printf("%d", argc);
        return -1;
    }

    // Reading the arguments
    scalar_value = strtof(argv[1], &eptr);
    printf("Scalar value: %f\n", scalar_value);
    DimA_M = strtoul(argv[2], &eptr, 10);
    printf("DimA_M: %d\n", (int)DimA_M);
    DimA_N = strtoul(argv[3], &eptr, 10);
    printf("DimA_N: %d\n", (int)DimA_N);
    DimB_M = strtoul(argv[4], &eptr, 10);
    printf("DimB_M: %d\n", (int)DimB_M);
    DimB_N = strtoul(argv[5], &eptr, 10);
    printf("DimB_N: %d\n", (int)DimB_N);
    num_threads = strtoul(argv[6], &eptr, 10);
    printf("Num_threads: %d\n", (int)num_threads);
    num_blocks = strtoul(argv[7], &eptr, 10);
    printf("Num_blocks: %d\n", (int)num_blocks);
    max_memory = strtoul(argv[8], &eptr, 10);
    printf("Max_memory: %d\n", (int)max_memory);
    matrixA_filename = argv[9];
    printf("MatrixA_filename: %s\n", matrixA_filename);
    matrixB_filename = argv[10];
    printf("MatrixB_filename: %s\n", matrixB_filename);
    result_filename = argv[11];
    printf("Result_filename: %s\n", result_filename);
    
    // Checking if the arguments are valid
    if(scalar_value == 0.0f){
        printf("Error: Invalid scalar value\n");
        return -1;
    }
    if(DimA_M == 0 || DimA_N == 0 || DimB_M == 0 || DimB_N == 0){
        printf("Error: Invalid matrix dimensions\n");
        return -1;
    }
    if(num_threads == 0 || num_blocks == 0){
        printf("Error: Invalid number of threads or blocks\n");
        return -1;
    }
    if(max_memory == 0){
        printf("Error: Invalid maximum memory\n");
        return -1;
    }

    // Checking if the matrix dimensions are valid
    if(DimA_N != DimB_M){
        printf("Error: Matrix dimensions are not valid\n");
        return -1;
    }

    // Allocating the matrices
    Matrix* matrixA = alloc_matrix(DimA_M, DimA_N);
    Matrix* matrixB = alloc_matrix(DimB_M, DimB_N);
    Matrix* matrixC = alloc_matrix(DimA_M, DimB_N);

    // Reading the matrices from the binary files
    printf("Reading matrix A from file %s\n", matrixA_filename);
    if(!read_matrix_from_binary(matrixA_filename, matrixA, DimA_M, DimA_N)){
        printf("Error: Could not read matrix A from file %s\n", matrixA_filename);
        return -1;
    }
    
    printf("Reading matrix B from file %s\n", matrixB_filename);
    if(!read_matrix_from_binary(matrixB_filename, matrixB, DimB_M, DimB_N)){
        printf("Error: Could not read matrix B from file %s\n", matrixB_filename);
        return -1;
    }

  // Marking the start of the scalar matrix multiplication
    gettimeofday(&start, NULL);

    // Setting the number of threads and blocks
    set_grid_size(num_threads, num_blocks, max_memory);

    // Performing the proper scalar multiplication

    // Marking the end of the scalar matrix multiplication
    if(scalar_matrix_mult(scalar_value, matrixA) != 1){
        printf("Error: Could not perform scalar matrix multiplication\n");
        return -1;
    }

    gettimeofday(&stop, NULL);

    // Displaying the time used for the scalar matrix multiplication
    printf("Time used for scalar matrix multiplication: %f seconds\n", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000000.0f);

    // Printing the result of the scalar matrix multiplication
    printf("Printing the result of the scalar matrix multiplication\n");
    print_matrix(matrixA);

    // Evaluating the result of the scalar matrix multiplication
    printf("Evaluating the result of the scalar matrix multiplication\n");
    if(!evaluate_scalar_matrix_mult(scalar_value, matrixA)){
        printf("Error: The result of the scalar matrix multiplication is not correct\n");
    }

    return 1;
}