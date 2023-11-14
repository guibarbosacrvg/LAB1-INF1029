#include "matrix_lib.h"
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

// global variables (consider making them constants)
size_t NUM_THREADS = 256; // You can adjust the number of threads per block
size_t NUM_BLOCKS = 256;  // You can adjust the number of blocks
size_t MAX_MEMORY = 1024 * 1024 * 1024; // You can adjust the maximum memory size in bytes (1 GB in this example)

int set_grid_size(size_t num_threads, size_t num_blocks, size_t max_memory) {
    // Check if the input values are valid
    if (num_threads == 0 || num_blocks == 0 || max_memory == 0) {
        return 0;
    }

    // Update the global variables
    NUM_THREADS = num_threads;
    NUM_BLOCKS = num_blocks;
    MAX_MEMORY = max_memory;

    return 1;
}

size_t get_array_size(Matrix *matrix) {
    printf("%ld - %ld\n", (size_t)matrix->width, (size_t)matrix->height);
    return (size_t)matrix->width * (size_t) matrix->height;
}

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
    }
}

__global__ void scalarMatrixMultKernel(float scalar_value, float* d_rows, size_t copy_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < copy_size; i += blockDim.x * gridDim.x) {
        d_rows[i] *= scalar_value;
    }
}

int scalar_matrix_mult(float scalar_value, Matrix *matrix) {
    cudaError_t error;

    if (matrix->width > 0 && matrix->height > 0 && matrix->h_rows != NULL && matrix->d_rows != NULL) {
        const size_t array_size = get_array_size(matrix);
        size_t remaining, copy_size;

        for (size_t i = 0; i < array_size; i += MAX_MEMORY) {
            remaining = array_size - i;
            copy_size = (remaining < MAX_MEMORY) ? remaining : MAX_MEMORY;
            // Copy data from host to device
            error = cudaMemcpy(matrix->d_rows + i, matrix->h_rows + i, copy_size * sizeof(float), cudaMemcpyHostToDevice);
            checkCudaError(error);

            // Launch the kernel
            scalarMatrixMultKernel<<<NUM_BLOCKS, NUM_THREADS>>>(scalar_value, matrix->d_rows + i, copy_size);

            // Copy the result back from device to host
            error = cudaMemcpy(matrix->h_rows + i, matrix->d_rows + i, copy_size * sizeof(float), cudaMemcpyDeviceToHost);
            checkCudaError(error);
        }
        error = cudaFree(matrix->d_rows);
        checkCudaError(error);
    
        return 1;
    }

    return 0;
}


__global__
void matrixMatrixMultKernel(float* d_rows, float* d_columns, float* d_result, size_t copy_size, size_t width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < copy_size; i += blockDim.x * gridDim.x) {
        int row = i / width;
        int column = i % width;
        float sum = 0;
        for (int j = 0; j < width; j++) {
            sum += d_rows[row * width + j] * d_columns[j * width + column];
        }
        d_result[i] = sum;
    }
}

int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC) {
    cudaError_t error;

    if (matrixA->width > 0 && matrixA->height > 0 && matrixA->h_rows != NULL && matrixA->d_rows != NULL &&
        matrixB->width > 0 && matrixB->height > 0 && matrixB->h_rows != NULL && matrixB->d_rows != NULL &&
        matrixC->width > 0 && matrixC->height > 0 && matrixC->h_rows != NULL && matrixC->d_rows != NULL) {
        const size_t array_size = get_array_size(matrixC);
        size_t remaining, copy_size;

        for (size_t i = 0; i < array_size; i += MAX_MEMORY) {
            remaining = array_size - i;
            copy_size = (remaining < MAX_MEMORY) ? remaining : MAX_MEMORY;
            // Copy data from host to device
            error = cudaMemcpy(matrixA->d_rows + i, matrixA->h_rows + i, copy_size * sizeof(float), cudaMemcpyHostToDevice);
            checkCudaError(error);
            error = cudaMemcpy(matrixB->d_rows + i, matrixB->h_rows + i, copy_size * sizeof(float), cudaMemcpyHostToDevice);
            checkCudaError(error);

            // Launch the kernel
            matrixMatrixMultKernel<<<NUM_BLOCKS, NUM_THREADS>>>(matrixA->d_rows + i, matrixB->d_rows + i, matrixC->d_rows + i, copy_size, matrixC->width);

            // Copy the result back from device to host
            error = cudaMemcpy(matrixC->h_rows + i, matrixC->d_rows + i, copy_size * sizeof(float), cudaMemcpyDeviceToHost);
            checkCudaError(error);
        }
        error = cudaFree(matrixA->d_rows);
        checkCudaError(error);
        error = cudaFree(matrixB->d_rows);
        checkCudaError(error);
        error = cudaFree(matrixC->d_rows);
        checkCudaError(error);

        return 1;
    }

    return 0;
}

int bago(){
    return (float)1;
}