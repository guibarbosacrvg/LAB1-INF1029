#include "matrix_lib.h"
#include <stddef.h>
#include <stdio.h>

// global variables
size_t NUM_THREADS = 1;
size_t NUM_BLOCKS = 1;
size_t MAX_MEMORY = 1;

int set_grid_size(size_t num_threads, size_t num_blocks, size_t max_memory){
    if(num_threads == 0 || num_blocks == 0 || max_memory == 0){
        return 0;
    }
    NUM_THREADS = num_threads;
    NUM_BLOCKS = num_blocks;
    MAX_MEMORY = max_memory;
    return 1;
}

size_t get_array_size(Matrix *matrix){
    return sizeof(float) * (size_t) matrix->width * matrix->height;
}

void checkCudaError(cudaError_t error){
    if(error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    }
}

__global__ void scalarMatrixMultKernel(float scalar_value, float* d_rows, size_t array_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < array_size; i += blockDim.x * gridDim.x) {
        d_rows[i] *= scalar_value;
    }
}


int scalar_matrix_mult(float scalar_value, Matrix *matrix){
    cudaError_t error;
    if(matrix->width > 0 && matrix->height > 0 && matrix->h_rows != NULL && matrix->d_rows != NULL){    
        size_t array_size = get_array_size(matrix);
        size_t remaining = array_size;
        size_t copy_size;
        float* d_scalar_value;
        error = cudaMalloc((void**)&d_scalar_value, sizeof(float));
        checkCudaError(error);
        error = cudaMemcpy(d_scalar_value, &scalar_value, sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError(error);
        for(size_t i = 0; i < array_size; i += MAX_MEMORY){
            remaining --;
            copy_size = (remaining < MAX_MEMORY) ? remaining : MAX_MEMORY;
            error = cudaMemcpy(matrix->d_rows + i, matrix->h_rows + i, copy_size * sizeof(float), cudaMemcpyHostToDevice);
            checkCudaError(error);
            scalarMatrixMultKernel<<<NUM_BLOCKS, NUM_THREADS>>>(scalar_value, matrix->d_rows + i, copy_size);
            error = cudaMemcpy(matrix->h_rows + i, matrix->d_rows + i, copy_size * sizeof(float), cudaMemcpyDeviceToHost);
            checkCudaError(error);
            error = cudaFree(matrix->d_rows + i);
            checkCudaError(error);
        }
        cudaDeviceSynchronize();
        return 1;
    }
    return 0;
}

