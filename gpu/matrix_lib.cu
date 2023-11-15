#include "matrix_lib.h"
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

// global variables (consider making them constants)
size_t NUM_THREADS = 256; // You can adjust the number of threads per block
size_t NUM_BLOCKS = 256;  // You can adjust the number of blocks
size_t MAX_MEMORY = 1024 * 1024 * 1024; // You can adjust the maximum memory size in bytes (1 GB in this example)
cudaError_t cudaError;

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

__global__ 
void mult_scalar(int n, float *matrix_rows, float scalar_value)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    matrix_rows[i] = matrix_rows[i] * scalar_value;
  }
}

/** Multiplica matriz por um valor fornecido utilizando a GPU. */
int scalar_matrix_mult(float scalar_value, Matrix * matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  int matrix_size = matrix->height * matrix->width;

  int loop_limit = (matrix_size + MAX_MEMORY - 1) / MAX_MEMORY;
  int chunk_size = MAX_MEMORY;

  for(int count = 0; count < loop_limit; ++count){
    if(matrix_size % MAX_MEMORY != 0 && count == loop_limit - 1){
      chunk_size = matrix_size % MAX_MEMORY;
    }
    
    cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h -> d) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    int blockSize = NUM_THREADS;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;
    
    if (numBlocks > NUM_BLOCKS){
      numBlocks = NUM_BLOCKS;
    }

    mult_scalar<<<numBlocks, blockSize>>>(chunk_size, matrix->d_rows, scalar_value);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaError = cudaMemcpy(matrix->h_rows+(count*chunk_size), matrix->d_rows, chunk_size*sizeof(float), cudaMemcpyDeviceToHost);
  
    if (cudaError != cudaSuccess){
      printf("cudaMemcpy (d -> h) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }
  }

  return 1;
}



/** ----------MATRIX MATRIX MULT---------- **/

// Kernel function to mult to array
__global__ 
void matrix_mult(int n, int matrix_a_width,int matrix_a_height, int matrix_b_height, int matrix_b_width, float * matrix_a_rows, float * matrix_b_rows, float * matrix_c_rows)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i+= stride){

    for(int k =0; k < matrix_a_width; k++){
      matrix_c_rows[i] += matrix_a_rows[matrix_a_width*(i/matrix_a_height) + k] * matrix_b_rows[(i%matrix_a_height) + k*matrix_b_width];
    }
  }

}

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada utilizando a GPU. */
int matrix_matrix_mult(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){

  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  int matrix_size = matrix_a->height * matrix_a->width;

  int loop_limit = (matrix_size + MAX_MEMORY - 1) / MAX_MEMORY;
  int chunk_size = MAX_MEMORY;
  for(int count = 0; count < loop_limit; ++count){
    if(matrix_size % MAX_MEMORY != 0 && count == loop_limit - 1){
      chunk_size = matrix_size % MAX_MEMORY;
    }
    
    cudaError = cudaMemcpy(matrix_a->d_rows, matrix_a->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h_a -> d_a) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    cudaError = cudaMemcpy(matrix_b->d_rows, matrix_b->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h_b -> d_b) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    cudaError = cudaMemcpy(matrix_c->d_rows, matrix_c->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h_c -> d_c) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    int blockSize = NUM_THREADS;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;
    if (numBlocks > NUM_BLOCKS) {
      numBlocks = NUM_BLOCKS;
    }
  
    matrix_mult<<<numBlocks, blockSize>>>(chunk_size, matrix_a->width, matrix_a->height, matrix_b->height, matrix_b->width, matrix_a->d_rows, matrix_b->d_rows, matrix_c->d_rows);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaError = cudaMemcpy(matrix_c->h_rows+(count*chunk_size), matrix_c->d_rows, chunk_size*sizeof(float), cudaMemcpyDeviceToHost);
  
    if (cudaError != cudaSuccess){
      printf("cudaMemcpy (d_c -> h_c) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }
  }

  return 1;
}