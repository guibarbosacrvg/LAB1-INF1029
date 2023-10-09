#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include "timer.h"
#include "matrix_lib.h"
#define VEC_STEP 8


// Defining a global variable to be used as the number of threads
int NUM_THREADS = 1;

void set_num_threads(int n){
    NUM_THREADS = n;
}

int scalar_matrix_mult_first(float scalar_value, Matrix* matrix){
    if(matrix != NULL){
        for(int i = 0; i < matrix->height; i++){
            for(int j = 0; j < matrix->width; j++){
                matrix->rows[i * matrix->width + j] *= scalar_value;
            }
        }
        return 1;
    }
    return 0;
}

// Optimized version of matrix_matrix_mult function, iterating over the columns of matrixB instead of the rows, to improve cache performance
int matrix_matrix_mult_optimized(Matrix* matrixA, Matrix* matrixB, Matrix* matrixC){
    if(matrixA != NULL && matrixB != NULL && matrixC != NULL){
        for (int i = 0; i < matrixC->height; i++) {
            for (int j = 0; j < matrixC->width; j++) {
                matrixC->rows[i * matrixC->width + j] = 0;
            }
        }
        for (int i = 0; i < matrixC->height; i++) {
            for (int k = 0; k < matrixA->width; k++) {
                for (int j = 0; j < matrixC->width; j++) {
                    matrixC->rows[i * matrixC->width + j] += matrixA->rows[i * matrixA->width + k] * matrixB->rows[k * matrixB->width + j];
                }
            }
        }
        return 1;
    }
    return 0;
}


// Implementation using intel vectorial processing to instead of processing the vector values one by one, process 8 values at a time
int scalar_matrix_mult_amx(float scalar_value, Matrix* matrix){
    if(matrix != NULL){
        __m256 scalar, vec, result;
        float * vec_next = matrix->rows;
        scalar = _mm256_set1_ps(scalar_value);

        for(int i = 0; i < matrix->height * matrix->width; i += VEC_STEP, vec_next += VEC_STEP){
            vec = _mm256_load_ps(vec_next);
            result = _mm256_mul_ps(scalar, vec);
            _mm256_store_ps(matrix->rows + i, result);
        }
        return 1;
    }
    return 0;
}


typedef struct scalar_args{
    int id;
    float scalar;
    Matrix *matrix;
} scalar_args;

void *scalar_mult_thread(void *threadid){
    scalar_args *arguments = (scalar_args *) threadid;
    int id = arguments->id;
    float scalar = arguments->scalar;
    Matrix* matrix = arguments->matrix;
    int start = id * matrix->height / NUM_THREADS;
    int end = (id + 1) * matrix->height / NUM_THREADS;
    __m256 scalar_value, vec, result;
    float * vec_next = matrix->rows;
    scalar_value = _mm256_set1_ps(scalar);
    for(int i = 0; i < matrix->height * matrix->width; i += VEC_STEP, vec_next += VEC_STEP){
        vec = _mm256_load_ps(vec_next);
        result = _mm256_mul_ps(scalar_value, vec);
        _mm256_store_ps(matrix->rows + i, result);
    }
    pthread_exit(NULL);
}


// Implementation of the scalar multiplication using the pthread lib
int scalar_matrix_mult(float scalar_value, Matrix *matrix){
    if(matrix != NULL){
        pthread_t threads[NUM_THREADS];
        scalar_args arguments[NUM_THREADS];
        int t = 0;
        for(t = 0; t < NUM_THREADS; t++){
            arguments[t].id = t;
            arguments[t].scalar = scalar_value;
            arguments[t].matrix = matrix;
            pthread_create(&threads[t], NULL, scalar_mult_thread, (void *) &arguments[t]);
        }

        for(t = 0; t < NUM_THREADS; t++){
            pthread_join(threads[t], NULL);
        }
        return 1;
    }
}


// Implementation using intel vectorial processing to instead of processing the vector values one by one, process 8 values at a time
int matrix_matrix_mult_amx(Matrix* matrixA, Matrix* matrixB, Matrix* matrixC){
    if(matrixA->width != matrixB->height || matrixA->height != matrixC->height || matrixB->width != matrixC->width){
        return 0;
    }

    float * vec_next = matrixC->rows;
    __m256 vec, result, vecA, vecB;
    
    for(int i = 0; i < matrixC->height; i++){
        for(int j = 0; j < matrixC->width; j += VEC_STEP, vec_next += VEC_STEP){
            result = _mm256_setzero_ps();
            for(int k = 0; k < matrixA->width; k++){
                vecA = _mm256_set1_ps(matrixA->rows[i * matrixA->width + k]);
                vecB = _mm256_load_ps(matrixB->rows + k * matrixB->width + j);
                vec = _mm256_mul_ps(vecA, vecB);
                result = _mm256_add_ps(result, vec);
            }
            _mm256_store_ps(matrixC->rows + i * matrixC->width + j, result);
        }
    }

    return 1;
}


// Further optimization of the matrix_matrix_mult function, using the AVX2 FMA instruction to perform the multiplication and addition in a single instruction
int matrix_matrix_mult_fma(Matrix* matrixA, Matrix* matrixB, Matrix* matrixC){
    if(matrixA->width != matrixB->height || matrixA->height != matrixC->height || matrixB->width != matrixC->width){
        return 0;
    }

    float * vec_next = matrixC->rows;
    __m256 vec, result, vecA, vecB;
    
    for(int i = 0; i < matrixC->height; i++){
        for(int j = 0; j < matrixC->width; j += VEC_STEP, vec_next += VEC_STEP){
            result = _mm256_setzero_ps();
            for(int k = 0; k < matrixA->width; k++){
                vecA = _mm256_set1_ps(matrixA->rows[i * matrixA->width + k]);
                vecB = _mm256_load_ps(matrixB->rows + k * matrixB->width + j);
                vec = _mm256_fmadd_ps(vecA, vecB, result);
                result = vec;
            }
            _mm256_store_ps(matrixC->rows + i * matrixC->width + j, result);
        }
    }

    return 1;
}

typedef struct args{
    int id;
    Matrix* matrixA;
    Matrix* matrixB;
    Matrix* matrixC;
} args;

void *matrix_matrix_mult_thread(void *threadid){
    args *arguments = (args *) threadid;
    int id = arguments->id;
    Matrix* matrixA = arguments->matrixA;
    Matrix* matrixB = arguments->matrixB;
    Matrix* matrixC = arguments->matrixC;

    int start = id * matrixC->height / NUM_THREADS;
    int end = (id + 1) * matrixC->height / NUM_THREADS;

    float * vec_next = matrixC->rows + start * matrixC->width;
    __m256 vec, result, vecA, vecB;
    
    for(int i = start; i < end; i++){
        for(int j = 0; j < matrixC->width; j += VEC_STEP, vec_next += VEC_STEP){
            result = _mm256_setzero_ps();
            for(int k = 0; k < matrixA->width; k++){
                vecA = _mm256_set1_ps(matrixA->rows[i * matrixA->width + k]);
                vecB = _mm256_load_ps(matrixB->rows + k * matrixB->width + j);
                vec = _mm256_fmadd_ps(vecA, vecB, result);
                result = vec;
            }
            _mm256_store_ps(matrixC->rows + i * matrixC->width + j, result);
        }
    }
    pthread_exit(NULL);
} 


int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC){
    if(matrixA->width != matrixB->height || matrixA->height != matrixC->height || matrixB->width != matrixC->width){
        return 0;
    }

    pthread_t threads[NUM_THREADS];
    args arguments[NUM_THREADS];
    int t = 0;
    for(t = 0; t < NUM_THREADS; t++){
        arguments[t].id = t;
        arguments[t].matrixA = matrixA;
        arguments[t].matrixB = matrixB;
        arguments[t].matrixC = matrixC;
        pthread_create(&threads[t], NULL, matrix_matrix_mult_thread, (void *) &arguments[t]);
    }

    for(t = 0; t < NUM_THREADS; t++){
        pthread_join(threads[t], NULL);
    }

    return 1;
}
