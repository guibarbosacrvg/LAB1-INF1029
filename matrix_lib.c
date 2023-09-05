#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "matrix_lib.h"

int scalar_matrix_mult_old(float scalar_value, Matrix* matrix){
    if(matrix != NULL){
        for(int i = 0; i < matrix->width; i++){
            for(int j = 0; j < matrix->height; j++){
                matrix->rows[i * matrix->width + j] *= scalar_value;
            }
        }
        return 1;
    }
    return 0;
}

// Optimizing the scalar matrix multiplication, iterating over the columns instead of the rows
int scalar_matrix_mult(float scalar_value, Matrix* matrix){
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

int matrix_matrix_mult_old(Matrix* matrixA, Matrix* matrixB, Matrix* matrixC){
    if(matrixA != NULL && matrixB != NULL){
        for (int i = 0; i < matrixC->height; i++) {
            for (int j = 0; j < matrixC->width; j++) {
                matrixC->rows[i * matrixC->width + j] = 0;
                for (int k = 0; k < matrixA->width; k++) {
                    matrixC->rows[i * matrixC->width + j] += matrixA->rows[i * matrixA->width + k] * matrixB->rows[k * matrixB->width + j];
                }
            }
        }
        return 1;
    }
    return 0;
}

// Optimized version of matrix_matrix_mult function, iterating over the columns of matrixB instead of the rows, to improve cache performance
int matrix_matrix_mult(Matrix* matrixA, Matrix* matrixB, Matrix* matrixC){
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
