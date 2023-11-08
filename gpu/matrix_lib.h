#include <stddef.h>

typedef struct matrix{
    unsigned long int height;
    unsigned long int width;
    float *h_rows; // host pointer
    float *d_rows; // device pointer
    int alloc_mode; // 1: FULL_ALLOC, 0: PARTIAL_ALLOC
}Matrix;

int scalar_matrix_mult(float scalar_value, Matrix *matrix);
int set_grid_size(size_t num_threads, size_t num_blocks, size_t max_memory);