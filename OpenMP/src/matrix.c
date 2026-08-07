#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

void print_matrix(matrix* matrix) {
    #pragma omp single
    for (int j = 0; j < matrix->rows; j++) {
        for (int i = 0; i < matrix->cols; i++) {
            printf("%d ", matrix->data[j][i]);
        }
        printf("\n");
    }
}

void free_matrix(matrix* matrix) {
    #pragma omp single
    {
        if (matrix->data != NULL) {
            free(matrix->data);
            matrix->data = NULL;
        }
    }
}

void allocate_matrix(matrix* matrix, int rows, int cols) {
    #pragma omp single
    {
        matrix->rows = rows;
        matrix->cols = cols;

        size_t total_size = rows*sizeof(u_int8_t*) + cols*rows*sizeof(u_int8_t);
        matrix->data = (u_int8_t**)malloc(total_size);
        memset(matrix->data, 0, total_size);

        matrix->data[0] = (u_int8_t*)(matrix->data + rows);
    }
    #pragma omp for
    for (int j = 1; j < rows; j++) {
        matrix->data[j] = matrix->data[0] + j * cols;
    }
}

void copy_matrix(matrix* src, matrix* dest) {
    allocate_matrix(dest, src->rows, src->cols);
    #pragma omp for collapse(2)
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i][j] = src->data[i][j];
        }
    }
}
