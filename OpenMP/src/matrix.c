#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

void print_matrix(matrix* matrix) {
    for (int j = 0; j < matrix->rows; j++) {
        for (int i = 0; i < matrix->cols; i++) {
            printf("%d ", matrix->data[j][i]);
        }
        printf("\n");
    }
}

void free_matrix(matrix* matrix) {
    if (matrix->data != NULL) {
        free(matrix->data);
        matrix->data = NULL;
        printf("Matrix memory freed successfully.\n");
    }
}

void allocate_matrix(matrix* matrix, int rows, int cols) {
    matrix->rows = rows;
    matrix->cols = cols;

    size_t total_size = rows*sizeof(u_int8_t*) + cols*rows*sizeof(u_int8_t);
    matrix->data = (u_int8_t**)malloc(total_size);
    memset(matrix->data, 0, total_size);

    matrix->data[0] = (u_int8_t*)(matrix->data + rows);
    for (int j = 1; j < rows; j++) {
        matrix->data[j] = matrix->data[j-1] + cols;
    }
}

void copy_matrix(matrix* src, matrix* dest) {
    allocate_matrix(dest, src->rows, src->cols);
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i][j] = src->data[i][j];
        }
    }
}
