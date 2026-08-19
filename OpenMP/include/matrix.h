#ifndef MATRIX_H
#define MATRIX_H

#include <sys/types.h>

typedef struct {
    int rows;
    int cols;
    u_int8_t** data;
} matrix;

void allocate_matrix(matrix* matrix, int rows, int cols);
void free_matrix(matrix* matrix);
void print_matrix(matrix* matrix);
void copy_matrix(matrix* src, matrix* dest);
// Variante senza direttive OpenMP, per il baseline sequenziale.
void copy_matrix_serial(matrix* src, matrix* dest);
void move_matrix_data(matrix* dst, matrix* src);

#endif