#include <stdio.h>

#include "matrix.h"

int main() {
    matrix mat;
    allocate_matrix(&mat, 3, 4);
    printf("Matrix columns: %d, rows: %d\n", mat.cols, mat.rows);
    printf("Matrix data pointer before free: %p\n", (void*)mat.data);

    // Fill the matrix with some values
    for (int j = 0; j < mat.rows; j++) {
        for (int i = 0; i < mat.cols; i++) {
            mat.data[j][i] = j * mat.cols + i;
        }
    }

    print_matrix(&mat);
    free_matrix(&mat);

    printf("Matrix data pointer after free: %p\n", (void*)mat.data);
    printf("DONE\n");
    return 0;
}