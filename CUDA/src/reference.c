#include <stdio.h>
#include <stdlib.h>
#include "reference.h"

static void apply_operator(matrix* src, matrix* dst, matrix* se, int is_erosion) {
    int rows = src->rows;
    int cols = src->cols;
    int se_rows = se->rows;
    int se_cols = se->cols;
    int radius_y = se_rows / 2;
    int radius_x = se_cols / 2;

    u_int8_t neutral = is_erosion ? 255 : 0;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            u_int8_t acc = neutral;

            for (int i = 0; i < se_rows; i++) {
                for (int j = 0; j < se_cols; j++) {
                    int img_row = row + i - radius_y;
                    int img_col = col + j - radius_x;

                    if (img_row < 0 || img_row >= rows || img_col < 0 || img_col >= cols) continue;
                    if (se->data[i][j] != 1) continue;

                    u_int8_t v = src->data[img_row][img_col];
                    if (is_erosion) {
                        if (v < acc) acc = v;
                    } else {
                        if (v > acc) acc = v;
                    }
                }
            }
            dst->data[row][col] = acc;
        }
    }
}

// Scambia soltanto i buffer: righe e colonne non cambiano durante la morfologia.
static void swap_data(matrix* a, matrix* b) {
    u_int8_t** tmp = a->data;
    a->data = b->data;
    b->data = tmp;
}

static matrix* allocate_scratch(matrix** img, int size) {
    matrix* scratch = (matrix*)calloc((size_t)size, sizeof(matrix));
    if (!scratch) {
        fprintf(stderr, "Impossibile allocare i descrittori dei buffer CPU\n");
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < size; k++) {
        allocate_matrix(&scratch[k], img[k]->rows, img[k]->cols);
    }
    return scratch;
}

static void free_scratch(matrix* scratch, int size) {
    for (int k = 0; k < size; k++) free_matrix(&scratch[k]);
    free(scratch);
}

// L'operatore non è calcolabile in place: il riferimento mantiene quindi un
// buffer di appoggio per immagine e scambia i puntatori dopo ogni passata.
static void run_pass(matrix* img, matrix* scratch, matrix* se, int is_erosion) {
    apply_operator(img, scratch, se, is_erosion);
    swap_data(img, scratch);
}

void ref_erosion(matrix** img, matrix* structuring_element, int size) {
    if (size <= 0) return;

    matrix* scratch = allocate_scratch(img, size);
    for (int k = 0; k < size; k++) {
        run_pass(img[k], &scratch[k], structuring_element, 1);
    }
    free_scratch(scratch, size);
}

void ref_opening(matrix** img, matrix* structuring_element, int size) {
    if (size <= 0) return;

    matrix* scratch = allocate_scratch(img, size);
    for (int k = 0; k < size; k++) {
        run_pass(img[k], &scratch[k], structuring_element, 1);
        run_pass(img[k], &scratch[k], structuring_element, 0);
    }
    free_scratch(scratch, size);
}

int compare_batches(matrix** a, matrix** b, int size) {
    int diffs = 0;

    for (int k = 0; k < size; k++) {
        if (a[k]->rows != b[k]->rows || a[k]->cols != b[k]->cols) {
            fprintf(stderr, "confronto impossibile: immagine %d e' %dx%d contro %dx%d\n",
                    k, a[k]->rows, a[k]->cols, b[k]->rows, b[k]->cols);
            return -1;
        }
        for (int i = 0; i < a[k]->rows; i++) {
            for (int j = 0; j < a[k]->cols; j++) {
                if (a[k]->data[i][j] != b[k]->data[i][j]) diffs++;
            }
        }
    }
    return diffs;
}
