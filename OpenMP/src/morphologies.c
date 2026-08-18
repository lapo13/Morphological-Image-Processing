#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "morphologies.h"
#include "image.h"

double last_op_seconds = 0.0;
static double op_t_start, op_t_end;

static void erosion_helper (matrix* img, matrix* structuring_element, matrix* scratch, int vectorization) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;

    #pragma omp single
        op_t_start = omp_get_wtime();

    u_int8_t row_min[img->cols];

    #pragma omp for schedule(static)
        for (int i = se_center_row; i < img->rows-se_center_row; i++) {
            #pragma omp simd if (vectorization != 0)
            for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                row_min[j] = 255;
            }

            for (int m = 0; m < se_rows; m++) {
                int x = i + m - se_center_row;
                for (int n = 0; n < se_cols; n++) {
                    int mask = -(int)structuring_element->data[m][n];
                    #pragma omp simd if (vectorization != 0)
                    for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                        int y = j + n - se_center_col;
                        int masked = (img->data[x][y] & mask) | (255 & ~mask);
                        row_min[j] = masked < row_min[j] ? masked : row_min[j];
                    }
                }
            }

            #pragma omp simd if (vectorization != 0)
            for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                scratch->data[i][j] = row_min[j];
            }
        }

    #pragma omp master
    {
        op_t_end = omp_get_wtime();
        last_op_seconds = op_t_end - op_t_start;
    }
}

void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int vectorization, int size) {
    int se_center_row = structuring_element->rows / 2;
    int se_center_col = structuring_element->cols / 2;
    int center = se_center_row > se_center_col ? se_center_row : se_center_col;

    for (int i = 0; i < size; i++) {
        pad_image(img[i], scratch, center, 255);

        #pragma omp single
            allocate_matrix(scratch, img[i]->rows, img[i]->cols);

        erosion_helper(img[i], structuring_element, scratch, vectorization);

        #pragma omp single
            move_matrix(img[i], scratch);

        crop_image(img[i], scratch, center);
    }
}

static void dilation_helper (matrix* img, matrix* structuring_element, matrix* scratch, int vectorization) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;

    #pragma omp single
        op_t_start = omp_get_wtime();

    u_int8_t row_max[img->cols];

    #pragma omp for schedule(static)
        for (int i = se_center_row; i < img->rows-se_center_row; i++) {
            #pragma omp simd if (vectorization != 0)
            for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                row_max[j] = 0;
            }

            for (int m = 0; m < se_rows; m++) {
                int x = i + m - se_center_row;
                for (int n = 0; n < se_cols; n++) {
                    int mask = -(int)structuring_element->data[m][n];
                    #pragma omp simd if (vectorization != 0)
                    for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                        int y = j + n - se_center_col;
                        int masked = (img->data[x][y] & mask) | (0 & ~mask);
                        row_max[j] = masked > row_max[j] ? masked : row_max[j];
                    }
                }
            }

            #pragma omp simd if (vectorization != 0)
            for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                scratch->data[i][j] = row_max[j];
            }
        }

    #pragma omp master
    {
        op_t_end = omp_get_wtime();
        last_op_seconds = op_t_end - op_t_start;
    }
}

void image_dilation(matrix** img, matrix* structuring_element, matrix* scratch, int vectorization, int size) {
    int se_center_row = structuring_element->rows / 2;
    int se_center_col = structuring_element->cols / 2;
    int center = se_center_row > se_center_col ? se_center_row : se_center_col;

    for(int i = 0; i < size; i++) {
        pad_image(img[i], scratch, center, 0);

        #pragma omp single
            allocate_matrix(scratch, img[i]->rows, img[i]->cols);

        dilation_helper(img[i], structuring_element, scratch, vectorization);

        #pragma omp single
            move_matrix(img[i], scratch);

        crop_image(img[i], scratch, center);
    }
}


