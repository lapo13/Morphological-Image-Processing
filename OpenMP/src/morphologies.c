#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "morphologies.h"
#include "image.h"

double last_op_seconds = 0.0;
static double op_t_start, op_t_end;

void image_erosion(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;
    int center = se_center_row;

    pad_image(img, scratch, center, 255);

    allocate_matrix(scratch, img->rows, img->cols);

    #pragma omp single
        op_t_start = omp_get_wtime();

    int row_min[img->cols];

    #pragma omp for schedule(static)
        for (int i = center; i < img->rows-center; i++) {
            #pragma omp simd if (vectorization != 0)
            for (int j = center; j < img->cols-center; j++) {
                row_min[j] = 255;
            }

            for (int m = 0; m < se_rows; m++) {
                int x = i + m - se_center_row;
                for (int n = 0; n < se_cols; n++) {
                    int mask = -(int)structuring_element->data[m][n];
                    #pragma omp simd if (vectorization != 0)
                    for (int j = center; j < img->cols-center; j++) {
                        int y = j + n - se_center_col;
                        int masked = (img->data[x][y] & mask) | (255 & ~mask);
                        row_min[j] = masked < row_min[j] ? masked : row_min[j];
                    }
                }
            }

            #pragma omp simd if (vectorization != 0)
            for (int j = center; j < img->cols-center; j++) {
                scratch->data[i][j] = row_min[j];
            }
        }

    #pragma omp master
    {
        op_t_end = omp_get_wtime();
        last_op_seconds = op_t_end - op_t_start;
    }

    #pragma omp single
    {
        free_matrix(img);
        img->data = scratch->data;
    }

    crop_image(img, scratch, center);
}

void image_dilation(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;
    int center = se_center_row;

    pad_image(img, scratch, center, 0);

    allocate_matrix(scratch, img->rows, img->cols);

    #pragma omp single
        op_t_start = omp_get_wtime();

    int row_max[img->cols];

    #pragma omp for schedule(static)
    for (int i = center; i < img->rows-center; i++) {
        #pragma omp simd if (vectorization != 0)
        for (int j = center; j < img->cols-center; j++) {
            row_max[j] = 0;
        }

        for (int m = 0; m < se_rows; m++) {
            int x = i + m - se_center_row;
            for (int n = 0; n < se_cols; n++) {
                int mask = -(int)structuring_element->data[m][n];
                #pragma omp simd if (vectorization != 0)
                for (int j = center; j < img->cols-center; j++) {
                    int y = j + n - se_center_col;
                    int masked = img->data[x][y] & mask;
                    row_max[j] = masked > row_max[j] ? masked : row_max[j];
                }
            }
        }

        #pragma omp simd if (vectorization != 0)
        for (int j = center; j < img->cols-center; j++) {
            scratch->data[i][j] = row_max[j];
        }
    }

    #pragma omp master
    {
        op_t_end = omp_get_wtime();
        last_op_seconds = op_t_end - op_t_start;
    }

    #pragma omp single
    {
        free_matrix(img);
        img->data = scratch->data;
    }

    crop_image(img, scratch, center);
}

void image_opening(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization) {
    image_erosion(img, structuring_element, scratch, vectorization);
    double erosion_seconds = last_op_seconds;
    image_dilation(img, structuring_element, scratch, vectorization);
    #pragma omp master
    last_op_seconds += erosion_seconds;
}

void image_closing(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization) {
    image_dilation(img, structuring_element, scratch, vectorization);
    double dilation_seconds = last_op_seconds;
    image_erosion(img, structuring_element, scratch, vectorization);
    #pragma omp master
    last_op_seconds += dilation_seconds;
}
