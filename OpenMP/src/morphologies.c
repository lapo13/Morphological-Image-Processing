#include <stdio.h>
#include <stdlib.h>
#include "morphologies.h"
#include "image.h"

void image_erosion(matrix* img, matrix* structuring_element, matrix* scratch) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;
    int center = se_center_row;

    pad_image(img, scratch, center, 255);

    allocate_matrix(scratch, img->rows, img->cols);

    #pragma omp for collapse(2) schedule(dynamic)
    for (int i = center; i < img->rows-center; i++) {
        for (int j = center; j < img->cols-center; j++) {
            int min_value = 255;

            for (int m = 0; m < se_rows; m++) {
                for (int n = 0; n < se_cols; n++) {
                    if (structuring_element->data[m][n] == 1) {
                        int x = i + m - se_center_row;
                        int y = j + n - se_center_col;
                        img->data[x][y] < min_value ? min_value = img->data[x][y] : min_value;
                    }
                }
            }
            scratch->data[i][j] = min_value;
        }
     }

    free_matrix(img);
    #pragma omp single
    img->data = scratch->data;

    crop_image(img, scratch, center);
}

void image_dilation(matrix* img, matrix* structuring_element, matrix* scratch) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;
    int center = se_center_row;

    pad_image(img, scratch, center, 0);

    allocate_matrix(scratch, img->rows, img->cols);

    #pragma omp for collapse(2) schedule(dynamic)
    for (int i = center; i < img->rows-center; i++) {
        for (int j = center; j < img->cols-center; j++) {
            int max_value = 0;

            for (int m = 0; m < se_rows; m++) {
                for (int n = 0; n < se_cols; n++) {
                    if (structuring_element->data[m][n] == 1) {
                        int x = i + m - se_center_row;
                        int y = j + n - se_center_col;
                        img->data[x][y] > max_value ? max_value = img->data[x][y] : max_value;
                    }
                }
            }
            scratch->data[i][j] = max_value;
        }
    }

    free_matrix(img);
    #pragma omp single
    img->data = scratch->data;

    crop_image(img, scratch, center);
}

void image_opening(matrix* img, matrix* structuring_element, matrix* scratch) {
    image_erosion(img, structuring_element, scratch);
    #pragma omp flush
    image_dilation(img, structuring_element, scratch);
}

void image_closing(matrix* img, matrix* structuring_element, matrix* scratch) {
    image_dilation(img, structuring_element, scratch);
    #pragma omp flush
    image_erosion(img, structuring_element, scratch);
}
