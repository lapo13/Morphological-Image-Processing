#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "morphologies.h"
#include "image.h"

double last_op_seconds = 0.0;

static double op_t_start, op_t_end;

static double comp_t_start, comp_t_end;


void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int size) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;
    int center = se_center_row > se_center_col ? se_center_row : se_center_col;

    #pragma omp single
        op_t_start = omp_get_wtime();

    for (int k = 0; k < size; k++) {
        pad_image(img[k], scratch, center, 255);

        #pragma omp single
            allocate_matrix(scratch, img[k]->rows, img[k]->cols);

        u_int8_t row_min[img[k]->cols];

        // guided: scelto sperimentalmente fra static/dynamic/guided e diverse
        // chunk size. Su CPU a core eterogenei (P-core + E-core) static
        // assegna a tutti lo stesso numero di righe e il team finisce per
        // aspettare gli E-core; guided riduce lo sbilanciamento assegnando
        // blocchi via via piu' piccoli. Vedi la tabella nel README.
        #pragma omp for schedule(guided)
            for (int i = se_center_row; i < img[k]->rows-se_center_row; i++) {
                #pragma omp simd
                for (int j = se_center_col; j < img[k]->cols-se_center_col; j++) {
                    row_min[j] = 255;
                }

                for (int m = 0; m < se_rows; m++) {
                    int x = i + m - se_center_row;
                    for (int n = 0; n < se_cols; n++) {
                        int mask = -(int)structuring_element->data[m][n];
                        #pragma omp simd
                        for (int j = se_center_col; j < img[k]->cols-se_center_col; j++) {
                            int y = j + n - se_center_col;
                            int masked = (img[k]->data[x][y] & mask) | (255 & ~mask);
                            row_min[j] = masked < row_min[j] ? masked : row_min[j];
                        }
                    }
                }

                #pragma omp simd
                for (int j = se_center_col; j < img[k]->cols-se_center_col; j++) {
                    scratch->data[i][j] = row_min[j];
                }
            }

        #pragma omp single
            move_matrix_data(img[k], scratch);

        crop_image(img[k], scratch, center);
    }

    #pragma omp single
    {
        op_t_end = omp_get_wtime();
        last_op_seconds = (op_t_end - op_t_start) / size;
    }
}

void image_dilation(matrix** img, matrix* structuring_element, matrix* scratch, int size) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;
    int center = se_center_row > se_center_col ? se_center_row : se_center_col;

    #pragma omp single
        op_t_start = omp_get_wtime();

    for (int k = 0; k < size; k++) {
        pad_image(img[k], scratch, center, 0);

        #pragma omp single
            allocate_matrix(scratch, img[k]->rows, img[k]->cols);

        u_int8_t row_max[img[k]->cols];

        // guided: scelto sperimentalmente fra static/dynamic/guided e diverse
        // chunk size. Su CPU a core eterogenei (P-core + E-core) static
        // assegna a tutti lo stesso numero di righe e il team finisce per
        // aspettare gli E-core; guided riduce lo sbilanciamento assegnando
        // blocchi via via piu' piccoli. Vedi la tabella nel README.
        #pragma omp for schedule(guided)
            for (int i = se_center_row; i < img[k]->rows-se_center_row; i++) {
                #pragma omp simd
                for (int j = se_center_col; j < img[k]->cols-se_center_col; j++) {
                    row_max[j] = 0;
                }

                for (int m = 0; m < se_rows; m++) {
                    int x = i + m - se_center_row;
                    for (int n = 0; n < se_cols; n++) {
                        int mask = -(int)structuring_element->data[m][n];
                        #pragma omp simd
                        for (int j = se_center_col; j < img[k]->cols-se_center_col; j++) {
                            int y = j + n - se_center_col;
                            int masked = (img[k]->data[x][y] & mask) | (0 & ~mask);
                            row_max[j] = masked > row_max[j] ? masked : row_max[j];
                        }
                    }
                }

                #pragma omp simd
                for (int j = se_center_col; j < img[k]->cols-se_center_col; j++) {
                    scratch->data[i][j] = row_max[j];
                }
            }

        #pragma omp single
            move_matrix_data(img[k], scratch);

        crop_image(img[k], scratch, center);
    }

    #pragma omp single
    {
        op_t_end = omp_get_wtime();
        last_op_seconds = (op_t_end - op_t_start) / size;
    }
}


static void row_range(int total, int rank, int nthreads, int* start, int* end) {
    int chunk = total / nthreads;
    int rem   = total % nthreads;
    *start = rank * chunk + (rank < rem ? rank : rem);
    *end   = *start + chunk + (rank < rem ? 1 : 0);
}

static void erosion_helper(matrix* img, matrix* structuring_element, matrix* dst,
                         int rank, int nthreads) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;

    int first = se_center_row;
    int last  = img->rows - se_center_row;
    int start, end;
    row_range(last - first, rank, nthreads, &start, &end);
    start += first;
    end   += first;

    u_int8_t row_min[img->cols];

    for (int i = start; i < end; i++) {
        #pragma omp simd
        for (int j = se_center_col; j < img->cols-se_center_col; j++) {
            row_min[j] = 255;
        }

        for (int m = 0; m < se_rows; m++) {
            int x = i + m - se_center_row;
            for (int n = 0; n < se_cols; n++) {
                int mask = -(int)structuring_element->data[m][n];
                #pragma omp simd
                for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                    int y = j + n - se_center_col;
                    int masked = (img->data[x][y] & mask) | (255 & ~mask);
                    row_min[j] = masked < row_min[j] ? masked : row_min[j];
                }
            }
        }

        #pragma omp simd
        for (int j = se_center_col; j < img->cols-se_center_col; j++) {
            dst->data[i][j] = row_min[j];
        }
    }
}

static void dilation_helper(matrix* img, matrix* structuring_element, matrix* dst,
                          int rank, int nthreads) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;
    int se_center_row = se_rows / 2;
    int se_center_col = se_cols / 2;

    int first = se_center_row;
    int last  = img->rows - se_center_row;
    int start, end;
    row_range(last - first, rank, nthreads, &start, &end);
    start += first;
    end   += first;

    u_int8_t row_max[img->cols];

    for (int i = start; i < end; i++) {
        #pragma omp simd
        for (int j = se_center_col; j < img->cols-se_center_col; j++) {
            row_max[j] = 0;
        }

        for (int m = 0; m < se_rows; m++) {
            int x = i + m - se_center_row;
            for (int n = 0; n < se_cols; n++) {
                int mask = -(int)structuring_element->data[m][n];
                #pragma omp simd
                for (int j = se_center_col; j < img->cols-se_center_col; j++) {
                    int y = j + n - se_center_col;
                    int masked = (img->data[x][y] & mask) | (0 & ~mask);
                    row_max[j] = masked > row_max[j] ? masked : row_max[j];
                }
            }
        }

        #pragma omp simd
        for (int j = se_center_col; j < img->cols-se_center_col; j++) {
            dst->data[i][j] = row_max[j];
        }
    }
}

void image_opening(matrix** img, matrix* structuring_element, matrix* scratch_a, int size) {
    int se_center_row = structuring_element->rows / 2;
    int se_center_col = structuring_element->cols / 2;
    int center = se_center_row > se_center_col ? se_center_row : se_center_col;

    int tid      = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    if (nthreads < 2) {
        #pragma omp single
            comp_t_start = omp_get_wtime();

        image_erosion(img, structuring_element, scratch_a, size);
        image_dilation(img, structuring_element, scratch_a, size);

        #pragma omp single
        {
            comp_t_end = omp_get_wtime();
            last_op_seconds = (comp_t_end - comp_t_start) / size;
        }
        return;
    }

    int half = nthreads / 2;                        // [0, half)        -> erosione
    int rank_a = tid,        na = half;             // [half, nthreads) -> dilatazione
    int rank_b = tid - half, nb = nthreads - half;

    static matrix scratch_b, out_a, out_b;

    #pragma omp single
    {
        scratch_b.data = NULL; scratch_b.rows = scratch_b.cols = 0;
        out_a.data     = NULL; out_a.rows     = out_a.cols     = 0;
        out_b.data     = NULL; out_b.rows     = out_b.cols     = 0;
        comp_t_start = omp_get_wtime();
    }

    for (int s = 0; s <= size; s++) {
        if (s < size) pad_image(img[s],   scratch_a,  center, 255);
        if (s > 0)    pad_image(img[s-1], &scratch_b, center, 0);

        #pragma omp single
        {
            if (s < size) allocate_matrix(&out_a, img[s]->rows,   img[s]->cols);
            if (s > 0)    allocate_matrix(&out_b, img[s-1]->rows, img[s-1]->cols);
        }

        if (tid <  half && s < size) erosion_helper(img[s],    structuring_element, &out_a, rank_a, na);
        if (tid >= half && s > 0)    dilation_helper(img[s-1], structuring_element, &out_b, rank_b, nb);

        #pragma omp barrier

        if (s < size) {
            #pragma omp single
                move_matrix_data(img[s], &out_a);
            crop_image(img[s], scratch_a, center);
        }
        if (s > 0) {
            #pragma omp single
                move_matrix_data(img[s-1], &out_b);
            crop_image(img[s-1], &scratch_b, center);
        }
    }

    #pragma omp single
    {
        comp_t_end = omp_get_wtime();
        last_op_seconds = (comp_t_end - comp_t_start) / size;
        free_matrix(&scratch_b);
        free_matrix(&out_a);
        free_matrix(&out_b);
    }
}
