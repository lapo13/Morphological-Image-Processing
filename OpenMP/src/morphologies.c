#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "morphologies.h"
#include "image.h"

double last_op_seconds = 0.0;

// Tutte le operazioni misurano la stessa cosa: il wall-clock dell'intero batch
// (padding e cropping inclusi) diviso per il numero di immagini, cioe' un tempo
// per-immagine. Cosi' i tempi delle quattro operazioni sono direttamente
// confrontabili fra loro, e ogni misura dura abbastanza da non essere dominata
// dal rumore di scheduling.
static double op_t_start, op_t_end;

// Variabili DISTINTE per le operazioni composte: image_erosion/image_dilation
// sovrascrivono op_t_start quando vengono chiamate dal percorso sequenziale.
static double comp_t_start, comp_t_end;

// ---------------------------------------------------------------------------
// Percorso team-wide: erosione e dilatazione classiche.
// Tutti i thread del team attraversano insieme le stesse #pragma omp for/single,
// sfruttandone le barriere implicite per sincronizzare le fasi.
// ---------------------------------------------------------------------------

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

static void erosion_rows(matrix* img, matrix* structuring_element, matrix* dst,
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

static void dilation_rows(matrix* img, matrix* structuring_element, matrix* dst,
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

// ---------------------------------------------------------------------------
// Pipeline software a due stadi.
//
// Al passo s meta' team applica lo stadio 1 a img[s] mentre l'altra meta'
// applica lo stadio 2 a img[s-1] (gia' passato per lo stadio 1 al passo
// precedente): i due stadi si sovrappongono nel tempo invece di eseguirsi
// uno dopo l'altro sull'intero batch.
//
// Solo la fase di CALCOLO e' divisa fra le due meta'. pad_image/crop_image
// restano in contesto team completo, dove i loro single/for sono leciti.
// ---------------------------------------------------------------------------

typedef void (*stage_fn)(matrix*, matrix*, matrix*, int, int);

static void run_pipeline(matrix** img, matrix* structuring_element, matrix* scratch_a,
                         int size,
                         stage_fn stage1, u_int8_t fill1,
                         stage_fn stage2, u_int8_t fill2) {
    int se_center_row = structuring_element->rows / 2;
    int se_center_col = structuring_element->cols / 2;
    int c = se_center_row > se_center_col ? se_center_row : se_center_col;

    int tid      = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int half = nthreads / 2;                        // [0, half)        -> stadio 1
    int rank_a = tid,        na = half;             // [half, nthreads) -> stadio 2
    int rank_b = tid - half, nb = nthreads - half;

    // Buffer separati per stadio: out_a/out_b coesistono durante il calcolo,
    // scratch_a/scratch_b restano distinti per non legare la correttezza al
    // fatto che pad/crop stiano fuori dallo split.
    static matrix scratch_b, out_a, out_b;

    #pragma omp single
    {
        scratch_b.data = NULL; scratch_b.rows = scratch_b.cols = 0;
        out_a.data     = NULL; out_a.rows     = out_a.cols     = 0;
        out_b.data     = NULL; out_b.rows     = out_b.cols     = 0;
        comp_t_start = omp_get_wtime();
    }

    for (int s = 0; s <= size; s++) {
        // --- team completo: padding ---
        if (s < size) pad_image(img[s],   scratch_a,  c, fill1);
        if (s > 0)    pad_image(img[s-1], &scratch_b, c, fill2);

        #pragma omp single
        {
            if (s < size) allocate_matrix(&out_a, img[s]->rows,   img[s]->cols);
            if (s > 0)    allocate_matrix(&out_b, img[s-1]->rows, img[s-1]->cols);
        }

        // --- split: due immagini calcolate in parallelo ---
        if (tid <  half && s < size) stage1(img[s],   structuring_element, &out_a, rank_a, na);
        if (tid >= half && s > 0)    stage2(img[s-1], structuring_element, &out_b, rank_b, nb);

        #pragma omp barrier

        // --- team completo: crop ---
        if (s < size) {
            #pragma omp single
                move_matrix_data(img[s], &out_a);
            crop_image(img[s], scratch_a, c);
        }
        if (s > 0) {
            #pragma omp single
                move_matrix_data(img[s-1], &out_b);
            crop_image(img[s-1], &scratch_b, c);
        }
    }

    #pragma omp single
    {
        comp_t_end = omp_get_wtime();
        // tempo per-immagine ammortizzato, confrontabile con erosion/dilation
        last_op_seconds = (comp_t_end - comp_t_start) / size;
        free_matrix(&scratch_b);
        free_matrix(&out_a);
        free_matrix(&out_b);
    }
}

// Con un solo thread non c'e' nulla da sovrapporre: composizione sequenziale.
static void run_composed(matrix** img, matrix* structuring_element, matrix* scratch,
                         int size, int erosion_first) {
    #pragma omp single
        comp_t_start = omp_get_wtime();

    if (erosion_first) {
        image_erosion(img, structuring_element, scratch, size);
        image_dilation(img, structuring_element, scratch, size);
    } else {
        image_dilation(img, structuring_element, scratch, size);
        image_erosion(img, structuring_element, scratch, size);
    }

    #pragma omp single
    {
        comp_t_end = omp_get_wtime();
        last_op_seconds = (comp_t_end - comp_t_start) / size;
    }
}

void image_opening(matrix** img, matrix* structuring_element, matrix* scratch, int size) {
    if (omp_get_num_threads() < 2) {
        run_composed(img, structuring_element, scratch, size, 1);
        return;
    }
    run_pipeline(img, structuring_element, scratch, size,
                 erosion_rows, 255, dilation_rows, 0);
}

void image_closing(matrix** img, matrix* structuring_element, matrix* scratch, int size) {
    if (omp_get_num_threads() < 2) {
        run_composed(img, structuring_element, scratch, size, 0);
        return;
    }
    run_pipeline(img, structuring_element, scratch, size,
                 dilation_rows, 0, erosion_rows, 255);
}
