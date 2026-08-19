// Baseline sequenziale in C puro: nessuna direttiva OpenMP, nemmeno omp simd.
// Non include <omp.h> di proposito, cosi' il file non puo' accidentalmente
// dipendere dal runtime OpenMP; il tempo si misura con clock_gettime.
//
// Questo file viene compilato DUE volte dal Makefile, con SEQ_SUFFIX diverso:
//   -DSEQ_SUFFIX=_simd                                   -> auto-vettorizzato
//   -DSEQ_SUFFIX=_scalar -fno-vectorize -fno-slp-vectorize -> scalare
// I due oggetti convivono nello stesso binario perche' ogni simbolo pubblico
// porta il suffisso. Vedi sequential.h per il perche' servono entrambi.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sequential.h"

#ifndef SEQ_SUFFIX
#error "sequential.c va compilato con -DSEQ_SUFFIX=_scalar oppure -DSEQ_SUFFIX=_simd"
#endif

#define SEQ_CAT_(a, b) a##b
#define SEQ_CAT(a, b)  SEQ_CAT_(a, b)
#define SEQ(name)      SEQ_CAT(name, SEQ_SUFFIX)

double SEQ(last_seq_seconds) = 0.0;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Espande img di padding_size per lato riempiendo il bordo con fill_value.
// Equivalente sequenziale di pad_image (che usa single/for).
static void seq_pad(matrix* img, int padding_size, u_int8_t fill_value) {
    int new_rows = img->rows + 2 * padding_size;
    int new_cols = img->cols + 2 * padding_size;

    matrix padded = {0};
    allocate_matrix(&padded, new_rows, new_cols);
    memset(padded.data[0], fill_value, (size_t)new_rows * new_cols);

    for (int i = 0; i < img->rows; i++) {
        memcpy(&padded.data[i + padding_size][padding_size], img->data[i], (size_t)img->cols);
    }

    move_matrix_data(img, &padded);
}

// Ritaglia crop_size per lato. Equivalente sequenziale di crop_image.
static void seq_crop(matrix* img, int crop_size) {
    int new_rows = img->rows - 2 * crop_size;
    int new_cols = img->cols - 2 * crop_size;

    matrix cropped = {0};
    allocate_matrix(&cropped, new_rows, new_cols);

    for (int i = 0; i < new_rows; i++) {
        memcpy(cropped.data[i], &img->data[i + crop_size][crop_size], (size_t)new_cols);
    }

    move_matrix_data(img, &cropped);
}

// Loop centrale dell'erosione su un'immagine gia' paddata; scrive in dst.
// Stesso trucco della maschera usato dalla versione parallela: se l'elemento
// strutturante vale 0 il pixel contribuisce con 255, che e' l'identita' del min.
static void erosion_kernel(matrix* img, matrix* se, matrix* dst) {
    int se_rows = se->rows, se_cols = se->cols;
    int cr = se_rows / 2, cc = se_cols / 2;

    u_int8_t row_min[img->cols];

    for (int i = cr; i < img->rows - cr; i++) {
        for (int j = cc; j < img->cols - cc; j++) row_min[j] = 255;

        for (int m = 0; m < se_rows; m++) {
            int x = i + m - cr;
            for (int n = 0; n < se_cols; n++) {
                int mask = -(int)se->data[m][n];
                for (int j = cc; j < img->cols - cc; j++) {
                    int y = j + n - cc;
                    int masked = (img->data[x][y] & mask) | (255 & ~mask);
                    row_min[j] = masked < row_min[j] ? masked : row_min[j];
                }
            }
        }

        for (int j = cc; j < img->cols - cc; j++) dst->data[i][j] = row_min[j];
    }
}

// Idem per la dilatazione: identita' del max = 0.
static void dilation_kernel(matrix* img, matrix* se, matrix* dst) {
    int se_rows = se->rows, se_cols = se->cols;
    int cr = se_rows / 2, cc = se_cols / 2;

    u_int8_t row_max[img->cols];

    for (int i = cr; i < img->rows - cr; i++) {
        for (int j = cc; j < img->cols - cc; j++) row_max[j] = 0;

        for (int m = 0; m < se_rows; m++) {
            int x = i + m - cr;
            for (int n = 0; n < se_cols; n++) {
                int mask = -(int)se->data[m][n];
                for (int j = cc; j < img->cols - cc; j++) {
                    int y = j + n - cc;
                    int masked = (img->data[x][y] & mask) | (0 & ~mask);
                    row_max[j] = masked > row_max[j] ? masked : row_max[j];
                }
            }
        }

        for (int j = cc; j < img->cols - cc; j++) dst->data[i][j] = row_max[j];
    }
}

static int se_center(matrix* se) {
    int cr = se->rows / 2, cc = se->cols / 2;
    return cr > cc ? cr : cc;
}

// Un singolo stadio completo: pad -> kernel -> crop.
static void run_stage(matrix* img, matrix* se, u_int8_t fill,
                      void (*kernel)(matrix*, matrix*, matrix*)) {
    int c = se_center(se);

    seq_pad(img, c, fill);

    matrix out = {0};
    allocate_matrix(&out, img->rows, img->cols);

    kernel(img, se, &out);

    move_matrix_data(img, &out);
    seq_crop(img, c);
}

// Tutte e quattro le operazioni cronometrano l'intero batch, padding e cropping
// inclusi, dividendo per il numero di immagini: stessa convenzione della
// versione parallela, quindi i tempi sono direttamente confrontabili.
void SEQ(seq_erosion)(matrix** img, matrix* structuring_element, int size) {
    double t0 = now_seconds();
    for (int k = 0; k < size; k++)
        run_stage(img[k], structuring_element, 255, erosion_kernel);
    SEQ(last_seq_seconds) = (now_seconds() - t0) / size;
}

void SEQ(seq_dilation)(matrix** img, matrix* structuring_element, int size) {
    double t0 = now_seconds();
    for (int k = 0; k < size; k++)
        run_stage(img[k], structuring_element, 0, dilation_kernel);
    SEQ(last_seq_seconds) = (now_seconds() - t0) / size;
}

void SEQ(seq_opening)(matrix** img, matrix* structuring_element, int size) {
    double t0 = now_seconds();
    for (int k = 0; k < size; k++) {
        run_stage(img[k], structuring_element, 255, erosion_kernel);
        run_stage(img[k], structuring_element, 0,   dilation_kernel);
    }
    SEQ(last_seq_seconds) = (now_seconds() - t0) / size;
}
