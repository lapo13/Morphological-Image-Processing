#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>

#include "matrix.h"
#include "image.h"
#include "morphologies.h"
#include "logger.h"

#define N 4 // Number of threads for OpenMP parallelization
#define SE_SIZE 5
#define GRID_ROWS 4
#define GRID_COLS 3
#define DATASET_DIR "../brain-cancer-mri-dataset/Brain_Cancer raw MRI data/Brain_Cancer/"
#define WARMUP_RUNS 10
#define TIMED_RUNS 40

static const char* MOSAIC_TILES[GRID_ROWS * GRID_COLS] = {
    DATASET_DIR "brain_menin/brain_menin_0001.jpg",
    DATASET_DIR "brain_menin/brain_menin_0002.jpg",
    DATASET_DIR "brain_menin/brain_menin_0003.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0001.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0002.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0003.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0001.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0002.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0003.jpg",
    DATASET_DIR "brain_menin/brain_menin_0004.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0004.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0004.jpg",
};

double t_start, t_end;
timing_entry run_timings[4];

typedef void (*morph_op)(matrix*, matrix*, matrix*);

static void benchmark_operation(morph_op op, const char* label, matrix* input, matrix* working, matrix* se, matrix* scratch, timing_entry* out) {
    for (int r = 0; r < WARMUP_RUNS; r++) {
        copy_matrix(input, working);
        op(working, se, scratch);
    }

    double sum = 0.0, min = DBL_MAX;
    for (int r = 0; r < TIMED_RUNS; r++) {
        copy_matrix(input, working);
        #pragma omp single
            t_start = omp_get_wtime();
        op(working, se, scratch);
        #pragma omp master
        {
            t_end = omp_get_wtime();
            double d = t_end - t_start;
            sum += d;
            if (d < min) min = d;
        }
    }

    #pragma omp master
    {
        *out = (timing_entry){label, sum / TIMED_RUNS, min};
        printf("%s: mean=%.4f s, min=%.4f s (%d warmup + %d timed runs)\n", label, out->mean_seconds, out->min_seconds, WARMUP_RUNS, TIMED_RUNS);
    }
}

int main() {
    matrix input_image;
    matrix structuring_element;

    structuring_element.rows = SE_SIZE;
    structuring_element.cols = SE_SIZE;

    matrix eroded_image;
    matrix dilated_image;
    matrix opened_image;
    matrix closed_image;

    matrix scratch;
    matrix tile_buffer;

    #pragma omp parallel num_threads(N)
    {
        build_mosaic_image(&input_image, &tile_buffer, MOSAIC_TILES, GRID_ROWS, GRID_COLS);

        #pragma omp single
        {
            printf("Number of threads: %d\n", omp_get_num_threads());
            printf("Input Image: %d x %d\n", input_image.rows, input_image.cols);
        }

        allocate_matrix(&structuring_element, structuring_element.rows, structuring_element.cols);
        #pragma omp for
        for (int i = 0; i < structuring_element.rows; i++) {
            for (int j = 0; j < structuring_element.cols; j++) {
                structuring_element.data[i][j] = 1;
            }
        }

        #pragma omp single
        {
        printf("Structuring Element:\n");
        }

        print_matrix(&structuring_element);

        benchmark_operation(image_erosion, "erosion", &input_image, &eroded_image, &structuring_element, &scratch, &run_timings[0]);
        benchmark_operation(image_dilation, "dilation", &input_image, &dilated_image, &structuring_element, &scratch, &run_timings[1]);
        benchmark_operation(image_opening, "opening", &input_image, &opened_image, &structuring_element, &scratch, &run_timings[2]);
        benchmark_operation(image_closing, "closing", &input_image, &closed_image, &structuring_element, &scratch, &run_timings[3]);

        #pragma omp master
        {
        log_timings_csv(run_timings, 4);
        }

        #pragma omp master
        {
        printf("Freeing allocated memory...\n");
        }

        free_matrix(&input_image);
        free_matrix(&structuring_element);

        free_matrix(&eroded_image);
        free_matrix(&dilated_image);
        free_matrix(&opened_image);
        free_matrix(&closed_image);
    }
    return 0;
}
