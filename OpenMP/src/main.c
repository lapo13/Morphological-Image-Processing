#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#include "matrix.h"
#include "image.h"
#include "morphologies.h"
#include "logger.h"

#define SE_SIZE 5
#define DATASET_DIR "../brain-cancer-mri-dataset/Brain_Cancer raw MRI data/Brain_Cancer/"
#define WARMUP_RUNS 5
#define TIMED_RUNS 40
#define MAX_MOSAIC_TILES 64

static const char* MOSAIC_TILES[MAX_MOSAIC_TILES] = {
    DATASET_DIR "brain_menin/brain_menin_0001.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0001.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0001.jpg",
    DATASET_DIR "brain_menin/brain_menin_0002.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0002.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0002.jpg",
    DATASET_DIR "brain_menin/brain_menin_0003.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0003.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0003.jpg",
    DATASET_DIR "brain_menin/brain_menin_0004.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0004.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0004.jpg",
    DATASET_DIR "brain_menin/brain_menin_0005.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0005.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0005.jpg",
    DATASET_DIR "brain_menin/brain_menin_0006.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0006.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0006.jpg",
    DATASET_DIR "brain_menin/brain_menin_0007.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0007.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0007.jpg",
    DATASET_DIR "brain_menin/brain_menin_0008.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0008.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0008.jpg",
    DATASET_DIR "brain_menin/brain_menin_0009.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0009.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0009.jpg",
    DATASET_DIR "brain_menin/brain_menin_0010.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0010.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0010.jpg",
    DATASET_DIR "brain_menin/brain_menin_0011.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0011.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0011.jpg",
    DATASET_DIR "brain_menin/brain_menin_0012.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0012.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0012.jpg",
    DATASET_DIR "brain_menin/brain_menin_0013.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0013.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0013.jpg",
    DATASET_DIR "brain_menin/brain_menin_0014.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0014.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0014.jpg",
    DATASET_DIR "brain_menin/brain_menin_0015.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0015.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0015.jpg",
    DATASET_DIR "brain_menin/brain_menin_0016.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0016.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0016.jpg",
    DATASET_DIR "brain_menin/brain_menin_0017.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0017.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0017.jpg",
    DATASET_DIR "brain_menin/brain_menin_0018.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0018.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0018.jpg",
    DATASET_DIR "brain_menin/brain_menin_0019.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0019.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0019.jpg",
    DATASET_DIR "brain_menin/brain_menin_0020.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0020.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0020.jpg",
    DATASET_DIR "brain_menin/brain_menin_0021.jpg",
    DATASET_DIR "brain_tumor/brain_tumor_0021.jpg",
    DATASET_DIR "brain_glioma/brain_glioma_0021.jpg",
    DATASET_DIR "brain_menin/brain_menin_0022.jpg"
};

typedef struct {
    int rows;
    int cols;
} grid_size;

static const grid_size TEST_GRID_SIZES[] = {
    {1, 1},
    {2, 2},
    {4, 4},
    {8, 8}
};
#define NUM_TEST_SIZES (int)(sizeof(TEST_GRID_SIZES) / sizeof(TEST_GRID_SIZES[0]))

static const int TEST_THREAD_COUNTS[] = {1, 2, 4, 6, 8, 10};
#define NUM_THREAD_COUNTS (int)(sizeof(TEST_THREAD_COUNTS) / sizeof(TEST_THREAD_COUNTS[0]))

#define MAX_SAMPLES (4 * TIMED_RUNS)
static timing_sample run_samples[MAX_SAMPLES];
static int sample_count = 0;

typedef void (*morph_op)(matrix*, matrix*, matrix*, int);

static void benchmark_operation(morph_op op, const char* label, matrix* input, matrix* working, matrix* se, matrix* scratch, int vectorization) {
    for (int r = 0; r < WARMUP_RUNS; r++) {
        copy_matrix(input, working);
        op(working, se, scratch, vectorization);
    }

    double sum = 0.0, min = DBL_MAX;
    for (int r = 0; r < TIMED_RUNS; r++) {
        copy_matrix(input, working);
        op(working, se, scratch, vectorization);
        #pragma omp master
        {
            double d = last_op_seconds;
            run_samples[sample_count++] = (timing_sample){label, r, d, vectorization};
            sum += d;
            if (d < min) min = d;
        }
    }

    #pragma omp master
    {
        printf("%s (vectorized=%d): mean=%.4f s, min=%.4f s (%d warmup + %d timed runs)\n", label, vectorization, sum / TIMED_RUNS, min, WARMUP_RUNS, TIMED_RUNS);
    }
}


#define VEC_TEST_GRID_ROWS 2
#define VEC_TEST_GRID_COLS 2

static void run_vectorization_test(const char* run_id) {
    matrix se;
    matrix input_image, tile_buffer;
    matrix eroded_image, dilated_image, opened_image, closed_image;
    matrix scratch;

    se.rows = SE_SIZE;
    se.cols = SE_SIZE;

    #pragma omp parallel num_threads(1)
    {
        #pragma omp single
        {
            printf("\n=== Vectorization comparison (1 thread) ===\n");
        }

        allocate_matrix(&se, se.rows, se.cols);
        #pragma omp for
        for (int i = 0; i < se.rows; i++) {
            for (int j = 0; j < se.cols; j++) {
                se.data[i][j] = 1;
            }
        }

        build_mosaic_image(&input_image, &tile_buffer, MOSAIC_TILES, VEC_TEST_GRID_ROWS, VEC_TEST_GRID_COLS);

        #pragma omp single
        {
            printf("Input Image (%dx%d tiles): %d x %d\n", VEC_TEST_GRID_ROWS, VEC_TEST_GRID_COLS, input_image.rows, input_image.cols);
        }

        for (int v = 1; v >= 0; v--) {
            #pragma omp single
            sample_count = 0;

            benchmark_operation(image_erosion, "erosion", &input_image, &eroded_image, &se, &scratch, v);
            benchmark_operation(image_dilation, "dilation", &input_image, &dilated_image, &se, &scratch, v);
            benchmark_operation(image_opening, "opening", &input_image, &opened_image, &se, &scratch, v);
            benchmark_operation(image_closing, "closing", &input_image, &closed_image, &se, &scratch, v);

            #pragma omp single
            log_timings_csv(run_id, run_samples, sample_count, 1, input_image.rows, input_image.cols);
        }

        #pragma omp master
        {
            free_matrix(&se);
            free_matrix(&input_image);
            free_matrix(&eroded_image);
            free_matrix(&dilated_image);
            free_matrix(&opened_image);
            free_matrix(&closed_image);
        }
    }
}

int main() {
    char run_id[32];
    time_t now = time(NULL);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    strftime(run_id, sizeof(run_id), "%Y-%m-%d %H:%M:%S", &tm_info);

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

    for (int t = 0; t < NUM_THREAD_COUNTS; t++) {
        int num_threads = TEST_THREAD_COUNTS[t];

        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp single
            {
                printf("\n=== Running with %d threads ===\n", num_threads);
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

            for (int s = 0; s < NUM_TEST_SIZES; s++) {
                int grid_rows = TEST_GRID_SIZES[s].rows;
                int grid_cols = TEST_GRID_SIZES[s].cols;

                build_mosaic_image(&input_image, &tile_buffer, MOSAIC_TILES, grid_rows, grid_cols);

                #pragma omp single
                {
                    printf("Input Image (%dx%d tiles): %d x %d\n", grid_rows, grid_cols, input_image.rows, input_image.cols);
                }

                for (int v = 1; v >= 0; v--) {
                    #pragma omp single
                    sample_count = 0;

                    benchmark_operation(image_erosion, "erosion", &input_image, &eroded_image, &structuring_element, &scratch, v);
                    benchmark_operation(image_dilation, "dilation", &input_image, &dilated_image, &structuring_element, &scratch, v);
                    benchmark_operation(image_opening, "opening", &input_image, &opened_image, &structuring_element, &scratch, v);
                    benchmark_operation(image_closing, "closing", &input_image, &closed_image, &structuring_element, &scratch, v);

                    #pragma omp single
                    log_timings_csv(run_id, run_samples, sample_count, num_threads, input_image.rows, input_image.cols);
                }

                #pragma omp master
                {
                    free_matrix(&input_image);
                    free_matrix(&eroded_image);
                    free_matrix(&dilated_image);
                    free_matrix(&opened_image);
                    free_matrix(&closed_image);
                }
            }

            #pragma omp master
            {
                printf("Freeing allocated memory...\n");
                free_matrix(&structuring_element);
            }
        }
    }

    run_vectorization_test(run_id);

    return 0;
}
