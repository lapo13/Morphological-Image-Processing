#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

#include "matrix.h"
#include "image.h"
#include "morphologies.h"
#include "reference.h"
#include "logger.h"

#define DATASET_DIR "../brain-cancer-mri-dataset/Brain_Cancer raw MRI data/Brain_Cancer"
#define WARMUP_RUNS 2
#define TIMED_RUNS 10

// Sweep della dimensione: larghezza fissa di 8 tile (4096 pixel) e altezza
// crescente per potenze di due. La GPU riceve sempre la griglia naturale.
#define PROBLEM_COLS  8
#define PROBLEM_BATCH 8
#define MAX_BATCH PROBLEM_BATCH

static const int TEST_GRID_ROWS[] = {1, 2, 4, 8, 16, 32, 64};
#define NUM_GRID_ROWS (int)(sizeof(TEST_GRID_ROWS) / sizeof(TEST_GRID_ROWS[0]))

// Validazione: mosaico piccolo e batch ridotto, perché il confronto è contro un
// riferimento single-thread su CPU.
#define VALIDATION_BATCH 2

// Il costo per pixel cresce col numero di celle (9, 25, 81): questo asse varia
// l'intensita' aritmetica a parita' di traffico di memoria.
static const int TEST_SE_SIZES[] = {3, 5, 9};
#define NUM_SE_SIZES (int)(sizeof(TEST_SE_SIZES) / sizeof(TEST_SE_SIZES[0]))

static const int VALIDATION_BLOCK_DIMS[] = {128, 256, 512, 1024};
#define NUM_VALIDATION_BLOCK_DIMS \
    (int)(sizeof(VALIDATION_BLOCK_DIMS) / sizeof(VALIDATION_BLOCK_DIMS[0]))

// La validazione prova più altezze del tile shared. La baseline global è
// volutamente naïve e ignora questo parametro.
static const int VALIDATION_ROWS_PER_BLOCK[] = {1, 4, 8, 16};
#define NUM_VALIDATION_ROWS_PER_BLOCK \
    (int)(sizeof(VALIDATION_ROWS_PER_BLOCK) / sizeof(VALIDATION_ROWS_PER_BLOCK[0]))

// Parametri fissi delle due implementazioni durante lo sweep.
#define BENCHMARK_BLOCK_DIM 256
#define SHARED_TILE_ROWS 8

#define MAX_SAMPLES (2 * TIMED_RUNS)

static timing_sample run_samples[MAX_SAMPLES];
static int sample_count = 0;

static matrix  source[MAX_BATCH];
static matrix  working[MAX_BATCH];
static matrix* batch[MAX_BATCH];

// Batch separato per il riferimento CPU, cosi' i due output coesistono e sono
// confrontabili pixel per pixel.
static matrix  ref_working[MAX_BATCH];
static matrix* ref_batch[MAX_BATCH];

typedef void (*morph_op)(matrix**, matrix*, int, cuda_config);
typedef void (*ref_op)(matrix**, matrix*, int);

// ---------------------------------------------------------------------------
// Gestione dei buffer
// ---------------------------------------------------------------------------

static void setup_batch(char** tiles, matrix* tile_buffer, int grid_rows, int grid_cols,
                        int batch_size, int with_reference) {
    int per_mosaic = grid_rows * grid_cols;

    for (int k = 0; k < batch_size; k++) {
        build_mosaic_image(&source[k], tile_buffer,
                           (const char**)&tiles[k * per_mosaic], grid_rows, grid_cols);

        allocate_matrix(&working[k], source[k].rows, source[k].cols);
        batch[k] = &working[k];

        if (with_reference) {
            allocate_matrix(&ref_working[k], source[k].rows, source[k].cols);
            ref_batch[k] = &ref_working[k];
        }
    }
}

static void teardown_batch(int batch_size) {
    for (int k = 0; k < batch_size; k++) {
        free_matrix(&source[k]);
        free_matrix(&working[k]);
        free_matrix(&ref_working[k]);
    }
}

static void reset_working(matrix* dst, int batch_size) {
    for (int k = 0; k < batch_size; k++) {
        memcpy(dst[k].data[0], source[k].data[0],
               (size_t)source[k].rows * source[k].cols * sizeof(u_int8_t));
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

static void benchmark_gpu(morph_op op, const char* label, matrix* se,
                          int batch_size, cuda_config cfg) {
    for (int r = 0; r < WARMUP_RUNS; r++) {
        reset_working(working, batch_size);
        op(batch, se, batch_size, cfg);
    }

    double sum = 0.0, min = DBL_MAX;
    for (int r = 0; r < TIMED_RUNS; r++) {
        reset_working(working, batch_size);
        op(batch, se, batch_size, cfg);
        run_samples[sample_count++] = (timing_sample){label, r, last_kernel_seconds};
        sum += last_kernel_seconds;
        if (last_kernel_seconds < min) min = last_kernel_seconds;
    }

    printf("  %-9s kernel=%.5f s  min=%.5f s  (timer host, per immagine)\n",
           label, sum / TIMED_RUNS, min);
}

// ---------------------------------------------------------------------------
// Validazione
// ---------------------------------------------------------------------------

static int validate_op(const char* label, morph_op gpu_op, ref_op cpu_op,
                       matrix* se, int batch_size, cuda_config cfg) {
    reset_working(working, batch_size);
    gpu_op(batch, se, batch_size, cfg);

    reset_working(ref_working, batch_size);
    cpu_op(ref_batch, se, batch_size);

    int diffs = compare_batches(batch, ref_batch, batch_size);
    printf("    %-9s block=%-5d shared=%d rows/blk=%-3d -> %s (%d pixel diversi)\n",
           label, cfg.block_dim, cfg.use_shared_memory, cfg.rows_per_block,
           diffs == 0 ? "ok" : "FAIL", diffs);
    return diffs;
}

static int run_validation(matrix* structuring_element, int max_threads) {
    int failures = 0;

    printf("\n\n########## Validazione contro il riferimento CPU ##########\n");

    for (int e = 0; e < NUM_SE_SIZES; e++) {
        int se_size = TEST_SE_SIZES[e];

        free_matrix(structuring_element);
        allocate_matrix(structuring_element, se_size, se_size);
        for (int i = 0; i < se_size; i++)
            for (int j = 0; j < se_size; j++)
                structuring_element->data[i][j] = 1;

        printf("\n===== Elemento strutturante %dx%d =====\n", se_size, se_size);

        for (int b = 0; b < NUM_VALIDATION_BLOCK_DIMS; b++) {
            int block_dim = VALIDATION_BLOCK_DIMS[b];
            if (block_dim > max_threads) continue;

            for (int shared = 0; shared <= 1; shared++) {
                // Con quattro pixel orizzontali per thread, 512/1024 thread
                // produrrebbero tile shared molto piu' larghi del benchmark e,
                // per SE 9x9, potrebbero superare il limite per blocco. Restano
                // comunque validati per il kernel naive.
                if (shared && block_dim > BENCHMARK_BLOCK_DIM) continue;

                int num_r = shared ? NUM_VALIDATION_ROWS_PER_BLOCK : 1;

                for (int rr = 0; rr < num_r; rr++) {
                    int rows_per_block = shared ? VALIDATION_ROWS_PER_BLOCK[rr] : 1;
                    cuda_config cfg = {shared, block_dim, rows_per_block};

                    failures += validate_op("erosion", image_erosion, ref_erosion,
                                            structuring_element, VALIDATION_BATCH, cfg);
                    failures += validate_op("opening", image_opening, ref_opening,
                                            structuring_element, VALIDATION_BATCH, cfg);
                }
            }
        }

    }

    return failures;
}

// ---------------------------------------------------------------------------

int main(void) {
    char run_id[32];
    time_t now = time(NULL);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    strftime(run_id, sizeof(run_id), "%Y-%m-%d %H:%M:%S", &tm_info);

    int sm_count = cuda_multiprocessor_count();
    int max_threads = cuda_max_threads_per_block();
    printf("GPU: %d SM, %d thread per blocco al massimo\n", sm_count, max_threads);

    char** tiles = NULL;
    int ntiles = scan_dataset(DATASET_DIR, &tiles);
    if (ntiles <= 0) {
        fprintf(stderr, "Nessuna immagine trovata in %s\n", DATASET_DIR);
        return EXIT_FAILURE;
    }
    printf("Dataset: %d tile trovati\n", ntiles);

    matrix structuring_element = {0};
    matrix tile_buffer = {0};

    for (int k = 0; k < MAX_BATCH; k++) {
        source[k]      = (matrix){0};
        working[k]     = (matrix){0};
        ref_working[k] = (matrix){0};
    }

    // Valida entrambe le modalità di lancio prima di scrivere misure nel CSV.
    setup_batch(tiles, &tile_buffer, 1, 1, VALIDATION_BATCH, 1);
    int failures = run_validation(&structuring_element, max_threads);
    teardown_batch(VALIDATION_BATCH);
    if (failures != 0) {
        fprintf(stderr, "\nVALIDAZIONE FALLITA: %d differenze totali rispetto al riferimento\n", failures);
        free_matrix(&structuring_element);
        free_matrix(&tile_buffer);
        free_dataset(tiles, ntiles);
        return EXIT_FAILURE;
    }
    printf("\nValidazione superata: output identico al riferimento CPU.\n");

    // -----------------------------------------------------------------------
    // Scaling rispetto alla dimensione del problema
    // -----------------------------------------------------------------------
    printf("\n\n########## Scaling sulla dimensione del problema ##########\n");

    for (int e = 0; e < NUM_SE_SIZES; e++) {
        int se_size = TEST_SE_SIZES[e];

        free_matrix(&structuring_element);
        allocate_matrix(&structuring_element, se_size, se_size);
        for (int i = 0; i < se_size; i++)
            for (int j = 0; j < se_size; j++)
                structuring_element.data[i][j] = 1;

        printf("\n===== Elemento strutturante %dx%d =====\n", se_size, se_size);

        for (int n = 0; n < NUM_GRID_ROWS; n++) {
            int grid_rows = TEST_GRID_ROWS[n];

            int needed = grid_rows * PROBLEM_COLS * PROBLEM_BATCH;
            if (needed > ntiles) {
                printf("\n--- mosaico %dx%d: servirebbero %d tile, disponibili %d: saltato ---\n",
                       grid_rows, PROBLEM_COLS, needed, ntiles);
                continue;
            }

            setup_batch(tiles, &tile_buffer, grid_rows, PROBLEM_COLS, PROBLEM_BATCH, 0);

            for (int shared = 0; shared <= 1; shared++) {
                cuda_config cfg = {
                    shared, BENCHMARK_BLOCK_DIM,
                    shared ? SHARED_TILE_ROWS : 1
                };

                printf("\n--- immagine %dx%d, %s ---\n",
                       source[0].rows, source[0].cols,
                       shared ? "shared" : "naive");
                sample_count = 0;

                benchmark_gpu(image_erosion, "erosion", &structuring_element,
                              PROBLEM_BATCH, cfg);
                benchmark_gpu(image_opening, "opening", &structuring_element,
                              PROBLEM_BATCH, cfg);

                run_config log = {"problem_size", "cuda", BENCHMARK_BLOCK_DIM,
                                  sm_count, shared,
                                  cfg.rows_per_block,
                                  se_size, source[0].rows, source[0].cols, PROBLEM_BATCH};
                log_timings_csv(run_id, log, run_samples, sample_count);
            }

            teardown_batch(PROBLEM_BATCH);
        }
    }

    printf("\nFreeing allocated memory...\n");
    free_matrix(&structuring_element);
    free_matrix(&tile_buffer);
    free_dataset(tiles, ntiles);
    return 0;
}
