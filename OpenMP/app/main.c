#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#include "matrix.h"
#include "image.h"
#include "morphologies.h"
#include "reference.h"
#include "logger.h"

#define DATASET_DIR "../brain-cancer-mri-dataset/Brain_Cancer raw MRI data/Brain_Cancer"
#define WARMUP_RUNS 2
#define TIMED_RUNS 10
#define PIPELINE_BATCH 16

// Weak scaling (legge di Gustafson-Barsis): il carico cresce con i thread, cosi'
// il lavoro per thread resta costante. Si scalano le RIGHE del mosaico, che sono
// l'asse su cui il loop parallelo distribuisce le iterazioni: a p thread il
// mosaico ha p righe di tile. Il batch resta fisso, altrimenti cambierebbe anche
// la profondita' della pipeline e i due effetti sarebbero indistinguibili.
#define WEAK_COLS  2
#define WEAK_BATCH 8

// La validazione usa un mosaico piccolo e un batch ridotto, perche' il
// confronto viene eseguito contro il riferimento scalare.
#define VALIDATION_BATCH 2
#define VALIDATION_SE 3
#define VALIDATION_THREADS 2

// Dimensioni dell'elemento strutturante. Il costo per pixel cresce col numero
// di celle (9, 25, 81), quindi questo asse varia l'intensita' aritmetica del
// kernel a parita' di traffico di memoria: utile per capire quanto le
// operazioni siano memory-bound o compute-bound.
static const int TEST_SE_SIZES[] = {3, 5, 9};
#define NUM_SE_SIZES (int)(sizeof(TEST_SE_SIZES) / sizeof(TEST_SE_SIZES[0]))

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

static const int TEST_THREAD_COUNTS[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
#define NUM_THREAD_COUNTS (int)(sizeof(TEST_THREAD_COUNTS) / sizeof(TEST_THREAD_COUNTS[0]))

#define MAX_SAMPLES (2 * TIMED_RUNS)
static timing_sample run_samples[MAX_SAMPLES];
static int sample_count = 0;

// source: mosaici intatti, ricostruiti ad ogni dimensione.
// working: copie di lavoro, rigenerate prima di ogni run (le operazioni
//          modificano l'immagine in place).
static matrix  source[PIPELINE_BATCH];
static matrix  working[PIPELINE_BATCH];
static matrix* batch[PIPELINE_BATCH];

// Output separato del riferimento scalare: deve coesistere con quello OpenMP
// per consentire il confronto pixel per pixel.
static matrix  seq_working[PIPELINE_BATCH];
static matrix* seq_batch[PIPELINE_BATCH];

typedef void (*morph_op)(matrix**, matrix*, matrix*, int);

static void benchmark_parallel(morph_op op, const char* label, matrix* se, matrix* scratch,
                               int batch_size) {
    for (int r = 0; r < WARMUP_RUNS; r++) {
        for (int k = 0; k < batch_size; k++) copy_matrix(&source[k], &working[k]);
        op(batch, se, scratch, batch_size);
    }

    double sum = 0.0, min = DBL_MAX;
    for (int r = 0; r < TIMED_RUNS; r++) {
        for (int k = 0; k < batch_size; k++) copy_matrix(&source[k], &working[k]);
        op(batch, se, scratch, batch_size);
        #pragma omp master
        {
            double d = last_op_seconds;
            run_samples[sample_count++] = (timing_sample){label, r, d};
            sum += d;
            if (d < min) min = d;
        }
    }

    #pragma omp master
    {
        printf("  %-9s mean=%.5f s  min=%.5f s  (per immagine)\n",
               label, sum / TIMED_RUNS, min);
    }
    #pragma omp barrier
}

typedef void (*seq_batch_op)(matrix**, matrix*, int);

static void record_sample(const char* label, int run_index, double seconds,
                          double* sum, double* min) {
    run_samples[sample_count++] = (timing_sample){label, run_index, seconds};
    *sum += seconds;
    if (seconds < *min) *min = seconds;
}

static void report(const char* label, double sum, double min) {
    printf("  %-9s mean=%.5f s  min=%.5f s  (per immagine)\n", label, sum / TIMED_RUNS, min);
}

// clock: puntatore alla variabile di timing della variante in uso.
static void benchmark_seq_batch(seq_batch_op op, const char* label, matrix* se,
                                const double* clock, int batch_size) {
    for (int r = 0; r < WARMUP_RUNS; r++) {
        for (int k = 0; k < batch_size; k++) copy_matrix_serial(&source[k], &working[k]);
        op(batch, se, batch_size);
    }

    double sum = 0.0, min = DBL_MAX;
    for (int r = 0; r < TIMED_RUNS; r++) {
        for (int k = 0; k < batch_size; k++) copy_matrix_serial(&source[k], &working[k]);
        op(batch, se, batch_size);
        record_sample(label, r, *clock, &sum, &min);
    }
    report(label, sum, min);
}

// Esegue le due operazioni per una delle varianti sequenziali e logga il blocco.
static void run_sequential_baseline(const char* impl_name, const char* run_id, matrix* se,
                                    seq_batch_op erosion, seq_batch_op opening,
                                    const double* clock) {
    printf("\n--- %s ---\n", impl_name);
    sample_count = 0;
    benchmark_seq_batch(erosion, "erosion", se, clock, PIPELINE_BATCH);
    benchmark_seq_batch(opening, "opening", se, clock, PIPELINE_BATCH);
    log_timings_csv(run_id, "strong", impl_name, run_samples, sample_count, 1,
                    se->rows, source[0].rows, source[0].cols);
}

// ---------------------------------------------------------------------------
// Validazione della correttezza contro il riferimento scalare
// ---------------------------------------------------------------------------

static int compare_batches(matrix** a, matrix** b, int batch_size) {
    int diffs = 0;

    for (int k = 0; k < batch_size; k++) {
        if (a[k]->rows != b[k]->rows || a[k]->cols != b[k]->cols) {
            fprintf(stderr, "Confronto impossibile: immagine %d e' %dx%d contro %dx%d\n",
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

static int validate_op(const char* label, morph_op parallel_op, seq_batch_op scalar_op,
                       matrix* se, matrix* scratch, int num_threads) {
    #pragma omp parallel num_threads(num_threads)
    {
        for (int k = 0; k < VALIDATION_BATCH; k++)
            copy_matrix(&source[k], &working[k]);
        parallel_op(batch, se, scratch, VALIDATION_BATCH);
    }

    for (int k = 0; k < VALIDATION_BATCH; k++)
        copy_matrix_serial(&source[k], &seq_working[k]);
    scalar_op(seq_batch, se, VALIDATION_BATCH);

    int diffs = compare_batches(batch, seq_batch, VALIDATION_BATCH);
    printf("    %-9s threads=%-3d -> %s (%d pixel diversi)\n",
           label, num_threads, diffs == 0 ? "ok" : "FAIL", diffs);

    return diffs < 0 ? 1 : diffs;
}

static int run_validation(char** tiles, matrix* tile_buffer, matrix* se, matrix* scratch) {
    int failures = 0;

    printf("\n\n########## Validazione contro il riferimento sequenziale scalare ##########\n");

    for (int k = 0; k < VALIDATION_BATCH; k++) {
        build_mosaic_image(&source[k], tile_buffer, (const char**)&tiles[k], 1, 1);
    }

    free_matrix(se);
    allocate_matrix(se, VALIDATION_SE, VALIDATION_SE);
    for (int i = 0; i < VALIDATION_SE; i++)
        for (int j = 0; j < VALIDATION_SE; j++)
            se->data[i][j] = 1;

    printf("\nImmagini %dx%d, batch %d, SE %dx%d, %d thread\n",
           source[0].rows, source[0].cols, VALIDATION_BATCH,
           VALIDATION_SE, VALIDATION_SE, VALIDATION_THREADS);

    failures += validate_op("erosion", image_erosion, seq_erosion_scalar,
                            se, scratch, VALIDATION_THREADS);
    failures += validate_op("opening", image_opening, seq_opening_scalar,
                            se, scratch, VALIDATION_THREADS);

    for (int k = 0; k < VALIDATION_BATCH; k++) {
        free_matrix(&source[k]);
        free_matrix(&working[k]);
        free_matrix(&seq_working[k]);
    }

    return failures;
}

int main(void) {
    char run_id[32];
    time_t now = time(NULL);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    strftime(run_id, sizeof(run_id), "%Y-%m-%d %H:%M:%S", &tm_info);

    char** tiles = NULL;
    int ntiles = scan_dataset(DATASET_DIR, &tiles);
    if (ntiles <= 0) {
        fprintf(stderr, "Nessuna immagine trovata in %s\n", DATASET_DIR);
        return EXIT_FAILURE;
    }
    printf("Dataset: %d tile trovati\n", ntiles);

    int max_tiles_needed = TEST_GRID_SIZES[NUM_TEST_SIZES-1].rows *
                           TEST_GRID_SIZES[NUM_TEST_SIZES-1].cols * PIPELINE_BATCH;
    if (max_tiles_needed > ntiles) {
        fprintf(stderr, "Servono %d tile (griglia piu' grande x batch %d) ma ne sono disponibili %d\n",
                max_tiles_needed, PIPELINE_BATCH, ntiles);
        free_dataset(tiles, ntiles);
        return EXIT_FAILURE;
    }

    matrix structuring_element = {0};
    matrix scratch     = {0};
    matrix tile_buffer = {0};

    for (int k = 0; k < PIPELINE_BATCH; k++) {
        source[k]  = (matrix){0};
        working[k] = (matrix){0};
        batch[k]   = &working[k];
        seq_working[k] = (matrix){0};
        seq_batch[k]   = &seq_working[k];
    }

    // La correttezza viene verificata prima dei benchmark, cosi' un errore non
    // produce campioni CSV che verrebbero poi analizzati come risultati validi.
    int failures = run_validation(tiles, &tile_buffer, &structuring_element, &scratch);
    if (failures != 0) {
        fprintf(stderr, "\nVALIDAZIONE FALLITA: %d differenze totali rispetto al riferimento scalare\n",
                failures);
        free_matrix(&structuring_element);
        free_matrix(&scratch);
        free_matrix(&tile_buffer);
        free_dataset(tiles, ntiles);
        return EXIT_FAILURE;
    }
    printf("\nValidazione superata: output OpenMP identico al riferimento scalare.\n");

    for (int s = 0; s < NUM_TEST_SIZES; s++) {
        int grid_rows  = TEST_GRID_SIZES[s].rows;
        int grid_cols  = TEST_GRID_SIZES[s].cols;
        int per_mosaic = grid_rows * grid_cols;

        // Ogni elemento del batch e' un mosaico distinto, costruito da tile
        // diversi: fetta [k*per_mosaic, (k+1)*per_mosaic) del dataset.
        for (int k = 0; k < PIPELINE_BATCH; k++) {
            build_mosaic_image(&source[k], &tile_buffer,
                               (const char**)&tiles[k * per_mosaic],
                               grid_rows, grid_cols);
        }

        printf("\n########## Immagine %d x %d (griglia %dx%d, batch di %d) ##########\n",
               source[0].rows, source[0].cols, grid_rows, grid_cols, PIPELINE_BATCH);

        for (int e = 0; e < NUM_SE_SIZES; e++) {
            int se_size = TEST_SE_SIZES[e];

            free_matrix(&structuring_element);
            allocate_matrix(&structuring_element, se_size, se_size);
            for (int i = 0; i < se_size; i++)
                for (int j = 0; j < se_size; j++)
                    structuring_element.data[i][j] = 1;

            printf("\n===== Elemento strutturante %dx%d (%d celle) =====\n",
                   se_size, se_size, se_size * se_size);

            run_sequential_baseline("sequential_scalar", run_id, &structuring_element,
                                    seq_erosion_scalar, seq_opening_scalar,
                                    &last_seq_seconds_scalar);

            run_sequential_baseline("sequential_simd", run_id, &structuring_element,
                                    seq_erosion_simd, seq_opening_simd,
                                    &last_seq_seconds_simd);

            for (int t = 0; t < NUM_THREAD_COUNTS; t++) {
                int num_threads = TEST_THREAD_COUNTS[t];
                printf("\n--- parallelo, %d thread ---\n", num_threads);

                #pragma omp parallel num_threads(num_threads)
                {
                    #pragma omp single
                    sample_count = 0;

                    benchmark_parallel(image_erosion,  "erosion",  &structuring_element, &scratch, PIPELINE_BATCH);
                    benchmark_parallel(image_opening,  "opening",  &structuring_element, &scratch, PIPELINE_BATCH);

                    #pragma omp single
                    log_timings_csv(run_id, "strong", "parallel", run_samples, sample_count,
                                    num_threads, se_size, source[0].rows, source[0].cols);
                }
            }
        }

        for (int k = 0; k < PIPELINE_BATCH; k++) {
            free_matrix(&source[k]);
            free_matrix(&working[k]);
        }
    }

    // ---------------------------------------------------------------------
    // Weak scaling: carico proporzionale ai thread (Gustafson-Barsis)
    // ---------------------------------------------------------------------
    printf("\n\n########## Weak scaling (righe del mosaico ∝ thread) ##########\n");

    for (int e = 0; e < NUM_SE_SIZES; e++) {
        int se_size = TEST_SE_SIZES[e];

        free_matrix(&structuring_element);
        allocate_matrix(&structuring_element, se_size, se_size);
        for (int i = 0; i < se_size; i++)
            for (int j = 0; j < se_size; j++)
                structuring_element.data[i][j] = 1;

        printf("\n===== Elemento strutturante %dx%d =====\n", se_size, se_size);

        for (int k = 0; k < WEAK_BATCH; k++)
            build_mosaic_image(&source[k], &tile_buffer,
                               (const char**)&tiles[k * WEAK_COLS], 1, WEAK_COLS);

        printf("\n--- sequenziale, carico base (%dx%d) ---\n", source[0].rows, source[0].cols);
        sample_count = 0;
        benchmark_seq_batch(seq_erosion_simd, "erosion", &structuring_element,
                            &last_seq_seconds_simd, WEAK_BATCH);
        benchmark_seq_batch(seq_opening_simd, "opening", &structuring_element,
                            &last_seq_seconds_simd, WEAK_BATCH);
        log_timings_csv(run_id, "weak", "sequential_simd", run_samples, sample_count, 1,
                        se_size, source[0].rows, source[0].cols);

        for (int k = 0; k < WEAK_BATCH; k++) {
            free_matrix(&source[k]);
            free_matrix(&working[k]);
        }

        for (int t = 0; t < NUM_THREAD_COUNTS; t++) {
            int num_threads = TEST_THREAD_COUNTS[t];

            // Carico proporzionale: num_threads righe di tile invece di una.
            int needed = num_threads * WEAK_COLS * WEAK_BATCH;
            if (needed > ntiles) {
                printf("\n--- %d thread: servirebbero %d tile, disponibili %d: saltato ---\n",
                       num_threads, needed, ntiles);
                continue;
            }
            for (int k = 0; k < WEAK_BATCH; k++)
                build_mosaic_image(&source[k], &tile_buffer,
                                   (const char**)&tiles[k * num_threads * WEAK_COLS],
                                   num_threads, WEAK_COLS);

            printf("\n--- parallelo, %d thread, carico %dx%d ---\n",
                   num_threads, source[0].rows, source[0].cols);

            #pragma omp parallel num_threads(num_threads)
            {
                #pragma omp single
                sample_count = 0;

                benchmark_parallel(image_erosion, "erosion", &structuring_element, &scratch, WEAK_BATCH);
                benchmark_parallel(image_opening, "opening", &structuring_element, &scratch, WEAK_BATCH);

                #pragma omp single
                log_timings_csv(run_id, "weak", "parallel", run_samples, sample_count,
                                num_threads, se_size, source[0].rows, source[0].cols);
            }

            for (int k = 0; k < WEAK_BATCH; k++) {
                free_matrix(&source[k]);
                free_matrix(&working[k]);
            }
        }
    }

    printf("\nFreeing allocated memory...\n");
    free_matrix(&structuring_element);
    free_matrix(&scratch);
    free_matrix(&tile_buffer);
    free_dataset(tiles, ntiles);

    return 0;
}
