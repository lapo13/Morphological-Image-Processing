#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "logger.h"

#define RESULTS_DIR "experiment_run"
#define RESULTS_FILE RESULTS_DIR "/results.csv"

void log_timings_csv(const char* run_id, timing_sample* samples, int count, int threads, int image_rows, int image_cols) {
    mkdir(RESULTS_DIR, 0755);

    struct stat st;
    int file_exists = (stat(RESULTS_FILE, &st) == 0);

    FILE* f = fopen(RESULTS_FILE, "a");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", RESULTS_FILE);
        return;
    }

    if (!file_exists) {
        fprintf(f, "run_id,threads,image_rows,image_cols,operation,run_index,seconds,vectorized\n");
    }

    for (int i = 0; i < count; i++) {
        fprintf(f, "%s,%d,%d,%d,%s,%d,%.6f,%d\n", run_id, threads, image_rows, image_cols,
                samples[i].operation, samples[i].run_index, samples[i].seconds, samples[i].vectorized);
    }

    fclose(f);
    printf("Timing results appended to %s (run=%s, threads=%d, image=%dx%d, samples=%d)\n", RESULTS_FILE, run_id, threads, image_rows, image_cols, count);
}
