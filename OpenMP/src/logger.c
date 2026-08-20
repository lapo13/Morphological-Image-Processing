#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "logger.h"

#define RESULTS_DIR "experiment_run"
#define RESULTS_FILE RESULTS_DIR "/results.csv"

void log_timings_csv(const char* run_id, const char* mode, const char* implementation,
                     timing_sample* samples, int count,
                     int threads, int se_size, int image_rows, int image_cols) {
    mkdir(RESULTS_DIR, 0755);

    struct stat st;
    int file_exists = (stat(RESULTS_FILE, &st) == 0);

    FILE* f = fopen(RESULTS_FILE, "a");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", RESULTS_FILE);
        return;
    }

    if (!file_exists) {
        fprintf(f, "run_id,mode,implementation,threads,se_size,image_rows,image_cols,operation,run_index,seconds\n");
    }

    for (int i = 0; i < count; i++) {
        fprintf(f, "%s,%s,%s,%d,%d,%d,%d,%s,%d,%.6f\n", run_id, mode, implementation,
                threads, se_size, image_rows, image_cols, samples[i].operation,
                samples[i].run_index, samples[i].seconds);
    }

    fclose(f);
    printf("  -> %d campioni in %s (%s, impl=%s, threads=%d, se=%dx%d, image=%dx%d)\n",
           count, RESULTS_FILE, mode, implementation, threads, se_size, se_size,
           image_rows, image_cols);
}
