#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "logger.h"

#define RESULTS_DIR "experiment_run"
#define RESULTS_FILE RESULTS_DIR "/kernel_results.csv"

static int append_samples(const char* path, const char* run_id, run_config cfg,
                          timing_sample* samples, int count) {
    struct stat st;
    int file_exists = (stat(path, &st) == 0);

    FILE* f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        return 0;
    }

    if (!file_exists) {
        fprintf(f, "run_id,mode,implementation,block_dim,sm_count,use_shared,"
                   "rows_per_block,se_size,image_rows,image_cols,batch,operation,run_index,seconds\n");
    }

    for (int i = 0; i < count; i++) {
        fprintf(f, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%.9f\n",
                run_id, cfg.mode, cfg.implementation,
                cfg.block_dim, cfg.sm_count, cfg.use_shared,
                cfg.rows_per_block, cfg.se_size, cfg.image_rows, cfg.image_cols, cfg.batch,
                samples[i].operation, samples[i].run_index, samples[i].seconds);
    }

    fclose(f);
    return 1;
}

void log_timings_csv(const char* run_id, run_config cfg, timing_sample* samples, int count) {
    mkdir(RESULTS_DIR, 0755);

    if (!append_samples(RESULTS_FILE, run_id, cfg, samples, count)) return;

    printf("  -> %d campioni kernel in %s (%s, threads_x=%d, shared=%d, output rows/blk=%d, se=%dx%d, image=%dx%d)\n",
           count, RESULTS_FILE, cfg.mode, cfg.block_dim, cfg.use_shared,
           cfg.rows_per_block, cfg.se_size, cfg.se_size, cfg.image_rows, cfg.image_cols);
}
