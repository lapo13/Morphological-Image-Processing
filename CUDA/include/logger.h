#ifndef LOGGER_H
#define LOGGER_H

typedef struct {
    const char* operation;
    int run_index;
    double seconds;
} timing_sample;

typedef struct {
    const char* mode;
    const char* implementation;
    int block_dim;
    int sm_count;
    int use_shared;
    int rows_per_block;
    int se_size;
    int image_rows;
    int image_cols;
    int batch;
} run_config;

void log_timings_csv(const char* run_id, run_config cfg, timing_sample* samples, int count);

#endif
