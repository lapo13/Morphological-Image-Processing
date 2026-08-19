#ifndef LOGGER_H
#define LOGGER_H

typedef struct {
    const char* operation;
    int run_index;
    double seconds;
    int vectorized;
} timing_sample;

void log_timings_csv(const char* run_id, timing_sample* samples, int count, int threads, int image_rows, int image_cols);

#endif
