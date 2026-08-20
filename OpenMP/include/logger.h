#ifndef LOGGER_H
#define LOGGER_H

typedef struct {
    const char* operation;
    int run_index;
    double seconds;
} timing_sample;

// mode: "strong" (problema fisso, thread crescenti) oppure "weak" (il problema
//       cresce con i thread, cosi' il lavoro per thread resta costante).
// implementation: "sequential_scalar" / "sequential_simd" per i due baseline in
//       C puro, "parallel" per la versione OpenMP. Le righe sequenziali usano threads=1.
void log_timings_csv(const char* run_id, const char* mode, const char* implementation,
                     timing_sample* samples, int count,
                     int threads, int se_size, int image_rows, int image_cols);

#endif
