#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_SHARED_PIXELS_PER_THREAD 4

typedef struct {
    int use_shared_memory;
    int block_dim;
    int rows_per_block;
} cuda_config;

int cuda_multiprocessor_count(void);
int cuda_max_threads_per_block(void);

#ifdef __cplusplus
}
#endif

#endif
