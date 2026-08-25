#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cuda_config.h"
#include "cuda_utils.cuh"

void cuda_check(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s -> %s\n", file, line, call, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void check_launch(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch fallito -> %s\n", what, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

const cudaDeviceProp& device_properties(void) {
    static cudaDeviceProp prop;
    static bool loaded = false;
    if (!loaded) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        loaded = true;
    }
    return prop;
}

extern "C" int cuda_multiprocessor_count(void) {
    return device_properties().multiProcessorCount;
}

extern "C" int cuda_max_threads_per_block(void) {
    return device_properties().maxThreadsPerBlock;
}
