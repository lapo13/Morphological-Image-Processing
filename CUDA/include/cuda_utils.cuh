#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

void cuda_check(cudaError_t err, const char* call, const char* file, int line);
void check_launch(const char* what);

const cudaDeviceProp& device_properties(void);

#endif
