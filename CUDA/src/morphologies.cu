#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#include "morphologies.h"
#include "image.h"

double last_op_seconds = 0.0;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__
    static void ErosionKernel(u_int8_t* d_img, u_int8_t* d_se, u_int8_t* d_scratch, int HEIGHT, int WIDTH, int se_rows, int se_cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int chunk = blockIdx.z; // Get the index of the image in the array
        if (row >= HEIGHT || col >= WIDTH) return; // Check if the thread is within the image bounds

        size_t base = (size_t)chunk * HEIGHT * WIDTH; // Offset of this image inside the batch
        u_int8_t min_value = 255;

        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                int img_row = row + i - se_rows / 2;
                int img_col = col + j - se_cols / 2;

                if (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH) {
                    if (d_se[i * se_cols + j] == 1) {
                        min_value = min(min_value, d_img[base + img_row * WIDTH + img_col]);
                    }
                }
            }
        }
        d_scratch[base + row * WIDTH + col] = min_value;
    }


extern "C" void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int size) {
    (void)scratch; // niente padding: il bounds check nel kernel fa lo stesso lavoro

    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);

    int max_threads_per_block = device_prop.maxThreadsPerBlock;
    int block_dim = (int)sqrt(max_threads_per_block); // Calculate the maximum block dimension based on max threads per block
    int shared_mem_per_block = device_prop.sharedMemPerBlock;
    int tile_size = (int)sqrt(shared_mem_per_block / sizeof(u_int8_t)); // Calculate the maximum tile size based on shared memory

    // Assuming all images have the same dimensions
    int rows = img[0]->rows;
    int cols = img[0]->cols;

    dim3 block(block_dim, block_dim);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, size);

    size_t image_mem_size = (size_t)rows * cols * sizeof(u_int8_t);
    size_t mem_size = image_mem_size * size;
    size_t se_mem_size = (size_t)se_rows * se_cols * sizeof(u_int8_t);

    u_int8_t* d_img; u_int8_t* d_se; u_int8_t* d_scratch;

    cudaMalloc((void**)&d_img, mem_size);
    cudaMalloc((void**)&d_se, se_mem_size);
    cudaMalloc((void**)&d_scratch, mem_size);
    cudaMemcpy(d_se, structuring_element->data[0], se_mem_size, cudaMemcpyHostToDevice);

    double op_t_start = now_seconds();

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(d_img + k * rows * cols, img[k]->data[0], image_mem_size, cudaMemcpyHostToDevice);
    }

    ErosionKernel<<<grid, block>>>(d_img, d_se, d_scratch, rows, cols, se_rows, se_cols);

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(img[k]->data[0], d_scratch + k * rows * cols, image_mem_size, cudaMemcpyDeviceToHost);
    }

    last_op_seconds = (now_seconds() - op_t_start) / size;

    cudaFree(d_img);
    cudaFree(d_se);
    cudaFree(d_scratch);
}

__global__
    static void DilationKernel(u_int8_t* d_img, u_int8_t* d_se, u_int8_t* d_scratch, int HEIGHT, int WIDTH, int se_rows, int se_cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int chunk = blockIdx.z; // Get the index of the image in the array
        if (row >= HEIGHT || col >= WIDTH) return; // Check if the thread is within the image bounds

        size_t base = (size_t)chunk * HEIGHT * WIDTH;
        u_int8_t max_value = 0;

        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                int img_row = row + i - se_rows / 2;
                int img_col = col + j - se_cols / 2;

                if (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH) {
                    if (d_se[i * se_cols + j] == 1) {
                        max_value = max(max_value, d_img[base + img_row * WIDTH + img_col]);
                    }
                }
            }
        }
        d_scratch[base + row * WIDTH + col] = max_value;
    }



extern "C" void image_opening(matrix** img, matrix* structuring_element, matrix* scratch, int size) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);

    int max_threads_per_block = device_prop.maxThreadsPerBlock;
    int block_dim = (int)sqrt(max_threads_per_block); // Calculate the maximum block dimension based on max threads per block

    // Assuming all images have the same dimensions
    int rows = img[0]->rows;
    int cols = img[0]->cols;

    dim3 block(block_dim, block_dim);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, size);

    size_t image_mem_size = (size_t)rows * cols * sizeof(u_int8_t);
    size_t mem_size = image_mem_size * size;
    size_t se_mem_size = (size_t)se_rows * se_cols * sizeof(u_int8_t);

    u_int8_t* d_img; u_int8_t* d_se; u_int8_t* d_intermediate;

    cudaMalloc((void**)&d_img, mem_size);
    cudaMalloc((void**)&d_se, se_mem_size);
    cudaMalloc((void**)&d_intermediate, mem_size);
    cudaMemcpy(d_se, structuring_element->data[0], se_mem_size, cudaMemcpyHostToDevice);

    double op_t_start = now_seconds();

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(d_img + k * rows * cols, img[k]->data[0], image_mem_size, cudaMemcpyHostToDevice);
    }

    ErosionKernel<<<grid, block>>>(d_img, d_se, d_intermediate, rows, cols, se_rows, se_cols);
    DilationKernel<<<grid, block>>>(d_intermediate, d_se, d_img, rows, cols, se_rows, se_cols);

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(img[k]->data[0], d_img + k * rows * cols, image_mem_size, cudaMemcpyDeviceToHost);
    }

    last_op_seconds = (now_seconds() - op_t_start) / size;

    cudaFree(d_img);
    cudaFree(d_se);
    cudaFree(d_intermediate);
}
