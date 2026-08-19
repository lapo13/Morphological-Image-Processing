#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <

#include "morphologies.h"
#include "image.h"

double last_op_seconds = 0.0;

__global__
    void ErosionKernel(u_int8_t* d_img, u_int8_t* d_se, u_int8_t* d_scratch, int HEIGHT, int WIDTH, int se_rows, int se_cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= HEIGHT || col >= WIDTH) return; // Check if the thread is within the image bounds

        u_int8_t min_value = 255;

        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                int img_row = row + i - se_rows / 2;
                int img_col = col + j - se_cols / 2;

                if (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH) {
                    if (d_se[i * se_cols + j] == 1) {
                        min_value = min(min_value, d_img[img_row * WIDTH + img_col]);
                    }
                }
            }
        }
        d_scratch[row * WIDTH + col] = min_value;
    }


extern "C" void image_erosion(matrix* img, matrix* structuring_element, matrix* scratch) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;

    pad_image(img, scratch, se_rows / 2, 255);

    int rows = img->rows;
    int cols = img->cols;

    int mem_size = rows * cols * sizeof(u_int8_t);
    int se_mem_size = se_rows * se_cols * sizeof(u_int8_t);

    u_int8_t* d_img; u_int8_t* d_se; u_int8_t* d_scratch;

    cudaMalloc((void**)&d_img, mem_size);
    cudaMemcpy(d_img, img->data[0], mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_se, se_mem_size);
    cudaMemcpy(d_se, structuring_element->data[0], se_mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_scratch, mem_size);

    ErosionKernel<<<dim3((cols + 15) / 16, (rows + 15) / 16), dim3(16, 16)>>>(d_img, d_se, d_scratch, rows, cols, se_rows, se_cols);

    cudaMemcpy(img->data[0], d_scratch, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_se);
    cudaFree(d_scratch);

    crop_image(img, scratch, se_rows / 2);
}

__global__
    void DilationKernel(u_int8_t* d_img, u_int8_t* d_se, u_int8_t* d_scratch, int HEIGHT, int WIDTH, int se_rows, int se_cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= HEIGHT || col >= WIDTH) return;

        u_int8_t max_value = 0;

        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                int img_row = row + i - se_rows / 2;
                int img_col = col + j - se_cols / 2;

                if (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH) {
                    if (d_se[i * se_cols + j] == 1) {
                        max_value = max(max_value, d_img[img_row * WIDTH + img_col]);
                    }
                }
            }
        }
        d_scratch[row * WIDTH + col] = max_value;
    }


extern "C" void image_dilation(matrix* img, matrix* structuring_element, matrix* scratch) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;

    pad_image(img, scratch, se_rows / 2, 0);

    int rows = img->rows;
    int cols = img->cols;

    int mem_size = rows * cols * sizeof(u_int8_t);
    int se_mem_size = se_rows * se_cols * sizeof(u_int8_t);

    u_int8_t* d_img; u_int8_t* d_se; u_int8_t* d_scratch;

    cudaMalloc((void**)&d_img, mem_size);
    cudaMemcpy(d_img, img->data[0], mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_se, se_mem_size);
    cudaMemcpy(d_se, structuring_element->data[0], se_mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_scratch, mem_size);

    DilationKernel<<<dim3((cols + 15) / 16, (rows + 15) / 16), dim3(16, 16)>>>(d_img, d_se, d_scratch, rows, cols, se_rows, se_cols);

    cudaMemcpy(img->data[0], d_scratch, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_se);
    cudaFree(d_scratch);

    crop_image(img, scratch, se_rows / 2);
}

// TODO: image_opening = erosione poi dilatazione.
extern "C" void image_opening(matrix* img, matrix* structuring_element, matrix* scratch) {
    (void)img; (void)structuring_element; (void)scratch;
    fprintf(stderr, "image_opening: kernel CUDA non ancora implementato\n");
    exit(EXIT_FAILURE);
}

// TODO: image_closing = dilatazione poi erosione.
extern "C" void image_closing(matrix* img, matrix* structuring_element, matrix* scratch) {
    (void)img; (void)structuring_element; (void)scratch;
    fprintf(stderr, "image_closing: kernel CUDA non ancora implementato\n");
    exit(EXIT_FAILURE);
}
