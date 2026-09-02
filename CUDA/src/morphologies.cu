#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#include "morphologies.h"
#include "cuda_utils.cuh"

double last_kernel_seconds = 0.0;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

constexpr int MAX_SE_ROWS = 9;
constexpr int MAX_SE_COLS = 9;

struct StructuringElement {
    int rows;
    int cols;
    int radius_y;
    int radius_x;

    uint8_t values[MAX_SE_ROWS * MAX_SE_COLS];
};

__constant__ StructuringElement c_se;

static void upload_structuring_element(matrix* structuring_element) {
    if (structuring_element->rows > MAX_SE_ROWS || structuring_element->cols > MAX_SE_COLS) {
        fprintf(stderr, "structuring element %dx%d oltre il massimo supportato %dx%d\n",
                structuring_element->rows, structuring_element->cols, MAX_SE_ROWS, MAX_SE_COLS);
        exit(EXIT_FAILURE);
    }

    StructuringElement se{};
    se.rows = structuring_element->rows;
    se.cols = structuring_element->cols;
    se.radius_y = se.rows / 2;
    se.radius_x = se.cols / 2;
    for (int i = 0; i < se.rows; ++i) {
        for (int j = 0; j < se.cols; ++j) {
            se.values[i * se.cols + j] = structuring_element->data[i][j];
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_se, &se, sizeof(StructuringElement)));
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__device__ inline uint8_t reduce(uint8_t a, uint8_t b, int is_erosion) {
    return is_erosion ? (a < b ? a : b) : (a > b ? a : b);
}

__global__ void MorphKernel(const uint8_t* d_in, uint8_t* d_out, int HEIGHT, int WIDTH,
                            int is_erosion) {
    const int se_rows = c_se.rows;
    const int se_cols = c_se.cols;
    const int radius_y = c_se.radius_y;
    const int radius_x = c_se.radius_x;

    // grid.x segue l'altezza (limite molto ampio), mentre grid.y contiene solo
    // i pochi gruppi orizzontali. Cosi' l'altezza non e' vincolata a 65535.
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int chunk = blockIdx.z;
    if (row >= HEIGHT || col >= WIDTH) return;

    const int base = chunk * HEIGHT * WIDTH;
    uint8_t acc = is_erosion ? 255 : 0;
    for (int i = 0; i < se_rows; ++i) {
        int img_row = row + i - radius_y;
        if (img_row < 0 || img_row >= HEIGHT) continue;

        for (int j = 0; j < se_cols; ++j) {
            int img_col = col + j - radius_x;
            if (img_col < 0 || img_col >= WIDTH) continue;
            if (c_se.values[i * se_cols + j] != 1) continue;

            acc = reduce(acc, d_in[base + img_row * WIDTH + img_col], is_erosion);
        }
    }

    d_out[base + row * WIDTH + col] = acc;
}

__global__ void SharedMorphKernel(const uint8_t* d_in, uint8_t* d_out, int HEIGHT, int WIDTH,
                                  int is_erosion, int rows_per_block) {
    extern __shared__ uchar4 shared_words[];

    const int se_rows = c_se.rows;
    const int se_cols = c_se.cols;
    const int radius_y = c_se.radius_y;
    const int radius_x = c_se.radius_x;

    const int tx = threadIdx.x;
    const int output_cols = blockDim.x * CUDA_SHARED_PIXELS_PER_THREAD;

    const int aligned_halo_x = (radius_x + 3) & ~3;
    const int tile_pitch = output_cols + 2 * aligned_halo_x;
    const int words_per_row = tile_pitch / 4;
    const int tile_rows = rows_per_block + 2 * radius_y;

    const uint8_t neutral = is_erosion ? 255 : 0;
    uint8_t* shared_pixels = reinterpret_cast<uint8_t*>(shared_words);

    const int row_base = blockIdx.y * rows_per_block;
    const int col_base = blockIdx.x * output_cols;
    const int output_col = col_base + tx * CUDA_SHARED_PIXELS_PER_THREAD;
    const int chunk = blockIdx.z;
    const int base = chunk * HEIGHT * WIDTH;

    for (int i = 0; i < tile_rows; ++i) {
        int img_row = row_base + i - radius_y;
        for (int word = tx; word < words_per_row; word += blockDim.x) {
            int img_col = col_base + word * 4 - aligned_halo_x;
            uchar4 packed = make_uchar4(neutral, neutral, neutral, neutral);

            if (img_row >= 0 && img_row < HEIGHT) {
                int input_offset = base + img_row * WIDTH + img_col;
                if (img_col >= 0 && img_col + 3 < WIDTH && (input_offset & 3) == 0) {
                    packed = *reinterpret_cast<const uchar4*>(
                        d_in + input_offset);
                } else {
                    if (img_col >= 0 && img_col < WIDTH)
                        packed.x = d_in[base + img_row * WIDTH + img_col];
                    if (img_col + 1 >= 0 && img_col + 1 < WIDTH)
                        packed.y = d_in[base + img_row * WIDTH + img_col + 1];
                    if (img_col + 2 >= 0 && img_col + 2 < WIDTH)
                        packed.z = d_in[base + img_row * WIDTH + img_col + 2];
                    if (img_col + 3 >= 0 && img_col + 3 < WIDTH)
                        packed.w = d_in[base + img_row * WIDTH + img_col + 3];
                }
            }
            shared_words[i * words_per_row + word] = packed;
        }
    }
    __syncthreads();

    if (output_col < WIDTH) {
        for (int r = 0; r < rows_per_block; ++r) {
            const int row = row_base + r;
            if (row >= HEIGHT) break;

            uint8_t acc0 = neutral;
            uint8_t acc1 = neutral;
            uint8_t acc2 = neutral;
            uint8_t acc3 = neutral;

            for (int i = 0; i < se_rows; ++i) {
                const int shared_start =
                    (r + i) * tile_pitch + aligned_halo_x + tx * 4 - radius_x;

                for (int x = 0; x < se_cols + 3; ++x) {
                    uint8_t value = shared_pixels[shared_start + x];

                    if (x < se_cols && c_se.values[i * se_cols + x] == 1)
                        acc0 = reduce(acc0, value, is_erosion);
                    if (x >= 1 && x - 1 < se_cols &&
                        c_se.values[i * se_cols + x - 1] == 1)
                        acc1 = reduce(acc1, value, is_erosion);
                    if (x >= 2 && x - 2 < se_cols &&
                        c_se.values[i * se_cols + x - 2] == 1)
                        acc2 = reduce(acc2, value, is_erosion);
                    if (x >= 3 && x - 3 < se_cols &&
                        c_se.values[i * se_cols + x - 3] == 1)
                        acc3 = reduce(acc3, value, is_erosion);
                }
            }

            const int output = base + row * WIDTH + output_col;
            d_out[output] = acc0;
            if (output_col + 1 < WIDTH) d_out[output + 1] = acc1;
            if (output_col + 2 < WIDTH) d_out[output + 2] = acc2;
            if (output_col + 3 < WIDTH) d_out[output + 3] = acc3;
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

typedef void (*launch_fn)(const uint8_t* d_in, uint8_t* d_out,
                          int rows, int cols, int se_rows, int se_cols,
                          int size, cuda_config cfg);

static void launch_morph(const uint8_t* d_in, uint8_t* d_out,
                         int rows, int cols, int se_rows, int se_cols,
                         int size, cuda_config cfg, int is_erosion) {
    const cudaDeviceProp& prop = device_properties();
    int threads_x = cfg.block_dim;
    if (threads_x <= 0 || threads_x > prop.maxThreadsPerBlock) {
        threads_x = prop.maxThreadsPerBlock;
    }

    const int rows_per_block = cfg.rows_per_block > 0 ? cfg.rows_per_block : 1;

    if (cfg.use_shared_memory) {
        const dim3 block(threads_x, 1, 1);
        const int output_cols = threads_x * CUDA_SHARED_PIXELS_PER_THREAD;
        const int radius_x = se_cols / 2;
        const int aligned_halo_x = (radius_x + 3) & ~3;
        const int tile_pitch = output_cols + 2 * aligned_halo_x;
        const int tile_rows = rows_per_block + 2 * (se_rows / 2);
        const size_t shared_bytes = (size_t)tile_rows * tile_pitch * sizeof(uint8_t);

        dim3 grid((unsigned int)((cols + output_cols - 1) / output_cols),
                  (unsigned int)((rows + rows_per_block - 1) / rows_per_block),
                  (unsigned int)size);
        SharedMorphKernel<<<grid, block, shared_bytes>>>(d_in, d_out, rows, cols,
                                                        is_erosion, rows_per_block);
        check_launch("SharedMorphKernel");
    } else {
        const int max_block_rows = prop.maxThreadsPerBlock / threads_x;
        const int block_rows = rows_per_block < max_block_rows
                                   ? rows_per_block
                                   : max_block_rows;
        const dim3 block(threads_x, block_rows, 1);
        const unsigned int blocks_per_row =
            (unsigned int)((cols + threads_x - 1) / threads_x);
        const unsigned int block_rows_count =
            (unsigned int)((rows + block_rows - 1) / block_rows);
        dim3 grid(block_rows_count, blocks_per_row, (unsigned int)size);
        MorphKernel<<<grid, block>>>(d_in, d_out, rows, cols, is_erosion);
        check_launch("MorphKernel");
    }
}

static void launch_erosion(const uint8_t* d_in, uint8_t* d_out,
                           int rows, int cols, int se_rows, int se_cols,
                           int size, cuda_config cfg) {
    launch_morph(d_in, d_out, rows, cols, se_rows, se_cols, size, cfg, 1);
}

static void launch_dilation(const uint8_t* d_in, uint8_t* d_out,
                            int rows, int cols, int se_rows, int se_cols,
                            int size, cuda_config cfg) {
    launch_morph(d_in, d_out, rows, cols, se_rows, se_cols, size, cfg, 0);
}

// ---------------------------------------------------------------------------
// I risultati intermedi non lasciano mai la GPU: una sola coppia di transfer per
// l'intera operazione, qualunque sia il numero di stage.
// ---------------------------------------------------------------------------

static void run_pipeline(matrix** img, matrix* structuring_element, int size,
                         cuda_config cfg, const launch_fn* stages, int nstages) {
    if (size <= 0 || nstages <= 0) return;

    const int se_rows = structuring_element->rows;
    const int se_cols = structuring_element->cols;

    // Tutte le immagini del batch hanno le stesse dimensioni.
    const int rows = img[0]->rows;
    const int cols = img[0]->cols;

    const size_t image_bytes = (size_t)rows * cols * sizeof(uint8_t);
    const size_t batch_bytes = image_bytes * size;

    uint8_t* buffers[2];
    CUDA_CHECK(cudaMalloc((void**)&buffers[0], batch_bytes));
    CUDA_CHECK(cudaMalloc((void**)&buffers[1], batch_bytes));
    upload_structuring_element(structuring_element);

    for (int k = 0; k < size; ++k) {
        CUDA_CHECK(cudaMemcpy(buffers[0] + (size_t)k * rows * cols, img[k]->data[0],
                              image_bytes, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const double kernel_start = now_seconds();

    int cur = 0;
    for (int s = 0; s < nstages; ++s) {
        stages[s](buffers[cur], buffers[1 - cur], rows, cols, se_rows, se_cols, size, cfg);
        cur = 1 - cur;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    last_kernel_seconds = (now_seconds() - kernel_start) / size;

    for (int k = 0; k < size; ++k) {
        CUDA_CHECK(cudaMemcpy(img[k]->data[0], buffers[cur] + (size_t)k * rows * cols,
                              image_bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
}

extern "C" void image_erosion(matrix** img, matrix* structuring_element,
                              int size, cuda_config cfg) {
    static const launch_fn stages[] = { launch_erosion };
    run_pipeline(img, structuring_element, size, cfg, stages, 1);
}

extern "C" void image_opening(matrix** img, matrix* structuring_element,
                              int size, cuda_config cfg) {
    static const launch_fn stages[] = { launch_erosion, launch_dilation };
    run_pipeline(img, structuring_element, size, cfg, stages, 2);
}
