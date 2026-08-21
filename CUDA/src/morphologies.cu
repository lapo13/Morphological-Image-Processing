#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cuda_runtime.h>

#include "morphologies.h"
#include "image.h"

double last_op_seconds = 0.0;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// I launch sono asincroni: senza questo controllo un fallimento (tipicamente
// shared memory richiesta oltre il limite del blocco) passa silenzioso e si
// manifesta come risultato sbagliato.
static void check_launch(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch fallito -> %s\n", what, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Le device properties non cambiano durante l'esecuzione: interrogarle a ogni
// launch significherebbe metterle dentro la regione cronometrata, due volte per
// ogni opening. Si leggono una volta sola.
static const cudaDeviceProp& device_properties(void) {
    static cudaDeviceProp prop;
    static bool loaded = false;
    if (!loaded) {
        cudaGetDeviceProperties(&prop, 0);
        loaded = true;
    }
    return prop;
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

    cudaMemcpyToSymbol(c_se, &se, sizeof(StructuringElement));
}

// Fallback senza shared memory: ogni thread legge la propria finestra direttamente
// dalla global. Prende il structuring element da c_se come la variante con shared,
// cosi' fra i due l'unica differenza e' l'accesso ai pixel dell'immagine.
__global__
    static void ErosionKernel(const u_int8_t* d_img, u_int8_t* d_scratch, int HEIGHT, int WIDTH) {
        int se_rows = c_se.rows;
        int se_cols = c_se.cols;
        int se_radius_y = c_se.radius_y;
        int se_radius_x = c_se.radius_x;

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int chunk = blockIdx.z; // Get the index of the image in the array
        if (row >= HEIGHT || col >= WIDTH) return; // Check if the thread is within the image bounds

        size_t base = (size_t)chunk * HEIGHT * WIDTH; // Offset of this image inside the batch
        u_int8_t min_value = 255;

        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                int img_row = row + i - se_radius_y;
                int img_col = col + j - se_radius_x;

                if (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH) {
                    if (c_se.values[i * se_cols + j] == 1) {
                        min_value = min(min_value, d_img[base + (size_t)img_row * WIDTH + img_col]);
                    }
                }
            }
        }
        d_scratch[base + (size_t)row * WIDTH + col] = min_value;
    }

__global__
    static void SharedErosionKernel(const u_int8_t* d_img, u_int8_t* d_scratch, int HEIGHT, int WIDTH) {
        extern __shared__ u_int8_t shared_pixels[];

        int se_rows = c_se.rows;
        int se_cols = c_se.cols;
        int se_radius_y = c_se.radius_y;
        int se_radius_x = c_se.radius_x;

        int tx = threadIdx.x;
        int row = blockIdx.y; // un blocco per riga dell'immagine
        int col = blockIdx.x * blockDim.x + tx;
        int chunk = blockIdx.z; // Get the index of the image in the array

        size_t base = (size_t)chunk * HEIGHT * WIDTH; // Offset of this image inside the batch
        int tile_cols = blockDim.x + 2 * se_radius_x; // stride di una riga del tile

        for (int i = 0; i < se_rows; ++i) {
            int img_row = row + i - se_radius_y;
            for (int s = tx; s < tile_cols; s += blockDim.x) {
                int img_col = blockIdx.x * blockDim.x + s - se_radius_x;
                shared_pixels[i * tile_cols + s] =
                    (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH)
                    ? d_img[base + (size_t)img_row * WIDTH + img_col]
                    : 255;
            }
        }
        __syncthreads();

        if (col >= WIDTH) return; // Check if the thread is within the image bounds

        u_int8_t min_value = 255;
        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                if (c_se.values[i * se_cols + j] == 1) {
                    min_value = min(min_value, shared_pixels[i * tile_cols + tx + j]);
                }
            }
        }
        d_scratch[base + (size_t)row * WIDTH + col] = min_value;
    }

static void launch_erosion(const u_int8_t* d_in, u_int8_t* d_out, int rows, int cols, int se_rows, int se_cols, int size, int use_shared_memory) {
    const cudaDeviceProp& device_prop = device_properties();

    int threads_x = device_prop.maxThreadsPerBlock;
    size_t shared_bytes = (size_t)se_rows * (threads_x + 2 * (se_cols / 2)) * sizeof(u_int8_t);

    dim3 block(threads_x, 1);
    dim3 grid((cols + threads_x - 1) / threads_x, rows, size);

    if (use_shared_memory) {
        SharedErosionKernel<<<grid, block, shared_bytes>>>(d_in, d_out, rows, cols);
    } else {
        ErosionKernel<<<grid, block>>>(d_in, d_out, rows, cols);
    } 
    check_launch("ErosionKernel");
}


extern "C" void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int size, int use_shared_memory) {
    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;

    // Assuming all images have the same dimensions
    int rows = img[0]->rows;
    int cols = img[0]->cols;

    size_t image_mem_size = (size_t)rows * cols * sizeof(u_int8_t);
    size_t mem_size = image_mem_size * size;

    u_int8_t* d_img; u_int8_t* d_scratch;

    cudaMalloc((void**)&d_img, mem_size);
    cudaMalloc((void**)&d_scratch, mem_size);
    upload_structuring_element(structuring_element);

    double op_t_start = now_seconds();

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(d_img + k * rows * cols, img[k]->data[0], image_mem_size, cudaMemcpyHostToDevice);
    }

    launch_erosion(d_img, d_scratch, rows, cols, se_rows, se_cols, size, use_shared_memory);

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(img[k]->data[0], d_scratch + k * rows * cols, image_mem_size, cudaMemcpyDeviceToHost);
    }

    last_op_seconds = (now_seconds() - op_t_start) / size;

    cudaFree(d_img);
    cudaFree(d_scratch);
}

// Fallback senza shared memory, speculare a ErosionKernel: 0 e' l'elemento neutro
// del massimo, quindi fuori immagine non alza nulla.
__global__
    static void DilationKernel(const u_int8_t* d_img, u_int8_t* d_scratch, int HEIGHT, int WIDTH) {
        int se_rows = c_se.rows;
        int se_cols = c_se.cols;
        int se_radius_y = c_se.radius_y;
        int se_radius_x = c_se.radius_x;

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int chunk = blockIdx.z; // Get the index of the image in the array
        if (row >= HEIGHT || col >= WIDTH) return; // Check if the thread is within the image bounds

        size_t base = (size_t)chunk * HEIGHT * WIDTH; // Offset of this image inside the batch
        u_int8_t max_value = 0;

        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                int img_row = row + i - se_radius_y;
                int img_col = col + j - se_radius_x;

                if (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH) {
                    if (c_se.values[i * se_cols + j] == 1) {
                        max_value = max(max_value, d_img[base + (size_t)img_row * WIDTH + img_col]);
                    }
                }
            }
        }
        d_scratch[base + (size_t)row * WIDTH + col] = max_value;
    }

__global__
    static void SharedDilationKernel(const u_int8_t* d_img, u_int8_t* d_scratch, int HEIGHT, int WIDTH) {
        // extern: la dimensione del tile arriva dal terzo argomento di <<<>>>,
        // perche' dipende da blockDim.x che e' scelto al launch.
        extern __shared__ u_int8_t shared_pixels[];

        int se_rows = c_se.rows;
        int se_cols = c_se.cols;
        int se_radius_y = c_se.radius_y;
        int se_radius_x = c_se.radius_x;

        int tx = threadIdx.x;
        int row = blockIdx.y; // un blocco per riga dell'immagine
        int col = blockIdx.x * blockDim.x + tx;
        int chunk = blockIdx.z; // Get the index of the image in the array

        size_t base = (size_t)chunk * HEIGHT * WIDTH; // Offset of this image inside the batch
        int tile_cols = blockDim.x + 2 * se_radius_x; // stride di una riga del tile

        for (int i = 0; i < se_rows; ++i) {
            int img_row = row + i - se_radius_y;
            for (int s = tx; s < tile_cols; s += blockDim.x) {
                int img_col = blockIdx.x * blockDim.x + s - se_radius_x;
                shared_pixels[i * tile_cols + s] =
                    (img_row >= 0 && img_row < HEIGHT && img_col >= 0 && img_col < WIDTH)
                    ? d_img[base + (size_t)img_row * WIDTH + img_col]
                    : 0;
            }
        }
        __syncthreads();

        if (col >= WIDTH) return; // Check if the thread is within the image bounds

        u_int8_t max_value = 0;
        for (int i = 0; i < se_rows; ++i) {
            for (int j = 0; j < se_cols; ++j) {
                if (c_se.values[i * se_cols + j] == 1) {
                    max_value = max(max_value, shared_pixels[i * tile_cols + tx + j]);
                }
            }
        }
        d_scratch[base + (size_t)row * WIDTH + col] = max_value;
    }

static void launch_dilation(const u_int8_t* d_in, u_int8_t* d_out, int rows, int cols, int se_rows, int se_cols, int size, int use_shared_memory) {
    const cudaDeviceProp& device_prop = device_properties();

    int threads_x = device_prop.maxThreadsPerBlock;
    size_t shared_bytes = (size_t)se_rows * (threads_x + 2 * (se_cols / 2)) * sizeof(u_int8_t);

    dim3 block(threads_x, 1);
    dim3 grid((cols + threads_x - 1) / threads_x, rows, size);

    if (use_shared_memory) {
        SharedDilationKernel<<<grid, block, shared_bytes>>>(d_in, d_out, rows, cols);
    } else {
        DilationKernel<<<grid, block>>>(d_in, d_out, rows, cols);
    }
    check_launch("DilationKernel");
}

// image_opening = erosione seguita da dilatazione. I due kernel girano uno dopo
// l'altro sull'intero batch e il risultato intermedio non lascia mai la GPU: una
// sola coppia di transfer per l'intera operazione.
extern "C" void image_opening(matrix** img, matrix* structuring_element, matrix* scratch, int size, int use_shared_memory) {
    (void)scratch; // niente padding: il bounds check nel kernel fa lo stesso lavoro

    int se_rows = structuring_element->rows;
    int se_cols = structuring_element->cols;

    // Assuming all images have the same dimensions
    int rows = img[0]->rows;
    int cols = img[0]->cols;

    size_t image_mem_size = (size_t)rows * cols * sizeof(u_int8_t);
    size_t mem_size = image_mem_size * size;

    u_int8_t* d_img; u_int8_t* d_intermediate;

    cudaMalloc((void**)&d_img, mem_size);
    cudaMalloc((void**)&d_intermediate, mem_size);
    upload_structuring_element(structuring_element);

    double op_t_start = now_seconds();

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(d_img + k * rows * cols, img[k]->data[0], image_mem_size, cudaMemcpyHostToDevice);
    }

    launch_erosion(d_img, d_intermediate, rows, cols, se_rows, se_cols, size, use_shared_memory);
    launch_dilation(d_intermediate, d_img, rows, cols, se_rows, se_cols, size, use_shared_memory);

    for (int k = 0; k < size; ++k) {
        cudaMemcpy(img[k]->data[0], d_img + k * rows * cols, image_mem_size, cudaMemcpyDeviceToHost);
    }

    last_op_seconds = (now_seconds() - op_t_start) / size;

    cudaFree(d_img);
    cudaFree(d_intermediate);
}
