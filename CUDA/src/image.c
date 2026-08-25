#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.h"


void load_image(const char* filename, matrix* img) {
     int width, height, channels;
     u_int8_t* data = stbi_load(filename, &width, &height, &channels, 1);
     if (!data) {
          fprintf(stderr, "Failed to load image: %s\n", stbi_failure_reason());
          exit(EXIT_FAILURE);
     }

     allocate_matrix(img, height, width);

     for (int i = 0; i < height; i++) {
          for (int j = 0; j < width; j++) {
               img->data[i][j] = data[i * width + j];
          }
     }
     stbi_image_free(data);
}

void save_image(const char* filename, matrix* img) {
     if (!stbi_write_png(filename, img->cols, img->rows, 1, img->data[0], img->cols)) {
          fprintf(stderr, "Failed to save image: %s\n", filename);
          exit(EXIT_FAILURE);
     }
}

void paste_image(matrix* dest, matrix* src, int row_offset, int col_offset) {
    if (dest->rows < src->rows + row_offset || dest->cols < src->cols + col_offset) {
        fprintf(stderr, "Error: Source image does not fit within destination image at the specified offset.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i + row_offset][j + col_offset] = src->data[i][j];
        }
    }
}

void build_mosaic_image(matrix* dest, matrix* tile_buffer, const char** tile_paths, int grid_rows, int grid_cols) {
    load_image(tile_paths[0], tile_buffer);
    int tile_rows = tile_buffer->rows;
    int tile_cols = tile_buffer->cols;

    allocate_matrix(dest, tile_rows * grid_rows, tile_cols * grid_cols);

    paste_image(dest, tile_buffer, 0, 0);

    free_matrix(tile_buffer);

    for (int t = 1; t < grid_rows * grid_cols; t++) {
        int gi = t / grid_cols;
        int gj = t % grid_cols;
        load_image(tile_paths[t], tile_buffer);
        paste_image(dest, tile_buffer, gi * tile_rows, gj * tile_cols);
        free_matrix(tile_buffer);
    }
}