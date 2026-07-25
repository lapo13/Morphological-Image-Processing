#include <stdio.h>
#include <stdlib.h>
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

void pad_image(matrix* img, int padding_size, u_int8_t fill_value) {
    matrix padded_img;
    int new_rows = img->rows + 2 * padding_size;
    int new_cols = img->cols + 2 * padding_size;
    allocate_matrix(&padded_img, new_rows, new_cols);

    memset(padded_img.data[0], fill_value, new_rows * new_cols);

    for (int i = 0; i < img->rows; i++) {
        for (int j = 0; j < img->cols; j++) {
            padded_img.data[i + padding_size][j + padding_size] = img->data[i][j];
        }
    }
    free_matrix(img);
    img->rows = new_rows;
    img->cols = new_cols;
    img->data = padded_img.data;
}

void crop_image(matrix* img, int crop_size) {
    matrix cropped_img;
    int new_rows = img->rows - 2 * crop_size;
    int new_cols = img->cols - 2 * crop_size;
    allocate_matrix(&cropped_img, new_rows, new_cols);

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            cropped_img.data[i][j] = img->data[i + crop_size][j + crop_size];
        }
    }
    free_matrix(img);
    img->rows = new_rows;
    img->cols = new_cols;
    img->data = cropped_img.data;
}
