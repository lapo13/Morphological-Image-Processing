#ifndef IMAGE_H
#define IMAGE_H

#include "matrix.h"

void load_image(const char* filename, matrix* img);
void save_image(const char* filename, matrix* img);
void pad_image(matrix* img, matrix* padded_img, int padding_size, u_int8_t fill_value);
void crop_image(matrix* img, matrix* cropped_img, int crop_size);

#endif
