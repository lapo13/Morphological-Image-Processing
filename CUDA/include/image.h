#ifndef IMAGE_H
#define IMAGE_H

#include "matrix.h"

void load_image(const char* filename, matrix* img);
void save_image(const char* filename, matrix* img);
void paste_image(matrix* dest, matrix* src, int row_offset, int col_offset);
void build_mosaic_image(matrix* dest, matrix* tile_buffer, const char** tile_paths, int grid_rows, int grid_cols);

#endif
