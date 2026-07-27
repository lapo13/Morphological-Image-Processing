#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"

void image_erosion(matrix* img, matrix* structuring_element, matrix* scratch);
void image_dilation(matrix* img, matrix* structuring_element, matrix* scratch);
void image_opening(matrix* img, matrix* structuring_element, matrix* scratch);
void image_closing(matrix* img, matrix* structuring_element, matrix* scratch);

#endif