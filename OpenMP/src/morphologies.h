#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"

// variabile resa globale per memorizzare il tempo dell'ultima operazione morfologica
extern double last_op_seconds;

void image_erosion(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization);
void image_dilation(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization);
void image_opening(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization);
void image_closing(matrix* img, matrix* structuring_element, matrix* scratch, int vectorization);

#endif