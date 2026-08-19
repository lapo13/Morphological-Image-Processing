#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"

// Necessario per 
#ifdef __cplusplus
extern "C" {
#endif

// tempo (in secondi) dell'ultima operazione morfologica eseguita
extern double last_op_seconds;

void image_erosion(matrix* img, matrix* structuring_element, matrix* scratch);
void image_dilation(matrix* img, matrix* structuring_element, matrix* scratch);
void image_opening(matrix* img, matrix* structuring_element, matrix* scratch);
void image_closing(matrix* img, matrix* structuring_element, matrix* scratch);

#ifdef __cplusplus
}
#endif

#endif
