#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"

// Necessario per 
#ifdef __cplusplus
extern "C" {
#endif

extern double last_op_seconds;


void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int size, int use_shared_memory);
void image_opening(matrix** img, matrix* structuring_element, matrix* scratch, int size, int use_shared_memory);

#ifdef __cplusplus
}
#endif

#endif
