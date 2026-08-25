#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"
#include "cuda_config.h"

// Incluso da main.c, che e' C: le API vanno esposte con linkage C.
#ifdef __cplusplus
extern "C" {
#endif

extern double last_kernel_seconds;

void image_erosion(matrix** img, matrix* structuring_element, int size, cuda_config cfg);
void image_opening(matrix** img, matrix* structuring_element, int size, cuda_config cfg);

#ifdef __cplusplus
}
#endif

#endif
