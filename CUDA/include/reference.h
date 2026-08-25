#ifndef REFERENCE_H
#define REFERENCE_H

#include "matrix.h"

// Riferimento sequenziale in C puro, usato per validare l'output dei kernel.

void ref_erosion(matrix** img, matrix* structuring_element, int size);
void ref_opening(matrix** img, matrix* structuring_element, int size);

// Confronta due batch pixel per pixel. Restituisce il numero di pixel diversi.
int compare_batches(matrix** a, matrix** b, int size);

#endif
