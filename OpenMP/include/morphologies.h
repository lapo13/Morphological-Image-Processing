#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"

// Tempo dell'ultima operazione morfologica, in secondi.
// Per erosione/dilatazione e' il solo loop centrale (padding e cropping esclusi);
// per opening e' il wall-clock dell'intera pipeline diviso per il numero
// di immagini del batch, cioe' un tempo per-immagine.
extern double last_op_seconds;

void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int size);
void image_dilation(matrix** img, matrix* structuring_element, matrix* scratch, int size);
void image_opening(matrix** img, matrix* structuring_element, matrix* scratch, int size);

#endif
