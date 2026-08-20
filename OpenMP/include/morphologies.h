#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"

// Tempo dell'ultima operazione morfologica, in secondi.
// Per tutte le operazioni e' il wall-clock dell'intero batch, padding e cropping
// inclusi, diviso per il numero di immagini: un tempo per-immagine, quindi
// confrontabile fra operazioni diverse.
extern double last_op_seconds;

void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int size);
void image_dilation(matrix** img, matrix* structuring_element, matrix* scratch, int size);
void image_opening(matrix** img, matrix* structuring_element, matrix* scratch, int size);

#endif
