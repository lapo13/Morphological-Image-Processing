#ifndef MORPHOLOGIES_H
#define MORPHOLOGIES_H

#include "matrix.h"

// Necessario per 
#ifdef __cplusplus
extern "C" {
#endif

// Tempo dell'ultima operazione morfologica, in secondi: wall-clock di transfer e
// kernel sull'intero batch, diviso per il numero di immagini. Restano fuori le
// allocazioni device. Come in OpenMP e' un tempo per-immagine.
extern double last_op_seconds;

// scratch non e' usato: senza padding il bounds check dentro al kernel fa lo
// stesso lavoro. Resta nella firma per allinearsi all'API OpenMP.
void image_erosion(matrix** img, matrix* structuring_element, matrix* scratch, int size);
void image_opening(matrix** img, matrix* structuring_element, matrix* scratch, int size);

#ifdef __cplusplus
}
#endif

#endif
