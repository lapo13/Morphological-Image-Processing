#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "matrix.h"

// Baseline sequenziale: implementazione in C puro, senza alcuna direttiva
// OpenMP. Il kernel e' identico a quello parallelo (stesso schema pad ->
// finestra scorrevole con maschera -> crop), cosi' l'unica variabile fra le
// versioni e' la parallelizzazione.
//
// sequential.c viene compilato DUE volte con flag diversi (vedi Makefile),
// producendo due varianti dello stesso codice sorgente:
//
//   _scalar : compilato con -fno-vectorize -fno-slp-vectorize.
//             Nessuna forma di parallelismo, nemmeno a livello di dato.
//             E' il T_sequential per lo speedup TOTALE.
//
//   _simd   : compilato con i flag standard, quindi auto-vettorizzato da clang.
//             E' il riferimento per lo speedup da THREADING, l'unico rispetto
//             al quale l'efficienza per-thread resta interpretabile.
//
// Attenzione: "C puro senza pragma" non basta a ottenere codice scalare —
// clang auto-vettorizza comunque a -O2. Serve il flag esplicito, ed e' anche
// il confronto "with/without vectorization flags" richiesto dalla traccia.

extern double last_seq_seconds_scalar;
extern double last_seq_seconds_simd;

void seq_erosion_scalar(matrix** img, matrix* structuring_element, int size);
void seq_dilation_scalar(matrix** img, matrix* structuring_element, int size);
void seq_opening_scalar(matrix** img, matrix* structuring_element, int size);

void seq_erosion_simd(matrix** img, matrix* structuring_element, int size);
void seq_dilation_simd(matrix** img, matrix* structuring_element, int size);
void seq_opening_simd(matrix** img, matrix* structuring_element, int size);

#endif
