#pragma once

#include "spmxv.h"

/*define message tags*/
#define SZDATA 1
#define INPART 2
#define OUTPART 3
#define IA_DATA 4
#define JA_DATA 5
#define VAL_DATA 6

#define MAXLINE 63000

typedef struct {
  int *rows;
  int *cols;
  double *vals;
  int sz;
} coord;

// Can be replaced safely with my CSR struct.
void readMatrixFromFile(buMatrix *A, char *fname);

void readMatrixParts(buMatrix *A, char *fname, int partScheme,
                     char *inpartFileName, char *outpartFileName);

void writeMatrix(buMatrix *A, char *fname);
void switchStore(buMatrix *A, buMatrix *AT);
void readCoordsPartVec(int *nnzPartVec, int nnz, char *fnameNnz, int numProcs,
                       int partScheme);
void readMatrixPartsV(int *outPart, int gloRowCnt, int *inPart, int gloColCnt,
                      char *fnameIn, char *fnameOut, int numProcs,
                      int partScheme);


