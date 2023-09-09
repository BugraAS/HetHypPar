#pragma once

#include <common.h>

CSC ReadSparseMatrix(char *fname);
int *CalcPartVec(int nparts,  CSC *cscmatrix);

// Assume the COO is Column Sorted.
//Note that this function allocates memory.
//It can be freed using freeSparseMatrix.
CSC COO_to_CSC(COO *in);