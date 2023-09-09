#pragma once
#include "common.h"
#include "spmxv.h"

/* Aout is the partitioned matrix after the distribution.
 * Ain is the CSC matrix read using our functions. Only defined in Master node.
 * partScheme can bee either PART_BY_ROWS or PART_BY_COLUMNS.
 * partArr is the result of PaToH and is only present in master node.
 */
void distributeMatrix(buMatrix *Aout, CSC *Ain, int partScheme, int* partArr);

/* Converts CSC matrix to buMatrix struct.
 */
void readMatrixFromCSC(CSC *Ain , buMatrix *Aout);

/* Fills out buMatrix partition map using PaToH output.
 */
void retrieveMatrixParts(buMatrix *A, int partScheme, int *inpartarr, int *outpartarr); 
