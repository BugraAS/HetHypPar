#pragma once

#include "stdlib.h"

// Column sorted Coordinate.
typedef struct {
    int* I; // Row index
    int* J; // Column index
    double* val; // value
    int nnz; // number of non-zeros
    int m; // number of rows
    int n; // number of columns
} COO;

// Compressed Sparse Row.
// .I contains the compressed index.
typedef COO CSR;

// Compressed Sparse Column.
// .J contains the compressed index.
typedef COO CSC;


//Note that this function allocates memory.
//It can be freed using freeSparseMatrix.
static inline CSR CSC_to_CSR(CSC *in){
    CSC out = {
        .I = (int *)malloc(sizeof(int) * (in->m + 1)),
        .J = (int *)malloc(sizeof(int) * in->nnz),
        .val = (double *)malloc(sizeof(double) * in->nnz),
        .nnz = in->nnz,
        .m = in->m,
        .n = in->n
    };
    if (out.I == NULL || out.J == NULL || out.val == NULL){
        return (CSR){0};
    }
    out.I[0] = 0;
    out.I[out.m] = out.nnz;
    for(int i = 0, row = 0, col = 0; i < out.nnz; col++){
        if(col >= in ->n){
            row++;
            out.I[row] = i;
            col = 0;
        }
        for(int j[2] = {in->J[col], in->J[col+1]}; j[0] < j[1] & in->I[j[0]] <= row; j[0]++){
            if(in->I[j[0]]== row){
                out.J[i] = col;
                i++;
                break;
            }
        }
    }
    return out;
}

// returns an array containing the rank of each row.
// NEEDS TO BE FREED.
static inline int* CalcWeights(CSR* in){
    int *out = (int*)malloc(in->m*sizeof(int));
    for(int i = 0; i < in->m; i++){
        out[i]= in->I[i+1] - in->I[i];
    }
    return out;
}

static inline COO SparseTranspose(COO in){
    int *temp = in.I;
    in.I = in.J;
    in.J = temp;
    int swap = in.m;
    in.m = in.n;
    in.n = swap;
    return in;
}

static inline void freeSparseMatrix(COO *in){
    free(in->I);
    free(in->J);
    free(in->val);
}
