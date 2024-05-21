#include "partition.h"
#include "split.h"
#include "cuSparse.h"
#include "stdio.h"
#include <stddef.h>
#include <string.h> 
#include <stdlib.h>
#include "common.h"

#include <cuda_runtime_api.h>    // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h> 

#include <ctype.h>

#include "deps/mmio/headers/mmio.h"
#include "util.h"


int* readPartVect(char* fileName, int size) ;

int main(int argc, char* argv[]) {  // matrix file ve part vector 

    if (argc < 3) {
        fprintf(stderr,"Usage: %s [martix-market-filename] [parts-vector]\n", argv[0]) ;
        EXIT_FAILURE;
    }

//-------------------------------------------------------------------------------------------------------
    
    CSC cmatrix = ReadSparseMatrix(argv[1]) ; 
    CSR rmatrix = CSC_to_CSR(&cmatrix) ;
    
//-------------------------------------------------------------------------------------------------------
    
    int* part_vec = readPartVect(argv[2], cmatrix.m) ;
   
//-------------------------------------------------------------------------------------------------------
// Check integrity of rmatrix
#ifndef NDEBUG
do{
    double sum = 0;
    for (size_t i = 0; i < cmatrix.nnz; i++)
    {
        sum += cmatrix.val[i];
    }
    DEBUGLOG("Sum of all values in matrix is %lf", sum);
}while(0);
#endif // !NDEBUG
//-------------------------------------------------------------------------------------------------------

    // SPLIT_CSR* splits =  cleanSplit( rmatrix, part_vec) ;
    SPLIT_CSR* splits =  sparseSplit( rmatrix, part_vec) ;
    SPLIT_CSR gpu_m = splits[0], cpu_m = splits[1] ; 
    
//-------------------------------------------------------------------------------------------------------

    double result = spmv_gpu(gpu_m) ;
    printf("spmv_gpu :  %lf \n",result) ; 

//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------
    freeSparseMatrix(&cmatrix) ;
    freeSparseMatrix(&rmatrix) ;
    freeSplit_CSR(&gpu_m) ;
    freeSplit_CSR(&cpu_m) ;
    free(part_vec) ; 

    return 0 ;
}
int* readPartVect(char* fileName, int size) {

    int* part_vec = (int*)malloc(sizeof(int) * size) ;
    if (part_vec == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
     }

     FILE* f = fopen(fileName,"r") ;
     if (f == NULL) {
        fprintf(stderr,"Part Vector File cannot be opened ! ") ;
        free(part_vec);
        return NULL ;
     }
    
    int i=0;
    while( i<size && fscanf(f, "%d",&part_vec[i++]) == 1 );
        
     
    if (i < size && !feof(f)) 
        fprintf(stderr, "Error reading file\n");
    

    fclose(f);

    return part_vec ;    
}
