#include "cuSparse.h"
#include "common.h"
#include "util.h"
#include <cuda_runtime_api.h>    // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>            // cusparseSpMV
#include <math.h>
#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

double spmv_gpu(SPLIT_CSR in){

printf("\n\n------------- values (in.loc.val) ---------------------------\n\n") ;
    for (int i=0 ;i< 10; i++) ;
        //printf(" %lf ",in.loc.val[i]) ; 
printf("\n\n--------------values (in.shr.val)------------------------------------------\n\n") ; 

    for (int i=0 ;i< 10; i++) 
        printf(" %lf ",in.shr.val[i]) ; 
printf("\n\n-------------------------------------------------------------\n\n") ;

 CSR csr ;
 for (int t =0 ; t<2 ;t++) {
    
    // -----------------------------------------------------------------------------------------------
    if(t == 0 ) 
       csr = in.loc ;
    else if (t == 1 && in.shr.val != NULL)  
       csr = in.shr ;
    else 
       continue; 
    // -----------------------------------------------------------------------------------------------
    int A_num_rows = csr.m ;
    int A_num_cols = csr.n ;
    int A_nnz = csr.nnz ;
    int* hA_csrOffsets = csr.I ;
    int* hA_columns = csr.J ;
    double* hA_values = csr.val ;

    int* hX = in.locp ; 
    double* hY = (double*)malloc(sizeof(double) * A_num_rows) ;

    printf("A_num_rows : %d\n", A_num_rows) ;
    printf("A_num_cols : %d\n", A_num_cols) ;
    printf("A_nnz : %d\n",A_nnz) ;

    printf("\n\n---------------of set -------------------------\n\n") ;
    for (int i=0 ;i< 10; i++) 
        printf(" %d ",hA_csrOffsets[i]) ; 

    
    printf("\n\n-------------- columns --------------------------\n\n") ;
    for (int i=0 ;i< 10; i++) 
        printf(" %d ",hA_columns[i]) ; 


    printf("\n\n------------- values ---------------------------\n\n") ;
    for (int i=0 ;i< 10; i++) 
        printf(" %lf ",hA_values[i]) ; 


    printf("\n\n----------------------------------------\n\n") ;
    for (int i=0 ;i< 10; i++) 
        printf(" %d ",hX[i]) ; 

    for (int i=0 ;i< A_num_cols; i++)
        hX[i] +=1 ;

    printf("\n\n----------------------------------------\n\n") ;
     for (int i=0 ;i< 10; i++) 
        printf(" %d ",hX[i]) ; 



    double     alpha           = 1.0;
    double     beta            = 0.0;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns,*dX;
    double *dA_values,  *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(double))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,(A_num_rows + 1) * sizeof(int),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(double),
                           cudaMemcpyDeviceToHost) )
    
    printf("\n\n----------------------------------------\n\n") ;
     for (int i=0 ;i< 10; i++) 
        printf(" %lf ",hY[i]) ; 
    
    printf("\n\n----------------------------------------\n\n") ;
    printf("spmv_csr_example test PASSED\n");
    

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
 }

    return 0.0 ;
}
