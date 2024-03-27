
#include "common.h"
#include "deps/spmxv/headers/spmxv.h"
#include "mpi.h"

#include "partition.h"
#include "spmxv.h"
#include "spmxv_wrapper.h"
#include "stdio.h"
#include <stddef.h>
#include <string.h> 
#include <stdlib.h>

int main(int argc, char *argv[]) {

  if (argc < 8) {
    
    fprintf(stderr, "Usage: %s [martix-market-filename] [number-of-parts] [benchmark-file] [imbalence-percentage] [seed] [output-file-name] [result-file-name]\n",
            argv[0]);
    exit(1);
  }
   
  int numparts = atoi(argv[2]);

  CSC cscmatrix = {0};

  int *partvec = NULL;

  double start = MPI_Wtime();
  cscmatrix = ReadSparseMatrix(argv[1]);
  printf("Reading File : %lf\n", MPI_Wtime() - start);

  start = MPI_Wtime();
  partvec = CalcPartVec(numparts, &cscmatrix,argv[3], atoi(argv[4])/100. , atoi(argv[5]), argv[7]);
  printf("part vector : %lf\n", MPI_Wtime() - start);

  
  FILE *fptr = fopen( argv[6], "w");  
  for (size_t i = 0; i < cscmatrix.n; i++)
    fprintf(fptr, "%d\n", partvec[i]);
  fclose(fptr);

//******************************************************************
  int vec[numparts];
  for (int i = 0; i < numparts; i++)
    vec[i] = 0; // Ask?

  for (int i = 0; i < cscmatrix.m; i++) {
    vec[partvec[i]]++;
  }
  FILE *fptr2 = fopen(argv[7],"a") ;
  if (fptr2 == NULL ) {
    printf("Result file cannot open from main_patoh.c \n") ;
    exit(-1) ;
  }
  else {
  fprintf(fptr2,"\nPart Vector : \n") ;
  for (int i = 0; i < numparts; i++) {
    printf("i : %d ", vec[i]);
    fprintf(fptr2," [%d] : %d",i,vec[i]) ;
  }
  }
  printf("\n");
//******************************************************************

  freeSparseMatrix(&cscmatrix) ; 
  free(partvec) ;
  
  return 0;
}
