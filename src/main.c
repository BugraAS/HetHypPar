#include "common.h"
#include "deps/spmxv/headers/spmxv.h"
#include "mpi.h"
#include "partition.h"
#include "spmxv.h"
#include "spmxv_wrapper.h"
#include "stdio.h"
#include <stddef.h>
#include <stdlib.h>

#define WRITE_PARTVEC 0 

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int myId;
  int numProcs;

  spmxv_const part_scheme = PART_BY_ROWS;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);

  if (argc < 4) {
    fprintf(stderr, "Usage: %s [martix-market-filename] [number-of-parts] [number-of-iterations]\n",
            argv[0]);
    MPI_Finalize();
    exit(1);
  }
  int numparts = atoi(argv[2]);
  int numiters = atoi(argv[3]);
  CSC cscmatrix = {0};
  int *partvec = NULL;
  if (myId == 0) {
    cscmatrix = ReadSparseMatrix(argv[1]);
    partvec = CalcPartVec(numparts, &cscmatrix);
  }
 
  initParLib(MPI_COMM_WORLD);

  buMatrix buMat = {0};

  if(!WRITE_PARTVEC)
    distributeMatrix(&buMat, &cscmatrix, part_scheme, partvec);
  else{
    if(myId==0){
     FILE *fptr = fopen("partvec.txt", "w");
      for(size_t i = 0; i < cscmatrix.n; i++)
        fprintf(fptr, "%d\n", partvec[i]);
      fclose(fptr);
    }
    readMatrix(&buMat, "matrices/butub1000.mtx", part_scheme, "partvec.txt", "partvec.txt");
  }

  buMatrix loc = {0}, cpl = {0};
  comm *in = allocComm(), *out = allocComm();
  setupMisG(part_scheme, &buMat, &loc, &cpl, in, out, MPI_COMM_WORLD);

  parMatrix A = {
    .loc=&loc,
    .cpl=&cpl,
    .in=in,
    .out=out,
    .scheme=part_scheme
  };
  buVector *x, *b = (buVector *)malloc(sizeof(buVector)), *y, *bhat;
  b->sz=0;

  /// Create the x vector to use in Ax = b
  if (part_scheme != PART_BY_COLUMNS) {
    if (A.loc->n - A.in->recv->all[numProcs]) {
      x = allocVector(A.loc->n - A.in->recv->all[numProcs]);
      for (int i = 0; i < x->sz; i++)
        x->val[i] = myId + 1;
    } else {
      printf("\nSomething wrong");
      MPI_Finalize();
      return 0;
    }
  } else {
    x = allocVector(A.loc->n);
    for (int i = 0; i < x->sz; i++)
      x->val[i] = myId + 1;
  }

  for (int i = 0; i < numiters; i++) {
    for (int jb = 0; jb < b->sz; jb++)
      b->val[jb] = 0.0;
    mxv(&A, x, b, MPI_COMM_WORLD);
  }

  freeMatrix(&buMat);
  freeMatrix(&loc);
  freeMatrix(&cpl);
  freeVector(x);
  freeVector(b);
  freeComm(out);
  freeComm(in);

  quitParLib(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
