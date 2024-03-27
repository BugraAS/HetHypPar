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
  double start ; // start time 
  
  spmxv_const part_scheme = PART_BY_ROWS;  // Bora hocanın kodu   (spmxv) 

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  
  if (argc < 4) {
    fprintf(stderr, "Usage: %s [martix-market-filename] [part-vector-file] [number-of-iterations]\n", argv[0]);
    MPI_Finalize();
    exit(1);
  }

  int numiters = atoi(argv[3]);

  
  initParLib(MPI_COMM_WORLD);

  buMatrix buMat = {0};

  readMatrixMarket(&buMat, argv[1], part_scheme, argv[2], argv[2]);

  MPI_Barrier(MPI_COMM_WORLD) ;
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
  MPI_Barrier(MPI_COMM_WORLD) ; 
  start = MPI_Wtime() ;
  for (int i = 0; i < numiters; i++) {
    for (int jb = 0; jb < b->sz; jb++)
      b->val[jb] = 0.0;
    mxv(&A, x, b, MPI_COMM_WORLD);
  }

  double elapsedTime =  MPI_Wtime()-start ;
  double finalTime = 0.0 ;

  MPI_Reduce(&elapsedTime, &finalTime, 1 , MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) ;
  
  if (myId == 0) 
     printf("%s RANK : %d   MxV : %lf\n",argv[2], myId , finalTime ) ; 

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
