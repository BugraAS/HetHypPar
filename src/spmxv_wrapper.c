#include "deps/spmxv/headers/spmxv.h"
#include "spmxv.h"
#include "spmxv_matrix.h"
#include <common.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// outpartarr can be NULL.
// It will use inpartarr for both parts if you do that.
void retrieveMatrixParts(buMatrix *A, spmxv_const partScheme, int *inpartarr,
                         int *outpartarr) {
  int numProcs;

  FILE *pf;
  int i;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  if (partScheme != PART_BY_ROWS && partScheme != PART_BY_COLUMNS) {
    printf("\nPARTSCHEME=? in 1D\n");
    MPI_Finalize();
    exit(120);
  }

  A->inPart = inpartarr;

  if (outpartarr == NULL)
    A->outPart = inpartarr;
  else
    A->outPart = outpartarr;

  printf("\n\t****\tRead the matrix parts\t****");
  return;
}

// Do NOT free the  CSC matrix since it uses same malloc blocks.
void readMatrixFromCSC(CSC *Ain , buMatrix *Aout) {

  int m, n, nnz;

  int i, ind = 0;

  m = Ain->m;
  n = Ain->n;
  nnz = Ain->nnz;

  Aout->gn = n;
  Aout->gm = m;
  Aout->m = m;
  Aout->n = n;
  Aout->nnz = nnz;
  Aout->store = STORE_BY_COLUMNS;
  Aout->ia = Ain->J;
  Aout->ja = Ain->I;
  Aout->nnz= Ain->nnz;
  Aout->val= Ain->val;
  
  #ifndef NDEBUG
  int confuse = Ain->J[Ain->n];
  int lastnum = Aout->ia[Aout->n -1];
  int lastnum2 = Aout->nnz;
  #endif
  if (Aout->ia[Aout->n] != Aout->nnz) {
    printf("\nWrong number of nonzeros read");
    MPI_Finalize();
    exit(10);
  }

  printf("\n\t****\tTransferred the CSC matrix to buMatrix\t****");
  return;
}

void distributeMatrix(buMatrix *Aout, CSC *Ain, spmxv_const partScheme, int* partArr) {

  int myId;
  int numProcs;

  buMatrix *Aall; /*procs 0 allocates*/
  int *tmp;       /*procs 0 allocates*/
  int i;          /*precs 0 uses*/

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);

  if (myId == 0) {
    int partsz;
    int *partMatrix; /*just an alias*/
    int *partOther;  /*just an alias*/
    buMatrix *mat;   /*just an alias*/

    buMatrix bigA, bigAT; /*reads bigA,
                    remember it is m by n, and stored rowwise.
                    store it columnwise in bigAT*/
    readMatrixFromCSC(Ain,&bigA);
    // REPLACE THIS PART SOMEHOWW
    retrieveMatrixParts(&bigA, partScheme, partArr, NULL);

    Aall = (buMatrix *)malloc(sizeof(buMatrix) * numProcs);
    tmp = (int *)malloc(sizeof(int) * (bigA.m + bigA.n));
    if (partScheme == PART_BY_ROWS)
      switchStore(&bigA, &bigAT);
    /*determine how to distribute matrix.
remember: whole matrix is stored rowwise in file.
                    bigA rowwise stored matrix,
    bigAT columnwise stored matrix.
*/

    if (partScheme == PART_BY_ROWS) {
      partsz = bigA.m;
      partMatrix = bigA.outPart;
      partOther = bigA.inPart;
      mat = &bigAT;
    } else if (partScheme == PART_BY_COLUMNS) {
      partsz = bigA.n;
      partMatrix = bigA.inPart;
      partOther = bigA.outPart;
      mat = &bigA;
    } else {
      printf("\n***\treadMatrix: partScheme is wrong\t***");
      MPI_Finalize();
      exit(1090);
    }

    /*initialization*/
    for (i = 0; i < numProcs; i++) {
      Aall[i].m = Aall[i].n = Aall[i].nnz = 0;
      if (partScheme == PART_BY_COLUMNS)
        Aall[i].store = STORE_BY_COLUMNS;
      else if (partScheme == PART_BY_ROWS)
        Aall[i].store = STORE_BY_ROWS;
      Aall[i].gn = bigA.n;
      Aall[i].gm = bigA.m;
    }
    /*to allocate space compute nnz and size(ia)*/
    for (i = 0; i < partsz; i++) {
      int p = partMatrix[i];
      Aall[p].nnz += mat->ia[i + 1] - mat->ia[i];
      IA_SIZE_INC(&(Aall[p]));
    }

    for (i = 0; i < numProcs; i++) {
      allocMatrix(&Aall[i]);
      /*will re-compute size(ia) and nnz*/
      Aall[i].m = Aall[i].n = Aall[i].nnz = 0;
      Aall[i].ia[0] = 0;
    }

    /*fill up the ia ja and val for each processor*/
    for (i = 0; i < partsz; i++) {
      int p = partMatrix[i];
      int sz = mat->ia[i + 1] - mat->ia[i];
      int pind;
      int jp;
      int jm;

      pind = IA_SIZE(&(Aall[p]));
      jp = Aall[p].ia[pind];
      jm = mat->ia[i];

      //TODO: identify whats wrong here
      //The program crashes on this exact line
      memcpy(&(Aall[p].ja[jp]), &(mat->ja[jm]), sizeof(int) * sz);

      memcpy(&(Aall[p].val[jp]), &(mat->val[jm]), sizeof(double) * sz);


      Aall[p].nnz += sz;
      Aall[p].ia[pind + 1] = Aall[p].nnz;
      IA_SIZE_INC(&(Aall[p]));
    }

    /*compute non-partitioned dimension.
if partitiong was rowwise, compute number of nonzero columns for each
            processor's matrix*/
    for (i = 0; i < numProcs; i++) {
      int szia;
      int inc_amnt = 0;
      int ii;
      szia = IA_SIZE(&(Aall[i]));
      memset(tmp, 0, sizeof(int) * (bigA.m + bigA.n));
      for (ii = 0; ii < szia; ii++) {
        int j;
        int js = Aall[i].ia[ii];
        int je = Aall[i].ia[ii + 1];
        for (j = js; j < je; j++) {
          int ind = Aall[i].ja[j];
          tmp[ind]++;
        }
      }
      for (ii = 0; ii < bigA.m + bigA.n - partsz; ii++) {
        int p = partOther[ii];
        if (p == i)
          tmp[ii]++;
      }
      for (ii = 0; ii < bigA.m + bigA.n - partsz; ii++) {
        if (tmp[ii] != 0)
          inc_amnt++;
      }
      OTHER_SIZE_INC(&(Aall[i]), inc_amnt);
    }

    /*send ia, ja, val etc to each processor*/
    for (i = 0; i < numProcs; i++) {
      int sz[5];

      sz[0] = Aall[i].m;
      sz[1] = Aall[i].n;
      sz[2] = Aall[i].nnz;
      sz[3] = Aall[i].gm;
      sz[4] = Aall[i].gn;

      if (i > 0) {
        MPI_Send(sz, 5, MPI_INT, i, SZDATA, MPI_COMM_WORLD);
        MPI_Send(bigA.inPart, Aall[i].gn, MPI_INT, i, INPART, MPI_COMM_WORLD);
        MPI_Send(bigA.outPart, Aall[i].gm, MPI_INT, i, OUTPART, MPI_COMM_WORLD);
        MPI_Send(Aall[i].ia, IA_SIZE(&(Aall[i])) + 1, MPI_INT, i, IA_DATA,
                 MPI_COMM_WORLD);
        MPI_Send(Aall[i].ja, Aall[i].nnz, MPI_INT, i, JA_DATA, MPI_COMM_WORLD);
        MPI_Send(Aall[i].val, Aall[i].nnz, MPI_DOUBLE, i, VAL_DATA,
                 MPI_COMM_WORLD);
      } else /*copy for itself*/
      {
        Aout->m = Aall[0].m;
        Aout->n = Aall[0].n;
        Aout->nnz = Aall[0].nnz;
        Aout->gm = Aall[0].gm;
        Aout->gn = Aall[0].gn;
        if (partScheme == PART_BY_ROWS)
          Aout->store = STORE_BY_ROWS;
        else if (partScheme == PART_BY_COLUMNS)
          Aout->store = STORE_BY_COLUMNS;

        allocMatrix(Aout);

        memcpy(Aout->inPart, bigA.inPart, sizeof(int) * bigA.n);
        memcpy(Aout->outPart, bigA.outPart, sizeof(int) * bigA.m);
        memcpy(Aout->ia, Aall[0].ia, sizeof(int) * (IA_SIZE(Aout) + 1));
        memcpy(Aout->ja, Aall[0].ja, sizeof(int) * Aout->nnz);
        memcpy(Aout->val, Aall[0].val, sizeof(double) * Aout->nnz);
      }
    }
    freeMatrix(&bigA);
    if (partScheme == PART_BY_COLUMNS)
      freeMatrix(&bigAT);
  } else {
    int sz[5];
    MPI_Status recvStatus;

    MPI_Recv(sz, 5, MPI_INT, 0, SZDATA, MPI_COMM_WORLD, &recvStatus);
    Aout->m = sz[0];
    Aout->n = sz[1];
    Aout->nnz = sz[2];
    Aout->gm = sz[3];
    Aout->gn = sz[4];
    if (partScheme == PART_BY_COLUMNS)
      Aout->store = STORE_BY_COLUMNS;
    else if (partScheme == PART_BY_ROWS)
      Aout->store = STORE_BY_ROWS;
    else if (partScheme == PART_2D) {
    }

    allocMatrix(Aout);

    MPI_Recv(Aout->inPart, Aout->gn, MPI_INT, 0, INPART, MPI_COMM_WORLD, &recvStatus);
    MPI_Recv(Aout->outPart, Aout->gm, MPI_INT, 0, OUTPART, MPI_COMM_WORLD,
             &recvStatus);
    MPI_Recv(Aout->ia, IA_SIZE(Aout) + 1, MPI_INT, 0, IA_DATA, MPI_COMM_WORLD,
             &recvStatus);
    MPI_Recv(Aout->ja, Aout->nnz, MPI_INT, 0, JA_DATA, MPI_COMM_WORLD, &recvStatus);
    MPI_Recv(Aout->val, Aout->nnz, MPI_DOUBLE, 0, VAL_DATA, MPI_COMM_WORLD,
             &recvStatus);
  }

  if (myId == 0) /*cleaning up space*/
  {
    for (i = 0; i < numProcs; i++)
      freeMatrix(&(Aall[i]));
    free(Aall);
    free(tmp);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (myId == 0) {
    printf("\n\t****\tMatrix is distributed\t****");
  }
  return;
}

