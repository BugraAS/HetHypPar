#include "spmxv.h"
#include "spmxv_matrix.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
#define DO_MULT_ONE
*/
// #define DEBUG

void copyCreateMatrix(buMatrix *from, buMatrix *to) {
  /*pre:
to is allocated. not the fields but the pointer itself.
*/

  to->m = from->m;
  to->n = from->n;
  to->nnz = from->nnz;
  to->gm = from->gm;
  to->gn = from->gn;
  to->store = from->store;

  allocMatrix(to);

  copyMatrix(from, to);

  return;
}

/**********************************************************************/
void copyMatrix(buMatrix *from, buMatrix *to) {
  /*pre:
1.  integer data is already exist in to
2.  memory is allocated for int*data, and double *data
*/
  int sziaf;
  int sziat;
  sziaf = IA_SIZE(from);
  sziat = IA_SIZE(to);
  /*if(to->store != from->store)
  {
          printf("\ncopy not possible due store scheme");
          MPI_Finalize();
          exit(1009);
  }
  if(sziat != sziaf)
  {
          printf("\ncopy not possible due szia");
          MPI_Finalize();
          exit(1009);
  }*/
  if (to->ia)
    memcpy(to->ia, from->ia, sizeof(int) * (sziaf + 1));
  /*else
  {
          printf("\ncopy not possible to->ia");
          MPI_Finalize();
          exit(1009);
  }*/
  if (to->ja)
    memcpy(to->ja, from->ja, sizeof(int) * from->nnz);
  /*else
  {
          printf("\ncopy not possible to->ja");
          MPI_Finalize();
          exit(1009);
  }*/
  if (to->val)
    memcpy(to->val, from->val, sizeof(double) * from->nnz);
  /*else
  {
          printf("\ncopy not possible to->val");
          MPI_Finalize();
          exit(1009);
  }*/
  if (to->inPart)
    memcpy(to->inPart, from->inPart, sizeof(int) * from->gn);
  /*else
  {
          printf("\ncopy not possible to->inPart");
          MPI_Finalize();
          exit(1009);
  }*/

  if (to->outPart)
    memcpy(to->outPart, from->outPart, sizeof(int) * from->gm);
  /*else
  {
          printf("\ncopy not possible to->outPart");
          MPI_Finalize();
          exit(1009);
  }*/
  return;
}

/**********************************************************************/
void freeMatrix(buMatrix *A) {

  if (A->ia)
    free(A->ia);
  if (A->ja)
    free(A->ja);
  if (A->val)
    free(A->val);
  if (A->inPart)
    free(A->inPart);
  if (A->outPart && A->inPart != A->outPart)
    free(A->outPart);

  return;
}
/**********************************************************************/
void allocMatrix(buMatrix *A) {
  int szia;

  if (A->store == STORE_BY_ROWS)
    szia = A->m + 1;
  else if (A->store == STORE_BY_COLUMNS)
    szia = A->n + 1;
  /*else
  {
          printf("\n***\tallocMatrix: store is not valid\t***");
          MPI_Finalize();
          exit(1909);
  }*/

  /*if(szia <=0)
  {
          printf("\n***\tallocMatrix: szia is not valid %d\t***", szia);
          MPI_Finalize();
          exit(1919);
  }

  if(A->gm <=0)
  {
          printf("\n***\tallocMatrix: gm is not valid\t***");
          MPI_Finalize();
          exit(1939);
  }
  if(A->gn <=0)
  {
          printf("\n***\tallocMatrix: gn is not valid\t***");
          MPI_Finalize();
          exit(1949);
  }
*/
  A->ia = (int *)malloc(sizeof(int) * szia);
  A->ja = (int *)malloc(sizeof(int) * (A->nnz));
  A->val = (double *)malloc(sizeof(double) * (A->nnz));

  A->inPart = (int *)malloc(sizeof(int) * A->gn);
  A->outPart = (int *)malloc(sizeof(int) * A->gm);

  return;
}
void freeCoords(coord *c) {
  free(c->rows);
  free(c->cols);
  free(c->vals);
  return;
}
void allocateCoords(coord *c) {
  int sz = c->sz;
  c->rows = (int *)malloc(sizeof(int) * sz);
  c->cols = (int *)malloc(sizeof(int) * sz);
  c->vals = (double *)malloc(sizeof(double) * sz);
  return;
}

void readMatrixCoordinates(int **rowIndices, int **colIndices, double **val,
                           int *nnz, int *gm, int *gn, int **rpv, int **cpv,
                           char *fname, char *fnameIn, char *fnameOut,
                           char *fnameNnz, MPI_Comm parentComm,
                           int partScheme) {
  int myId, numProcs;
  MPI_Comm libComm;

  getLibComm(parentComm, &libComm);
  MPI_Comm_rank(libComm, &myId);
  MPI_Comm_size(libComm, &numProcs);

  MPI_Barrier(libComm);

  if (myId == 0) {
    int i;
    int *nnzPartVec;
    int *inPart, *outPart;

    int gloRowCnt = 0, gloColCnt = 0, gloNnzCnt = 0;

    int *allRows;
    int *allCols;
    double *allVals;
    int *insertAt;

    int *rv, *cv, *ar, *ac; /*alias*/
    double *av;             /*alias*/

    coord *allCoords;

    allCoords = (coord *)malloc(sizeof(coord) * numProcs);
    insertAt = (int *)malloc(sizeof(int) * numProcs);
    memset(insertAt, 0, sizeof(int) * numProcs);

    for (i = 0; i < numProcs; i++)
      allCoords[i].sz = 0;

    readMatrixCoordsFromFile(&gloRowCnt, &gloColCnt, &gloNnzCnt, &allRows,
                             &allCols, &allVals, fname);
    nnzPartVec = (int *)malloc(sizeof(int) * gloNnzCnt);
    inPart = (int *)malloc(sizeof(int) * gloColCnt);
    outPart = (int *)malloc(sizeof(int) * gloRowCnt);

    readCoordsPartVec(nnzPartVec, gloNnzCnt, fnameNnz, numProcs, partScheme);

    readMatrixPartsV(outPart, gloRowCnt, inPart, gloColCnt, fnameIn, fnameOut,
                     numProcs, partScheme);

    for (i = 0; i < gloNnzCnt; i++) {
      int p = nnzPartVec[i];
      allCoords[p].sz++;
    }
    for (i = 0; i < numProcs; i++)
      allocateCoords(&allCoords[i]);
    for (i = 0; i < gloNnzCnt; i++) {
      int p = nnzPartVec[i];
      int atp = insertAt[p];
      allCoords[p].rows[atp] = allRows[i];
      allCoords[p].cols[atp] = allCols[i];
      allCoords[p].vals[atp] = allVals[i];
      insertAt[p]++;
    }
    for (i = 1; i < numProcs; i++) {
      int szData[3];
      int sz = allCoords[i].sz;
      szData[0] = gloRowCnt;
      szData[1] = gloColCnt;
      szData[2] = sz;
      MPI_Send(szData, 3, MPI_INT, i, SZDATA, libComm);
      MPI_Send(inPart, gloColCnt, MPI_INT, i, INPART, libComm);
      MPI_Send(outPart, gloRowCnt, MPI_INT, i, OUTPART, libComm);
      MPI_Send(allCoords[i].rows, sz, MPI_INT, i, IA_DATA, libComm);
      MPI_Send(allCoords[i].cols, sz, MPI_INT, i, JA_DATA, libComm);
      MPI_Send(allCoords[i].vals, sz, MPI_DOUBLE, i, VAL_DATA, libComm);
    }
    /*copy for itself*/

    *gm = gloRowCnt;
    *gn = gloColCnt;
    *nnz = allCoords[0].sz;
    rv = *rpv = (int *)malloc(sizeof(int) * gloRowCnt);
    cv = *cpv = (int *)malloc(sizeof(int) * gloColCnt);
    memcpy(cv, inPart, sizeof(int) * gloColCnt);
    memcpy(rv, outPart, sizeof(int) * gloRowCnt);
    ar = *rowIndices = (int *)malloc(sizeof(int) * (*nnz));
    ac = *colIndices = (int *)malloc(sizeof(int) * (*nnz));
    av = *val = (double *)malloc(sizeof(double) * (*nnz));
    memcpy(ar, allCoords[0].rows, sizeof(int) * (*nnz));
    memcpy(ac, allCoords[0].cols, sizeof(int) * (*nnz));
    memcpy(av, allCoords[0].vals, sizeof(double) * (*nnz));
    /*end copy for itself*/

#ifdef DO_MULT_ONE
    {
      double *sums;
      int rr;
      sums = (double *)malloc(sizeof(double) * gloRowCnt);
      for (rr = 0; rr < gloRowCnt; rr++)
        sums[rr] = 0.0;
      for (rr = 0; rr < gloNnzCnt; rr++) {
        int rowrr = allRows[rr];
        int colcc = allCols[rr];
        int partpp = inPart[colcc] + 1;
        sums[rowrr] += (allVals[rr] * partpp);
      }
      printDblVec(sums, gloRowCnt);
      free(sums);
    }
#endif
    free(inPart);
    free(outPart);
    free(nnzPartVec);
    free(allRows);
    free(allCols);
    free(allVals);
    free(insertAt);
    for (i = 0; i < numProcs; i++)
      freeCoords(&allCoords[i]);
    free(allCoords);

  } else /*if myId != 0*/
  {
    int szData[3];
    int *rv, *cv, *ar, *ac;
    double *av;
    MPI_Status recvStatus;

    MPI_Recv(szData, 3, MPI_INT, 0, SZDATA, libComm, &recvStatus);
    *gm = szData[0];
    *gn = szData[1];
    *nnz = szData[2];
    rv = *rpv = (int *)malloc(sizeof(int) * szData[0]);
    cv = *cpv = (int *)malloc(sizeof(int) * szData[1]);
    MPI_Recv(cv, szData[1], MPI_INT, 0, INPART, libComm, &recvStatus);
    MPI_Recv(rv, szData[0], MPI_INT, 0, OUTPART, libComm, &recvStatus);
    ar = *rowIndices = (int *)malloc(sizeof(int) * szData[2]);
    ac = *colIndices = (int *)malloc(sizeof(int) * szData[2]);
    av = *val = (double *)malloc(sizeof(double) * szData[2]);
    MPI_Recv(ar, szData[2], MPI_INT, 0, IA_DATA, libComm, &recvStatus);
    MPI_Recv(ac, szData[2], MPI_INT, 0, JA_DATA, libComm, &recvStatus);
    MPI_Recv(av, szData[2], MPI_DOUBLE, 0, VAL_DATA, libComm, &recvStatus);
  }
  MPI_Barrier(libComm);
  if (myId == 0)
    printf("\t\t***\treadMatrixCoordinates OK\t***\n");

  return;
}

void readMatrix(buMatrix *A, char *fname, int partScheme, char *inpartfname,
                char *outpartfname) {

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

    readMatrixFromFile(&bigA, fname);

    readMatrixParts(&bigA, fname, partScheme, inpartfname, outpartfname);

    Aall = (buMatrix *)malloc(sizeof(buMatrix) * numProcs);
    tmp = (int *)malloc(sizeof(int) * (bigA.m + bigA.n));
    if (partScheme == PART_BY_COLUMNS)
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
      mat = &bigA;
    } else if (partScheme == PART_BY_COLUMNS) {
      partsz = bigA.n;
      partMatrix = bigA.inPart;
      partOther = bigA.outPart;
      mat = &bigAT;
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
        A->m = Aall[0].m;
        A->n = Aall[0].n;
        A->nnz = Aall[0].nnz;
        A->gm = Aall[0].gm;
        A->gn = Aall[0].gn;
        if (partScheme == PART_BY_ROWS)
          A->store = STORE_BY_ROWS;
        else if (partScheme == PART_BY_COLUMNS)
          A->store = STORE_BY_COLUMNS;

        allocMatrix(A);

        memcpy(A->inPart, bigA.inPart, sizeof(int) * bigA.n);
        memcpy(A->outPart, bigA.outPart, sizeof(int) * bigA.m);
        memcpy(A->ia, Aall[0].ia, sizeof(int) * (IA_SIZE(A) + 1));
        memcpy(A->ja, Aall[0].ja, sizeof(int) * A->nnz);
        memcpy(A->val, Aall[0].val, sizeof(double) * A->nnz);
      }
    }
    freeMatrix(&bigA);
    if (partScheme == PART_BY_COLUMNS)
      freeMatrix(&bigAT);
  } else {
    int sz[5];
    MPI_Status recvStatus;

    MPI_Recv(sz, 5, MPI_INT, 0, SZDATA, MPI_COMM_WORLD, &recvStatus);
    A->m = sz[0];
    A->n = sz[1];
    A->nnz = sz[2];
    A->gm = sz[3];
    A->gn = sz[4];
    if (partScheme == PART_BY_COLUMNS)
      A->store = STORE_BY_COLUMNS;
    else if (partScheme == PART_BY_ROWS)
      A->store = STORE_BY_ROWS;
    else if (partScheme == PART_2D) {
    }

    allocMatrix(A);

    MPI_Recv(A->inPart, A->gn, MPI_INT, 0, INPART, MPI_COMM_WORLD, &recvStatus);
    MPI_Recv(A->outPart, A->gm, MPI_INT, 0, OUTPART, MPI_COMM_WORLD,
             &recvStatus);
    MPI_Recv(A->ia, IA_SIZE(A) + 1, MPI_INT, 0, IA_DATA, MPI_COMM_WORLD,
             &recvStatus);
    MPI_Recv(A->ja, A->nnz, MPI_INT, 0, JA_DATA, MPI_COMM_WORLD, &recvStatus);
    MPI_Recv(A->val, A->nnz, MPI_DOUBLE, 0, VAL_DATA, MPI_COMM_WORLD,
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

/**********************************************************************/
void printMatrix(buMatrix *A) {
  int i;

  printf("\n******************************");
  printf("\nA(m,n,nnz) = %d %d %d", A->m, A->n, A->nnz);
  printf("\nA->store = %s size %d", A->store == STORE_BY_ROWS ? "rows" : "cols",
         IA_SIZE(A));
  printf("\nA(gm, gn) = %d %d", A->gm, A->gn);

  for (i = 0; i < IA_SIZE(A); i++) {
    int j, js = A->ia[i], je = A->ia[i + 1];
    for (j = js; j < je; j++) {
      printf("\n(%d,%d)=%lf", i, A->ja[j], A->val[j]);
    }
  }
  printf("\nprinting inpart");
  printIntVec(A->inPart, A->gn);
  printf("\nprinting outpart");
  printIntVec(A->outPart, A->gm);
  printf("\n******************************");
  return;
}

void readCoordsPartVec(int *nnzPartVec, int nnz, char *fnameNnz, int numProcs,
                       int partScheme) {
  FILE *ifp;
  int i;
  if ((partScheme != PART_2D) && (partScheme != PART_CHK))

  {
    printf("\npartScheme =%d is not available for 2d part", partScheme);
    MPI_Finalize();
    exit(120);
  }
  ifp = fopen(fnameNnz, "r");
  if (!ifp) {
    printf("\nCan not open %s", fnameNnz);
    MPI_Finalize();
    exit(120);
  }

  for (i = 0; i < nnz; i++)
    fscanf(ifp, "%d", &nnzPartVec[i]);

  fclose(ifp);
  printf("\t\t***\tCoords partvec is read\t***\n");
  return;
}
void readMatrixPartsV(int *outPart, int gloRowCnt, int *inPart, int gloColCnt,
                      char *fnameIn, char *fnameOut, int numProcs,
                      int partScheme) {

  FILE *pf;
  int i;
  if ((partScheme != PART_2D) && (partScheme != PART_CHK)) {
    printf("\npartScheme %d is not ok for 2d", partScheme);
    MPI_Finalize();
    exit(120);
  }
  pf = fopen(fnameIn, "r");
  if (!pf) {
    printf("\nCan not open %s", fnameIn);
    MPI_Finalize();
    exit(120);
  }
  for (i = 0; i < gloColCnt; i++)
    fscanf(pf, "%d", &inPart[i]);
  fclose(pf);

  pf = fopen(fnameOut, "r");
  if (!pf) {
    printf("\nCan not open %s", fnameOut);
    MPI_Finalize();
    exit(110);
  }
  for (i = 0; i < gloRowCnt; i++)
    fscanf(pf, "%d", &outPart[i]);
  fclose(pf);
  printf("\t\t***\trow col partvec is read\t***\n");
  return;
}

/**********************************************************************/
void readMatrixParts(buMatrix *A, char *fname, int partScheme,
                     char *inpartfname, char *outpartfname) {
  int numProcs;

  FILE *pf;
  int i;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  if (partScheme != PART_BY_ROWS && partScheme != PART_BY_COLUMNS) {
    printf("\nPARTSCHEME=? in 1D\n");
    MPI_Finalize();
    exit(120);
  }

  pf = fopen(inpartfname, "r");
  if (!pf) {
    printf("\nCan not open %s", inpartfname);
    MPI_Finalize();
    exit(120);
  }

  for (i = 0; i < A->gn; i++)
    fscanf(pf, "%d", &A->inPart[i]);
  fclose(pf);

  pf = fopen(outpartfname, "r");
  if (!pf) {
    printf("\nCan not open %s", outpartfname);
    MPI_Finalize();
    exit(110);
  }
  for (i = 0; i < A->gm; i++)
    fscanf(pf, "%d", &A->outPart[i]);
  fclose(pf);

  printf("\n\t****\tMatrix parts readed\t****");
  return;
}

void readMatrixCoordsFromFile(int *rowCnt, int *colCnt, int *nnzCnt, int **rows,
                              int **cols, double **vals, char *fname) {

  FILE *ifp = fopen(fname, "r");
  int m, n, nnz;

  int *aliasrow;
  int *aliascol;
  double *aliasval;

  char *line;
  int i;

  if (!ifp) {
    printf("\nCan not open %s", fname);
    MPI_Finalize();
    exit(10);
  }

  line = (char *)malloc(sizeof(char) * MAXLINE);

  do {
    fgets(line, MAXLINE, ifp);
  } while (line[0] == '%');

  sscanf(line, "%d %d %d", &m, &n, &nnz);
  *rowCnt = m;
  *colCnt = n;
  *nnzCnt = nnz;
  aliasrow = *rows = (int *)malloc(sizeof(int) * nnz);
  aliascol = *cols = (int *)malloc(sizeof(int) * nnz);
  aliasval = *vals = (double *)malloc(sizeof(double) * nnz);

  for (i = 0; i < nnz; i++) {
    int r, c;
    double v;
    fscanf(ifp, "%d %d %lf", &r, &c, &v);
    if (r == 0) {
      printf("\nassumed row indices to be >0 but =0 in file %s", fname);
      MPI_Finalize();
      exit(11);
    }
    if (c == 0) {
      printf("\nassumed col indices to be >0 but =0 in file %s", fname);
      MPI_Finalize();
      exit(11);
    }
    aliasrow[i] = r - 1;
    aliascol[i] = c - 1;
    aliasval[i] = v;
  }

  fclose(ifp);
  free(line);

  printf("\n\t\t***\tMatrix File readed. rows: %d cols: %d nonzeros: %d\t***\n",
         m, n, nnz);
  return;
}

/**********************************************************************/
void readMatrixFromFile(buMatrix *A, char *fname) {

  FILE *ifp = fopen(fname, "r");
  int m, n, nnz;

  int i, ind = 0;

  int *rowindices, *colindices;
  double *vals;

  if (!ifp) {
    printf("\nCan not open %s", fname);
    MPI_Finalize();
    exit(10);
  }

  fscanf(ifp, "%d %d %d\n", &m, &n, &nnz);

  A->gn = n;
  A->gm = m;
  A->m = m;
  A->n = n;
  A->nnz = nnz;
  A->store = STORE_BY_ROWS;

  allocMatrix(A);
  rowindices = (int *)malloc(sizeof(int) * (nnz + 1));
  colindices = (int *)malloc(sizeof(int) * (nnz + 1));
  vals = (double *)malloc(sizeof(double) * (nnz + 1));

  for (i = 0; i <= m; i++)
    A->ia[i] = 0;

  //Store matrix as unordered COO and store indices for CSR
  for (i = 0; i < nnz; i++) {
    fscanf(ifp, "%d %d %lf\n", &(rowindices[i]), &(colindices[i]), &(vals[i]));

    rowindices[i]--;
    A->ia[rowindices[i]]++;
    colindices[i]--;
  }
  for (i = 1; i <= m; i++)
    A->ia[i] += A->ia[i - 1];

  // fills out the uncompressed column indices
  // this has no guarantee that the matrix is sorted
  // i think it is relying on the fact column are sorted by default in the file
  for (i = nnz - 1; i >= 0; i--) {
    int r = rowindices[i], c = colindices[i];
    double v = vals[i];
    int off = --A->ia[r];
    A->ja[off] = c;
    A->val[off] = v;
  }

  //inadequate check
  if (A->ia[0] != 0) {
    fclose(ifp);
    printf("\nWrong number of nonzeros read");
    MPI_Finalize();
    exit(10);
  }

  fclose(ifp);
  /*printMatrix(A);*/

  free(rowindices);
  free(colindices);
  free(vals);
  printf("\n\t****\tMatrix File readed\t****");
  return;
}
/**********************************************************************/
void writeMatrix(buMatrix *A, char *fname) {

  FILE *ifp;
  int i;

  int ialmt;
  if (fname)
    ifp = fopen(fname, "w");
  else
    ifp = stdout;

  if (A->store == STORE_BY_ROWS)
    ialmt = A->m;
  else if (A->store == STORE_BY_COLUMNS)
    ialmt = A->n;
  else {
  }
  fprintf(ifp, "%d %d %d %d %d\n", 1, A->n + A->m - ialmt, ialmt, A->nnz, 0);
  for (i = 0; i < ialmt; i++) {
    int j, js = A->ia[i], je = A->ia[i + 1];
    for (j = js; j < je; j++) {
      fprintf(ifp, " %d", A->ja[j] + 1);
    }
    fprintf(ifp, "\n");
  }
  for (i = 0; i < A->nnz; i++) {
    fprintf(ifp, " %lf", A->val[i]);
  }
}

/************************************************************************
 *  given an m by n matrix having nnz nonzeros stored in csr (csc),     *
 *             convert it into m by n matrix with nnz nonzeros stored in*
 *             csc (csr) format.                                        *
 *----------------------------------------------------------------------*/

void switchStore(buMatrix *A, buMatrix *AT) {
  int *ia, *tia, *ja, *tja;
  double *aval, *atval;
  int ialmt, tialmt;
  int n = A->n, m = A->m;
  int i;

  int cntNnz;

  if (A->store == STORE_BY_COLUMNS)
    AT->store = STORE_BY_ROWS;
  else if (A->store == STORE_BY_ROWS)
    AT->store = STORE_BY_COLUMNS;
  else
    printf("specify storage\n");
  AT->m = A->m;
  AT->n = A->n;
  AT->nnz = A->nnz;
  AT->gn = A->gn;
  AT->gm = A->gm;

  if (AT->store == STORE_BY_ROWS) {
    ialmt = n;
    tialmt = m;
  } else if (AT->store == STORE_BY_COLUMNS) {
    ialmt = m;
    tialmt = n;
  } else {
    printf("specify storage 2 \n");
    exit(1);
  }

  allocMatrix(AT);

  if (!A->inPart) {
    printf("\nswitch is not possible due A->inPart");
    MPI_Finalize();
    exit(189);
  }
  if (!A->outPart) {
    printf("\nswitch is not possible due A->outPart");
    MPI_Finalize();
    exit(189);
  }

  if (AT->inPart)
    memcpy(AT->inPart, A->inPart, sizeof(int) * A->gn);
  else {
    printf("\nswitch is not possible due AT->inPart");
    MPI_Finalize();
    exit(189);
  }
  if (AT->outPart)
    memcpy(AT->outPart, A->outPart, sizeof(int) * A->gm);
  else {
    printf("\nswitch is not possible due AT->outPart");
    MPI_Finalize();
    exit(189);
  }

  ia = A->ia;
  ja = A->ja;
  aval = A->val;

  tia = AT->ia;
  tja = AT->ja;
  atval = AT->val;

  memset(tia, 0, sizeof(int) * (tialmt + 1));
  cntNnz = 0;
  for (i = ialmt - 1; i >= 0; i--) {
    int j, js = ia[i], je = ia[i + 1];

    for (j = js; j < je; j++) {
      if (ja[j] >= tialmt)
        printf("\t***\t***!!!! stupid\n");
      if (ja[j] < 0)
        printf("\t***\t***^^^ buggy\n");
      cntNnz++;
      tia[ja[j]]++; /*count occurences*/
    }
  }

  for (i = 1; i <= tialmt; i++)
    tia[i] += tia[i - 1];
#ifdef DEBUG
  if (tia[tialmt] != AT->nnz) {
    printf("\nOlmadi transpose %d vs %d--%d", tia[tialmt], AT->nnz, cntNnz);
    MPI_Finalize();
    exit(100);
  }
#endif
  for (i = 0; i < ialmt; i++) {
    int j, js = ia[i], je = ia[i + 1];
    for (j = js; j < je; j++) {
      int t = ja[j];
      double val = aval[j];
      int ind = --tia[t];
#ifdef DEBUG
      if (ind < 0) {
        printf("\ntranpose: ind<0");
        MPI_Finalize();
        exit(1078);
      }

#endif
      tja[ind] = i;
      atval[ind] = val;
    }
  }
  return;
}
