#include "spmxv.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*#include "ulib.h"*/

// #define DEBUG

/*

#define  PRINT_SIZES
 */

void split(int commStep, buMatrix *A, buMatrix *Aloc, buMatrix *Acpl,
           MPI_Comm libComm);
void split_Rowwise(buMatrix *A, buMatrix *Aloc, buMatrix *Acpl,
                   MPI_Comm libComm);
void split_Colwise(buMatrix *A, buMatrix *Aloc, buMatrix *Acpl,
                   MPI_Comm libComm);

void get_LocalIndices(buMatrix *A, int commStep, int *map, MPI_Comm libComm);
void get_outLocalIndices(buMatrix *A, int *yLocalIndex, MPI_Comm libComm);
void get_inLocalIndices(buMatrix *A, int *xLocalIndex, MPI_Comm libComm);

void set_LocalIndices(buMatrix *Aloc, buMatrix *Acpl, comm *in, comm *out,
                      int *map, int commStep, MPI_Comm libComm);
void set_inLocalIndices(buMatrix *Aloc, buMatrix *Acpl, comm *in, int *map,
                        MPI_Comm libComm);
void set_outLocalIndices(buMatrix *Aloc, buMatrix *Acpl, comm *out, int *map,
                         MPI_Comm libComm);

void informFromWhomToReceive(buMatrix *A, comm *inX, comm *outY, int commStep,
                             MPI_Comm libComm);
void informFromWhomToReceive_Rowwise(buMatrix *A, comm *inX, MPI_Comm libComm);
void informFromWhomToReceive_Columnwise(buMatrix *A, comm *outY,
                                        MPI_Comm libComm);

/******************************* 2D ******************************/
void informToWhomToSend_Y(int *rowIndices, int indCnt, int gloRowCnt,
                          int *yPartVec, comm *outY, MPI_Comm libComm);

void findRowMap(int *rowIndices, int rowIndCnt, int *yPartVec, int ySz,
                MPI_Comm libComm, int *map);

void setRowLocalIndices(int *rowIndices, int rowIndCnt, int *map);
void set_rowIndices_out(comm *out, int *map, MPI_Comm libComm);

void assembleFromCoordinate(int *rowIndices, int *colIndices, double *val,
                            int cnt, buMatrix *A, int storageScheme, int myId,
                            MPI_Comm libComm);

int countDiffIndices(int *lst, int lstSz, int gloSz, int *partVec, int thisP);

#define SEND_ME 7
#define RECV_ME 8

static int *extra_state;
static int libKey = MPI_KEYVAL_INVALID;

typedef struct {
  MPI_Comm libCommHandle;
  int mtrxId;
} libStruct;

comm *allocComm() {
  comm *cm;
  cm = (comm *)malloc(sizeof(comm));
  cm->send = NULL;
  cm->recv = NULL;

  return cm;
}

commHandle *allocCommHandle() {
  commHandle *ch;

  ch = (commHandle *)malloc(sizeof(commHandle));

  ch->val = (double *)NULL;
  ch->ind = (int *)NULL;
  ch->all = (int *)NULL;
  ch->lst = (int *)NULL;
  ch->num = 0;

  return ch;
}
void freeCommHandle(commHandle *ch) {
  if (ch != NULL) {
    if (ch->val != NULL)
      free(ch->val);
    if (ch->ind != NULL)
      free(ch->ind);
    if (ch->all != NULL)
      free(ch->all);
    if (ch->lst != NULL)
      free(ch->lst);
    free(ch);
  }

  return;
}

void freeComm(comm *cm) {
  if (cm != NULL) {
    if (cm->send != NULL)
      freeCommHandle(cm->send);
    if (cm->recv != NULL)
      freeCommHandle(cm->recv);
    free(cm);
  }
  return;
}

/**********************************************************************/
int deleteParLib_internal(
    MPI_Comm parentComm, int keyval, void *attr,
    void *extra_state) { /*automatically called when the parentComm freed*/
  libStruct *libSt = (libStruct *)attr;
  if (libSt == NULL)
    return 1;
  if (libSt->libCommHandle != MPI_COMM_NULL)
    MPI_Comm_free(&libSt->libCommHandle);
  free(attr);
  return 0;
}

void initParLib(MPI_Comm parentComm) {
  MPI_Comm libComm;
  libStruct *libSt;

  if (libKey != MPI_KEYVAL_INVALID) {
    printf("\ninit already done");
    MPI_Finalize();
    exit(12);
  }
  MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, deleteParLib_internal, &libKey,
                         extra_state);

  MPI_Comm_dup(parentComm, &libComm);
  libSt = (libStruct *)malloc(sizeof(libStruct));
  libSt->libCommHandle = libComm;
  libSt->mtrxId = 1;
  MPI_Comm_set_attr(parentComm, libKey, libSt);

  return;
}

void quitParLib(MPI_Comm parentComm) {

  void *attr;
  libStruct *attrSt;
  int flag;
  if (libKey == MPI_KEYVAL_INVALID)
    return;
  MPI_Comm_get_attr(parentComm, libKey, &attr, &flag);
  if (flag) /*library comm exist in parentComm*/
  {
    attrSt = (libStruct *)attr;
    if (attrSt->libCommHandle != MPI_COMM_NULL) {
      MPI_Comm_free(&(attrSt->libCommHandle));
      MPI_Comm_free_keyval(&libKey);
    }
  }
  return;
}

/****************************/
void incrementTagMult(MPI_Comm parentComm) {
  void *attr;
  libStruct *attrSt;
  int flag;
  MPI_Comm_get_attr(parentComm, libKey, &attr, &flag);
  if (flag) {
    attrSt = (libStruct *)attr;
    attrSt->mtrxId += 1;
    if (attrSt->mtrxId == 2048) /*do not let it boost*/
      attrSt->mtrxId = 1;
  } else {
    printf("\n\t\t***************************\n\t\t*Library is not "
           "initialized*\n\t\t****************************\n");
    MPI_Finalize();
    exit(10);
  }

  return;
}

void getTagMult(MPI_Comm parentComm, int *tagMult) {
  void *attr;
  libStruct *attrSt;
  int flag;
  MPI_Comm_get_attr(parentComm, libKey, &attr, &flag);
  if (flag) {
    attrSt = (libStruct *)attr;
    *tagMult = attrSt->mtrxId;
  } else {
    printf("\n\t\t***************************\n\t\t*Library is not "
           "initialized*\n\t\t****************************\n");

    MPI_Finalize();
    exit(10);
  }

  return;
}
void getLibComm(MPI_Comm parentComm, MPI_Comm *libComm) {
  void *attr;
  libStruct *attrSt;
  int flag;
  MPI_Comm_get_attr(parentComm, libKey, &attr, &flag);
  if (flag) {
    attrSt = (libStruct *)attr;
    if (attrSt->libCommHandle != MPI_COMM_NULL)
      *libComm = attrSt->libCommHandle;
    else {
      printf("\n\nError in getLibComm\n");
      MPI_Finalize();
      exit(11);
    }
  } else {
    printf("\n\t\t***************************\n\t\t*Library is not "
           "initialized*\n\t\t****************************\n");

    MPI_Finalize();
    exit(10);
  }

  return;
}

/**********************************************************************/
void setup2D(int *rowIndices, int *colIndices, double *val,
             int indexCnt, /*coordinate format*/
             buMatrix *mtrx, buMatrix *loc, buMatrix *cpl, comm *ins,
             comm *outs, MPI_Comm parentComm) {
  int *map;
  int szmap;
  int i;

  buMatrix *AlocT;
  buMatrix *AcplT;

  int myId;
  buMatrix *A = mtrx;
  buMatrix *Aloc = loc;
  buMatrix *Acpl = cpl;
  comm *in = ins;
  comm *out = outs;

  MPI_Comm libComm;

  getLibComm(parentComm, &libComm);
  MPI_Comm_rank(libComm, &myId);

  szmap = 2 * (mtrx->gn + mtrx->gm);
  map = (int *)malloc(sizeof(int) * szmap);

  MPI_Barrier(libComm);

  /* informToWhomToSendY () using coordinate format*/
  informToWhomToSend_Y(rowIndices, indexCnt, mtrx->gm, mtrx->outPart, out,
                       libComm);

  /*find new indices for rows*/
  findRowMap(rowIndices, indexCnt, mtrx->outPart, mtrx->gm, libComm, map);

  A->m = countDiffIndices(rowIndices, indexCnt, A->gm, A->outPart, myId);

  A->n = countDiffIndices(colIndices, indexCnt, A->gn, A->inPart, myId);

  /*set row indices in coordinate data*/
  setRowLocalIndices(rowIndices, indexCnt, map);

  /*set indices in out */
  set_rowIndices_out(out, map, libComm);

  /*assemble CSR format from coordinate where row indices are from 0 to local
   * row cnt*/

  assembleFromCoordinate(rowIndices, colIndices, val, indexCnt, A,
                         STORE_BY_ROWS, myId, libComm);

  /*use CSR format and partition on X to determine communication <in> */
  informFromWhomToReceive_Rowwise(A, in, libComm);

  /*split the csr matrix into Aloc and Acpl
    where Aloc contains entries corresponding to local cols*/
  split_Rowwise(A, Aloc, Acpl, libComm);

  if (A->nnz != (Aloc->nnz + Acpl->nnz)) {
    printf("I<%d>, A->nnz %d, Aloc->nnz %d, Acpl->nnz %d\n", myId, A->nnz,
           Aloc->nnz, Acpl->nnz);
    MPI_Finalize();
    exit(12);
  }

  /*as in precomm, get x local indices*/

  get_inLocalIndices(A, map, libComm);

  /*as in precomm, set x local indices*/
  set_inLocalIndices(Aloc, Acpl, in, map, libComm);

  /* storeMatrices in CSC format*/
  /* 1. store Aloc*/
  AlocT = (buMatrix *)malloc(sizeof(buMatrix));

  switchStore(Aloc, AlocT);

  freeMatrix(loc);

  loc->store = STORE_BY_COLUMNS;

  allocMatrix(loc);

  copyMatrix(AlocT, loc);

  freeMatrix(AlocT);
  free(AlocT);

  /* 2. store Acpl*/
  AcplT = (buMatrix *)malloc(sizeof(buMatrix));

  switchStore(Acpl, AcplT);

  freeMatrix(cpl);

  cpl->store = STORE_BY_COLUMNS;

  allocMatrix(cpl);

  copyMatrix(AcplT, cpl);

  freeMatrix(AcplT);
  free(AcplT);

  free(map);
  MPI_Barrier(libComm);
  if (myId == 0) {
    // printf("\t\t***\tSetup 2D is OK\t***\n");
  }
  return;
}

void set_rowIndices_out(comm *out, int *map, MPI_Comm libComm) {

  int i;

  int myId;
  MPI_Comm_rank(libComm, &myId);

  /*set in out->recv*/
  for (i = 0; i < out->recv->num; i++) {
    int p = out->recv->lst[i];
    int j, js = out->recv->all[p], je = out->recv->all[p + 1];
    for (j = js; j < je; j++) {
      int ind = out->recv->ind[j];
      int new = map[ind];
      out->recv->ind[j] = new;
    }
  }
  /*set in out->send*/
  for (i = 0; i < out->send->num; i++) {
    int p = out->send->lst[i];
    int j, js = out->send->all[p], je = out->send->all[p + 1];
    for (j = js; j < je; j++) {
      int ind = out->send->ind[j];
      int new = map[ind];
      out->send->ind[j] = new;
    }
  }
  MPI_Barrier(libComm);
  if (myId == 0) {
    // printf("\t\t***\trow indices in out comm are set\t***\n");
  }
  return;
}

void informToWhomToSend_Y(int *rowIndices, int indCnt, int gloRowCnt,
                          int *yPartVec, comm *outY, MPI_Comm libComm)

{
  int numProcs;
  int myId;

  int i;
  int *recvSizes, *sendSizes;
  int *allRows;
  int numRecvs, numSends, volRecvs = 0, volSends = 0;

  MPI_Request *mySendReqs;
  MPI_Request *myRecvReqs;
  MPI_Status *stts;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  recvSizes = (int *)malloc(sizeof(int) * numProcs);
  memset(recvSizes, 0, sizeof(int) * numProcs);
  sendSizes = (int *)malloc(sizeof(int) * numProcs);
  memset(sendSizes, 0, sizeof(int) * numProcs);
  allRows = (int *)malloc(sizeof(int) * gloRowCnt);

  for (i = 0; i < gloRowCnt; i++)
    allRows[i] = -2;

  for (i = 0; i < indCnt; i++) {
    int row = rowIndices[i];
    int p = yPartVec[row];
    if (allRows[row] == -2) {
      allRows[row] = -1;
      sendSizes[p]++;
    }
  }

  MPI_Alltoall(sendSizes, 1, MPI_INT, recvSizes, 1, MPI_INT, libComm);

  numRecvs = numSends = 0;
  recvSizes[myId] = 0;
  sendSizes[myId] = 0;
  for (i = 0; i < numProcs; i++) {
    volRecvs += recvSizes[i];
    volSends += sendSizes[i];
    if (recvSizes[i] != 0)
      numRecvs++;

    if (sendSizes[i] != 0)
      numSends++;
  }

  mySendReqs = (MPI_Request *)malloc(sizeof(MPI_Request) * numProcs);
  myRecvReqs = (MPI_Request *)malloc(sizeof(MPI_Request) * numProcs);
  stts = (MPI_Status *)malloc(sizeof(MPI_Status) * numProcs);

  outY->send = allocCommHandle();
  outY->send->all = (int *)malloc(sizeof(int) * (numProcs + 1));
  outY->send->num = numSends;
  if (numSends > 0)
    outY->send->lst = (int *)malloc(sizeof(int) * numSends);
  if (volSends > 0)
    outY->send->ind = (int *)malloc(sizeof(int) * volSends);

  outY->recv = allocCommHandle();
  outY->recv->all = (int *)malloc(sizeof(int) * (numProcs + 1));
  outY->recv->num = numRecvs;
  if (numRecvs > 0)
    outY->recv->lst = (int *)malloc(sizeof(int) * numRecvs);
  if (volRecvs > 0) {
    outY->recv->ind = (int *)malloc(sizeof(int) * volRecvs);
    outY->recv->val = (double *)malloc(sizeof(double) * volRecvs);
  }

  /*set pointers in send and recv*/
  numRecvs = numSends = volRecvs = volSends = 0;
  outY->recv->all[0] = outY->send->all[0] = 0;
  for (i = 0; i < numProcs; i++) {
    if (recvSizes[i] != 0) {
      outY->recv->lst[numRecvs++] = i;
      volRecvs += recvSizes[i];
    }
    outY->recv->all[i + 1] = volRecvs;
    if (sendSizes[i] != 0) {
      outY->send->lst[numSends++] = i;
      volSends += sendSizes[i];
    }
    outY->send->all[i + 1] = volSends;
    if (numRecvs > outY->recv->num) {
      printf("\nolmuyor in column.. myId=%d, num=%d, recvnum=%d", myId,
             numRecvs, outY->recv->num);
      MPI_Finalize();
      exit(1045);
    }
    if (numSends > outY->send->num) {
      printf("\nolmuyor in column.. asdasd2222");
      MPI_Finalize();
      exit(1045);
    }
  }
  /*fill up indices for send*/
  for (i = 0; i < gloRowCnt; i++) {
    if ((allRows[i] == -1) && (yPartVec[i] != myId)) {
      int p = yPartVec[i];
      int put = outY->send->all[p];
      outY->send->all[p]++;
      outY->send->ind[put] = i;
    }
  }

  /*re-adjust send->all*/
  for (i = numProcs; i > 0; i--) {
    outY->send->all[i] = outY->send->all[i - 1];
  }
  outY->send->all[0] = 0;

  for (i = 0; i < outY->send->num; i++) {
    int p = outY->send->lst[i];
    int sz = outY->send->all[p + 1] - outY->send->all[p];
    int st = outY->send->all[p];
    MPI_Isend(&(outY->send->ind[st]), sz, MPI_INT, p, RECV_ME, libComm,
              &mySendReqs[i]);
  }

  for (i = 0; i < outY->recv->num; i++) {
    int p = outY->recv->lst[i];
    int sz = outY->recv->all[p + 1] - outY->recv->all[p];
    int get = outY->recv->all[p];
    MPI_Irecv(&(outY->recv->ind[get]), sz, MPI_INT, p, RECV_ME, libComm,
              &myRecvReqs[i]);
  }

  if (outY->send->num > 0)
    MPI_Waitall(outY->send->num, mySendReqs, stts);

  if (outY->recv->num > 0)
    MPI_Waitall(outY->recv->num, myRecvReqs, stts);

  MPI_Barrier(libComm);

  if (myId == 0) {
    // printf("\t\t***\tInformed from whom to receive partial Y
    // results\t***\n");
  }
  free(allRows);
  free(recvSizes);
  free(sendSizes);
  free(mySendReqs);
  free(myRecvReqs);
  free(stts);
  return;
}


void setupMisG(int partScheme, buMatrix *mtrx, buMatrix *loc, buMatrix *cpl,
               comm *ins, comm *outs, MPI_Comm parentComm) {
  /*matrix indices are global*/
  
  int i;

  /// specifies if communication is reuired BEFORE or AFTER the multiplication
  int commStep;

  buMatrix *AlocT;
  buMatrix *AcplT;

  int myId;
  buMatrix *A = mtrx;
  buMatrix *Aloc = loc;
  buMatrix *Acpl = cpl;
  comm *in = ins;
  comm *out = outs;

  MPI_Comm libComm;

  getLibComm(parentComm, &libComm);
  MPI_Comm_rank(libComm, &myId);

  int szmap = 2 * (mtrx->gn + mtrx->gm); /// double the size of the global matrix
  int *map = (int *)malloc(sizeof(int) * szmap);

  MPI_Barrier(libComm);

  if (partScheme == PART_BY_ROWS)
    commStep = PRECOMM;
  else if (partScheme == PART_BY_COLUMNS)
    commStep = POSTCOMM;
  else {
    printf("\n\n\t****partSchem %d is not here", partScheme);
    MPI_Finalize();
    exit(1212);
  }

  informFromWhomToReceive(A, in, out, commStep, libComm);

  split(commStep, A, Aloc, Acpl, libComm);

  get_LocalIndices(A, commStep, map, libComm);

  set_LocalIndices(Aloc, Acpl, in, out, map, commStep, libComm);

  if (commStep == PRECOMM) {
    if (Aloc->store == STORE_BY_ROWS) {
      AlocT = (buMatrix *)malloc(sizeof(buMatrix));

      switchStore(Aloc, AlocT);

      freeMatrix(loc);

      loc->store = STORE_BY_COLUMNS;

      allocMatrix(loc);

      copyMatrix(AlocT, loc);

      freeMatrix(AlocT);
      free(AlocT);
    }

    if (Acpl->store == STORE_BY_ROWS) {
      AcplT = (buMatrix *)malloc(sizeof(buMatrix));

      switchStore(Acpl, AcplT);

      freeMatrix(cpl);

      cpl->store = STORE_BY_COLUMNS;

      allocMatrix(cpl);

      copyMatrix(AcplT, cpl);

      freeMatrix(AcplT);
      free(AcplT);
    }
  } else if (commStep == POSTCOMM) {
    if (Aloc->store == STORE_BY_COLUMNS) {
      AlocT = (buMatrix *)malloc(sizeof(buMatrix));

      switchStore(Aloc, AlocT);

      freeMatrix(loc);

      loc->store = STORE_BY_ROWS;

      allocMatrix(loc);

      copyMatrix(AlocT, loc);

      freeMatrix(AlocT);
      free(AlocT);
    }
    if (Acpl->store == STORE_BY_COLUMNS) {
      AcplT = (buMatrix *)malloc(sizeof(buMatrix));

      switchStore(Acpl, AcplT);

      freeMatrix(cpl);

      cpl->store = STORE_BY_ROWS;

      allocMatrix(cpl);

      copyMatrix(AcplT, cpl);

      freeMatrix(AcplT);
      free(AcplT);
    }

  } else if (commStep == PREPOSTCMM) {

  } else {
    printf("\n\n\t****commStep %d is not implemented", commStep);
    MPI_Finalize();
    exit(1212);
  }

  MPI_Barrier(libComm);
  // if(myId==0)
  //    printf("\n\t****\tSet up is OK\t\t****");

  free(map);
  return;
}

/**********************************************************************/
void inverseComm(comm *from, comm *to, MPI_Comm parentComm) {
  /*
  pre: from and to allocated,
       from->send, from->recv allocated.
       to->send == NULL
       to->recv == NULL
  */
  MPI_Comm libComm;

  int myId;
  int numProcs = 0;

  getLibComm(parentComm, &libComm);
  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

#ifdef DEBUG
  if (from == NULL) {
    printf("\n\n\t****from is null in inverseComm");
    MPI_Finalize();
    exit(1212);
  }
  if (to == NULL) {
    printf("\n\n\t****to is null in inverseComm");
    MPI_Finalize();
    exit(1213);
  }
#endif

  if (to->send != NULL) {
    printf("\n\n\t****to->send is not null in inverseComm");
    MPI_Finalize();
    exit(1214);
  }

  if (to->recv != NULL) {
    printf("\n\n\t****to->recv is not null in inverseComm");
    MPI_Finalize();
    exit(1215);
  }

  /*copy from->recv 2 to->send*/
  if (from->recv != NULL) {
    int vol = from->recv->all[numProcs];
    int num = from->recv->num;

    to->send = allocCommHandle();
    to->send->num = num;
    to->send->all = (int *)malloc(sizeof(int) * (numProcs + 1));
    memcpy(to->send->all, from->recv->all, sizeof(int) * (numProcs + 1));

    if (num > 0) {
      to->send->lst = (int *)malloc(sizeof(int) * num);
      memcpy(to->send->lst, from->recv->lst, sizeof(int) * num);
    }

    if (vol > 0) {
      to->send->ind = (int *)malloc(sizeof(int) * vol);
      memcpy(to->send->ind, from->recv->ind, sizeof(int) * vol);

      to->send->val = (double *)malloc(sizeof(double) * vol);
    }
  }

  /*copy from->send 2 to->recv*/

  if (from->send != NULL) {
    int vol = from->send->all[numProcs];
    int num = from->send->num;

    to->recv = allocCommHandle();
    to->recv->num = num;
    to->recv->all = (int *)malloc(sizeof(int) * (numProcs + 1));
    memcpy(to->recv->all, from->send->all, sizeof(int) * (numProcs + 1));

    if (num > 0) {
      to->recv->lst = (int *)malloc(sizeof(int) * num);
      memcpy(to->recv->lst, from->send->lst, sizeof(int) * num);
    }
    if (vol > 0) {
      to->recv->ind = (int *)malloc(sizeof(int) * vol);
      memcpy(to->recv->ind, from->send->ind, sizeof(int) * vol);

      to->recv->val = (double *)malloc(sizeof(double) * vol);
    }
  }

  return;
}

/**********************************************************************/
void setup(int mxCnt, int *commSteps, buMatrix **matrixChain, buMatrix **locs,
           buMatrix **cpls, comm **ins, comm **outs, MPI_Comm parentComm) {
  int *map;
  int szmap;
  int i;

  buMatrix *AlocT;
  buMatrix *AcplT;

  int myId;
  MPI_Comm libComm;

  getLibComm(parentComm, &libComm);
  MPI_Comm_rank(libComm, &myId);

  szmap = matrixChain[0]->gn + matrixChain[0]->gm;
  map = (int *)malloc(sizeof(int) * szmap);

  MPI_Barrier(libComm);

  for (i = 0; i < mxCnt; i++) {
    buMatrix *A = matrixChain[i];
    buMatrix *Aloc = locs[i];
    buMatrix *Acpl = cpls[i];
    comm *in = ins[i];
    comm *out = outs[i];

    informFromWhomToReceive(A, in, out, commSteps[i], libComm);

    split(commSteps[i], A, Aloc, Acpl, libComm);

    get_LocalIndices(A, commSteps[i], map, libComm);

    set_LocalIndices(Aloc, Acpl, in, out, map, commSteps[i], libComm);

    if (commSteps[i] == PRECOMM) {
      if (Aloc->store == STORE_BY_ROWS) {
        AlocT = (buMatrix *)malloc(sizeof(buMatrix));

        switchStore(Aloc, AlocT);

        freeMatrix(locs[i]);

        locs[i]->store = STORE_BY_COLUMNS;

        allocMatrix(locs[i]);

        copyMatrix(AlocT, locs[i]);

        freeMatrix(AlocT);
        free(AlocT);
      }

      if (Acpl->store == STORE_BY_ROWS) {
        AcplT = (buMatrix *)malloc(sizeof(buMatrix));

        switchStore(Acpl, AcplT);

        freeMatrix(cpls[i]);

        cpls[i]->store = STORE_BY_COLUMNS;

        allocMatrix(cpls[i]);

        copyMatrix(AcplT, cpls[i]);

        freeMatrix(AcplT);
        free(AcplT);
      }
    } else {
      if (Aloc->store == STORE_BY_COLUMNS) {
        AlocT = (buMatrix *)malloc(sizeof(buMatrix));

        switchStore(Aloc, AlocT);

        freeMatrix(locs[i]);

        locs[i]->store = STORE_BY_ROWS;

        allocMatrix(locs[i]);

        copyMatrix(AlocT, locs[i]);

        freeMatrix(AlocT);
        free(AlocT);
      }
      if (Acpl->store == STORE_BY_COLUMNS) {
        AcplT = (buMatrix *)malloc(sizeof(buMatrix));

        switchStore(Acpl, AcplT);

        freeMatrix(cpls[i]);

        cpls[i]->store = STORE_BY_ROWS;

        allocMatrix(cpls[i]);

        copyMatrix(AcplT, cpls[i]);

        freeMatrix(AcplT);
        free(AcplT);
      }
    }
  }

  MPI_Barrier(libComm);
  // if(myId==0)
  //    printf("\n\t****\tSet up is OK\t\t****");

  free(map);
  return;
}

/**********************************************************************/
void set_LocalIndices(buMatrix *Aloc, buMatrix *Acpl, comm *in, comm *out,
                      int *map, int commStep, MPI_Comm libComm) {
  if (commStep == PRECOMM)
    set_inLocalIndices(Aloc, Acpl, in, map, libComm);
  else if (commStep == POSTCOMM)
    set_outLocalIndices(Aloc, Acpl, out, map, libComm);
  else if (commStep == PREPOSTCMM) {

  } else {
    printf("\n\t*** commStep %d is not possible", commStep);
    MPI_Finalize();
    exit(1002);
  }
  return;
}

void set_inLocalIndices(buMatrix *Aloc, buMatrix *Acpl, comm *in, int *map,
                        MPI_Comm libComm) {
  int i;

  /*set in Aloc*/
  if (Aloc->store != STORE_BY_ROWS) {
    printf("\n\t*** Aloc'is not stored rowwise in set_inLocalIndices ***");
    MPI_Finalize();
    exit(1002);
  }
  for (i = 0; i < Aloc->m; i++) {
    int j, js = Aloc->ia[i], je = Aloc->ia[i + 1];
    for (j = js; j < je; j++) {
      int col = Aloc->ja[j];
      int new = map[col];
      Aloc->ja[j] = new;
    }
  }

  /*set in Acpl*/
  if (Acpl->store != STORE_BY_ROWS) {
    printf("\n\t*** Acpl'is not stored rowwise in set_inLocalIndices ***");
    MPI_Finalize();
    exit(1002);
  }
  for (i = 0; i < Acpl->m; i++) {
    int j, js = Acpl->ia[i], je = Acpl->ia[i + 1];
    for (j = js; j < je; j++) {
      int col = Acpl->ja[j];
      int new = map[col];
      Acpl->ja[j] = new;
    }
  }

  /*set in inVector*/

  /*set in in->recv*/
  for (i = 0; i < in->recv->num; i++) {
    int p = in->recv->lst[i];
    int j, js = in->recv->all[p], je = in->recv->all[p + 1];
    for (j = js; j < je; j++) {
      int ind = in->recv->ind[j];
      int new = map[ind];
      in->recv->ind[j] = new;
    }
  }

  /*set in in->send*/
  for (i = 0; i < in->send->num; i++) {
    int p = in->send->lst[i];
    int j, js = in->send->all[p], je = in->send->all[p + 1];
    for (j = js; j < je; j++) {
      int ind = in->send->ind[j];
      int new = map[ind];
      in->send->ind[j] = new;
    }
  }

  return;
}
void set_outLocalIndices(buMatrix *Aloc, buMatrix *Acpl, comm *out, int *map,
                         MPI_Comm libComm) {

  int i;
  /*set in Aloc*/
  if (Aloc->store != STORE_BY_COLUMNS) {
    printf("\n\t*** Aloc is not stored columnwise in set_outLocalIndices ***");
    MPI_Finalize();
    exit(1002);
  }
  for (i = 0; i < Aloc->n; i++) {
    int j, js = Aloc->ia[i], je = Aloc->ia[i + 1];
    for (j = js; j < je; j++) {
      int row = Aloc->ja[j];
      int new = map[row];
      Aloc->ja[j] = new;
    }
  }

  /*set in Acpl*/
  if (Acpl->store != STORE_BY_COLUMNS) {
    printf("\n\t*** Acpl'is not stored columnwise in set_outLocalIndices ***");
    MPI_Finalize();
    exit(1002);
  }
  for (i = 0; i < Acpl->n; i++) {
    int j, js = Acpl->ia[i], je = Acpl->ia[i + 1];
    for (j = js; j < je; j++) {
      int row = Acpl->ja[j];
      int new = map[row];
      Acpl->ja[j] = new;
    }
  }
  /*set in outVector*/

  /*set in out->recv*/
  for (i = 0; i < out->recv->num; i++) {
    int p = out->recv->lst[i];
    int j, js = out->recv->all[p], je = out->recv->all[p + 1];
    for (j = js; j < je; j++) {
      int ind = out->recv->ind[j];
      int new = map[ind];
      out->recv->ind[j] = new;
    }
  }
  /*set in out->send*/
  for (i = 0; i < out->send->num; i++) {
    int p = out->send->lst[i];
    int j, js = out->send->all[p], je = out->send->all[p + 1];
    for (j = js; j < je; j++) {
      int ind = out->send->ind[j];
      int new = map[ind];
      out->send->ind[j] = new;
    }
  }
  return;
}
/**********************************************************************/

void get_LocalIndices(buMatrix *A, int commStep, int *map, MPI_Comm libComm) {
  if (commStep == PRECOMM)
    get_inLocalIndices(A, map, libComm);
  else if (commStep == POSTCOMM)
    get_outLocalIndices(A, map, libComm);
  else if (commStep == PREPOSTCMM) {

  } else {
    printf("\n\t*** commStep %d is not available", commStep);
    MPI_Finalize();
    exit(1002);
  }
  return;
}

/**********************************************************************
 *
 * Note: inLocalIndex are such that the x's belong to processor p-1
 *                                         are before processor p.
 *
 **********************************************************************/
void get_inLocalIndices(buMatrix *A, /*input*/
                        int *xLocalIndex /*output:has global space*/,
                        MPI_Comm libComm) {

  int numProcs;
  int myId;
  int i;
  int *perProcs;
  int xCnt = A->gn;
  int *xpartvec = A->inPart;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  if (A->store != STORE_BY_ROWS) {
    printf("\n\t*** A'is not stored rowwise in get_inLocalIndices ***");
    MPI_Finalize();
    exit(1002);
  }

  perProcs = (int *)malloc(sizeof(int) * numProcs);
  memset(perProcs, 0, sizeof(int) * numProcs);

  for (i = 0; i < xCnt; i++)
    xLocalIndex[i] = -2;

  for (i = 0; i < A->m; i++) {
    int j, js = A->ia[i], je = A->ia[i + 1];
    for (j = js; j < je; j++) {
      int col = A->ja[j];
      int p = xpartvec[col];
      if (xLocalIndex[col] == -2) {
        xLocalIndex[col] = -1;
        perProcs[p]++;
      }
    }
  }
  for (i = 0; i < A->gn; i++) {
    int p = A->inPart[i];
    if ((p == myId) && (xLocalIndex[i] == -2)) {
      xLocalIndex[i] = -1;
      perProcs[p]++;
    }
  }

  for (i = 1; i < numProcs; i++) {
    perProcs[i] += perProcs[i - 1];
  }
  for (i = xCnt - 1; i >= 0; i--) {
    if (xLocalIndex[i] == -1) {
      int p = xpartvec[i];
      int ind = --perProcs[p];

#ifdef DEBUG
      if (ind < 0) {
        printf("\n\t*** ind < 0 in get_x_LocalIndex");
        MPI_Finalize();
        exit(1040);
      }
#endif
      xLocalIndex[i] = ind;
    }
  }

  free(perProcs);

  return;
}

void assembleFromCoordinate(int *rowIndices, int *colIndices, double *val,
                            int cnt, buMatrix *A, int storageScheme, int myId,
                            MPI_Comm libComm) {
  /* PRE:  A->gm, A->gn, A->nnz are set before call.
   * PRE:  A->inPart, A->outPart are allocated and set before call.
   * PRE: A->m, A->n,  are set in this routine.
   * POST: A->ia, A->ja, A->val are allocated and set in this routine.
   */
  int i;
  int *useIa;
  int *useJa;
  int iaSz;

  if (A->nnz != cnt) {
    printf("\n\t*** A'is nnz is not set");
    MPI_Finalize();
    exit(1002);
  }
  A->store = storageScheme;

#ifdef PRINT_SIZES
  printf("\n%d m = %d, n=%d, nnz=%d\n", myId, A->m, A->n, A->nnz);
#endif
  if (storageScheme == STORE_BY_ROWS) {
    useIa = rowIndices;
    useJa = colIndices;
    A->ia = (int *)malloc(sizeof(int) * (A->m + 1));
    memset(A->ia, 0, sizeof(int) * (A->m + 1));
    iaSz = A->m + 1;
  } else if (storageScheme == STORE_BY_COLUMNS) {
    useIa = colIndices;
    useJa = rowIndices;
    A->ia = (int *)malloc(sizeof(int) * (A->n + 1));
    memset(A->ia, 0, sizeof(int) * (A->n + 1));
    iaSz = A->n + 1;
  }

  A->val = (double *)malloc(sizeof(double) * A->nnz);
  A->ja = (int *)malloc(sizeof(int) * A->nnz);

  for (i = 0; i < cnt; i++) {
    int indIa = useIa[i];
    if (indIa < 0) {
      printf("\n\t*** indIa(%d) < 0 in traverse***", indIa);
      MPI_Finalize();
      exit(1002);
    }
    if (indIa >= iaSz) {
      printf("\n\t*** indIa(%d) >= iaSz in traverse***", indIa);
      MPI_Finalize();
      exit(1002);
    }
    A->ia[indIa]++;
  }

  for (i = 1; i < iaSz; i++)
    A->ia[i] += A->ia[i - 1];

  for (i = cnt - 1; i >= 0; i--) {
    int indIa = useIa[i];
    int indJa = useJa[i];
    double vali = val[i];

    int putInd;

    if (indIa < 0) {
      printf("\n\t*** indIa(%d) < 0***", indIa);
      MPI_Finalize();
      exit(1002);
    }
    putInd = --A->ia[indIa];
    if (putInd < 0) {
      printf("\n\t*** putInd (%d) < 0***", putInd);
      MPI_Finalize();
      exit(1002);
    }

    A->ja[putInd] = indJa;
    A->val[putInd] = vali;
  }

  if (A->ia[0] != 0) {
    printf("\n\t*** A->ia[0] is not zero but %d***", A->ia[0]);
    MPI_Finalize();
    exit(1002);
  }
  if (A->ia[iaSz - 1] != A->nnz) {
    printf("\n\t*** A->ia[sizeIA] is %d but %d nnz., %d***", A->ia[iaSz - 1],
           A->nnz, myId);
    MPI_Finalize();
    exit(1002);
  }
  MPI_Barrier(libComm);
  if (myId == 0) {
    // printf("\t\t***\tCSR matrix assembled\t***\n");
  }

  return;
}

int countDiffIndices(int *lst, int lstSz, int gloSz, int *partVec, int thisP) {
  int i;
  int cnt;
  int *map;
  map = (int *)malloc(sizeof(int) * gloSz);
  memset(map, 0, sizeof(int) * gloSz);

  cnt = 0;
  for (i = 0; i < lstSz; i++) {
    int ind = lst[i];
    if (ind >= gloSz) {
      printf("\nsomething stupid\n");
      MPI_Finalize();
      exit(111);
    }

    map[ind]++;

    if (map[ind] == 1)
      cnt++;
  }
  /*this is necessary,
    thisP may be responsible for some y[i]/x[i]
    although it has not got any single entry in row[i]/col[i]*/
  for (i = 0; i < gloSz; i++) {
    if ((partVec[i] == thisP) && (map[i] == 0))
      cnt++;
  }
  free(map);
  return cnt;
}

void setRowLocalIndices(int *rowIndices, int rowIndCnt, int *map) {
  int i;
  for (i = 0; i < rowIndCnt; i++) {
    int row = rowIndices[i];
    int newInd = map[row];
    rowIndices[i] = newInd;
  }

  return;
}

void findRowMap(int *rowIndices, int rowIndCnt, int *yPartVec, int ySz,
                MPI_Comm libComm, int *map /*out*/) {
  int *perProcs;
  int numProcs;
  int myId;
  int i;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  perProcs = (int *)malloc(sizeof(int) * numProcs);
  memset(perProcs, 0, sizeof(int) * numProcs);

  for (i = 0; i < ySz; i++)
    map[i] = -2;

  for (i = 0; i < rowIndCnt; i++) {
    int row = rowIndices[i];
    int p = yPartVec[row];
    if (map[row] == -2) {
      map[row] = -1;
      perProcs[p]++;
    }
  }

  for (i = 0; i < ySz; i++) {
    int p = yPartVec[i];
    if ((p == myId) && (map[i] == -2)) {
      map[i] = -1;
      perProcs[p]++;
    }
  }

  for (i = 1; i < numProcs; i++) {
    perProcs[i] += perProcs[i - 1];
  }

  for (i = ySz - 1; i >= 0; i--) {
    if (map[i] == -1) {
      int p = yPartVec[i];
      int ind = --perProcs[p];

#ifdef DEBUG
      if (ind < 0) {
        printf("\n\t*** ind < 0 in get_x_LocalIndex");
        MPI_Finalize();
        exit(1040);
      }
#endif
      map[i] = ind;
    }
  }

  free(perProcs);
  MPI_Barrier(libComm);
  if (myId == 0) {
    // printf("\t\t***\tFilled Row map\t***\n");
  }
  return;
}

/**********************************************************************/
void get_outLocalIndices(
    buMatrix *A,      /*input*/
    int *yLocalIndex, /*output: global,Pre:memory allocated*/
    MPI_Comm libComm) {

  int numProcs;
  int myId;
  int i;
  int *perProcs;
  int yCnt = A->gm;
  int *ypartvec = A->outPart;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  if (A->store != STORE_BY_COLUMNS) {
    printf("\n\t*** A'is not stored rowwise in get_outLocalIndices ***");
    MPI_Finalize();
    exit(1002);
  }

  perProcs = (int *)malloc(sizeof(int) * numProcs);
  memset(perProcs, 0, sizeof(int) * numProcs);

  for (i = 0; i < yCnt; i++)
    yLocalIndex[i] = -2;

  for (i = 0; i < A->n; i++) {
    int j, js = A->ia[i], je = A->ia[i + 1];
    for (j = js; j < je; j++) {
      int row = A->ja[j];
      int p = ypartvec[row];
      if (yLocalIndex[row] == -2) {
        yLocalIndex[row] = -1;
        perProcs[p]++;
      }
    }
  }

  for (i = 0; i < A->gm; i++) {
    int p = A->outPart[i];
    if ((p == myId) && (yLocalIndex[i] == -2)) {
      yLocalIndex[i] = -1;
      perProcs[p]++;
    }
  }

  for (i = 1; i < numProcs; i++) {
    perProcs[i] += perProcs[i - 1];
  }
  for (i = yCnt - 1; i >= 0; i--) {
    if (yLocalIndex[i] == -1) {
      int p = ypartvec[i];
      int ind = --perProcs[p];

#ifdef DEBUG
      if (ind < 0) {
        printf("\n\t*** ind < 0 in get_x_LocalIndex");
        MPI_Finalize();
        exit(1040);
      }
#endif
      yLocalIndex[i] = ind;
    }
  }

  free(perProcs);

  return;
}

/**********************************************************************/
/*send global indices of input elements which belong to some other processor p,
       to processor p
*/

void informFromWhomToReceive(buMatrix *A, comm *inX, comm *outY, int commStep,
                             MPI_Comm libComm) {
  if (commStep == PRECOMM)
    informFromWhomToReceive_Rowwise(A, inX, libComm);
  else if (commStep == POSTCOMM)
    informFromWhomToReceive_Columnwise(A, outY, libComm);
  else if (commStep == PREPOSTCMM) {

  } else {
    printf("\n\t*** commStep %d is not implemented", commStep);
    MPI_Finalize();
    exit(1041);
  }
  return;
}

void informFromWhomToReceive_Rowwise(buMatrix *A, comm *inX, MPI_Comm libComm) {
  int numProcs;
  int myId;

  int i;
  int *recvSizes, *sendSizes;
  int *allColumns;
  int *xpartvec = A->inPart;
  int xCnt = A->gn;
  int numRecvs, numSends, volRecvs = 0, volSends = 0;

  MPI_Request *mySendReqs;
  MPI_Request *myRecvReqs;
  MPI_Status *stts;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  if (A->store != STORE_BY_ROWS) {
    printf("\n\t*** A->store != BY_ROWS in informFromWhomToReceive_Rowwise");
    MPI_Finalize();
    exit(1041);
  }

  recvSizes = (int *)malloc(sizeof(int) * numProcs);
  memset(recvSizes, 0, sizeof(int) * numProcs);
  sendSizes = (int *)malloc(sizeof(int) * numProcs);
  memset(sendSizes, 0, sizeof(int) * numProcs);
  allColumns = (int *)malloc(sizeof(int) * A->gn);

  for (i = 0; i < xCnt; i++)
    allColumns[i] = -2;
  for (i = 0; i < A->m; i++) {
    int j, js = A->ia[i], je = A->ia[i + 1];
    for (j = js; j < je; j++) {
      int c = A->ja[j];
      int p = xpartvec[c];
      if (allColumns[c] == -2) {
        allColumns[c] = -1;
        recvSizes[p]++;
      }
    }
  }
  MPI_Alltoall(recvSizes, 1, MPI_INT, sendSizes, 1, MPI_INT, libComm);

  numRecvs = numSends = 0;

  recvSizes[myId] = 0;
  sendSizes[myId] = 0;

  for (i = 0; i < numProcs; i++) {
    volRecvs += recvSizes[i];
    volSends += sendSizes[i];
    if (recvSizes[i] != 0)
      numRecvs++;

    if (sendSizes[i] != 0)
      numSends++;
  }

  mySendReqs = (MPI_Request *)malloc(sizeof(MPI_Request) * numProcs);
  myRecvReqs = (MPI_Request *)malloc(sizeof(MPI_Request) * numProcs);
  stts = (MPI_Status *)malloc(sizeof(MPI_Status) * numProcs);

  inX->send = allocCommHandle();
  inX->send->all = (int *)malloc(sizeof(int) * (numProcs + 1));
  inX->send->num = numSends;
  if (numSends > 0)
    inX->send->lst = (int *)malloc(sizeof(int) * numSends);
  if (volSends > 0) {
    inX->send->ind = (int *)malloc(sizeof(int) * volSends);
    inX->send->val = (double *)malloc(sizeof(double) * volSends);
  }

  inX->recv = allocCommHandle();
  inX->recv->all = (int *)malloc(sizeof(int) * (numProcs + 1));
  inX->recv->num = numRecvs;
  if (numRecvs > 0)
    inX->recv->lst = (int *)malloc(sizeof(int) * numRecvs);
  if (volRecvs > 0)
    inX->recv->ind = (int *)malloc(sizeof(int) * volRecvs);

  /*set pointers in send and recv*/
  numRecvs = numSends = volRecvs = volSends = 0;
  inX->recv->all[0] = inX->send->all[0] = 0;
  for (i = 0; i < numProcs; i++) {

    if (recvSizes[i] != 0) {
      inX->recv->lst[numRecvs++] = i;
      volRecvs += recvSizes[i];
    }
    inX->recv->all[i + 1] = volRecvs;
    if (sendSizes[i] != 0) {
      inX->send->lst[numSends++] = i;
      volSends += sendSizes[i];
    }
    inX->send->all[i + 1] = volSends;
    if (numRecvs > inX->recv->num) {
      printf("\nolmuyor in asdasd myId=%d, num=%d, recvnum=%d", myId, numRecvs,
             inX->recv->num);
      MPI_Finalize();
      exit(1045);
    }
    if (numSends > inX->send->num) {
      printf("\nolmuyor in asdasd2222");
      MPI_Finalize();
      exit(1045);
    }
  }
  /*fill up indices for recv*/
  for (i = 0; i < xCnt; i++) {
    if ((allColumns[i] == -1) && (xpartvec[i] != myId)) {
      int p = xpartvec[i];
      int put = inX->recv->all[p];
      inX->recv->all[p]++;
      inX->recv->ind[put] = i;
    }
  }

  /*re-adjust recv->all*/
  for (i = numProcs; i > 0; i--) {
    inX->recv->all[i] = inX->recv->all[i - 1];
  }
  inX->recv->all[0] = 0;

  for (i = 0; i < inX->recv->num; i++) {
    int p = inX->recv->lst[i];
    int sz = inX->recv->all[p + 1] - inX->recv->all[p];
    int st = inX->recv->all[p];
    MPI_Isend(&(inX->recv->ind[st]), sz, MPI_INT, p, SEND_ME, libComm,
              &mySendReqs[i]);
  }

  for (i = 0; i < inX->send->num; i++) {
    int p = inX->send->lst[i];
    int sz = inX->send->all[p + 1] - inX->send->all[p];
    int get = inX->send->all[p];
    MPI_Irecv(&(inX->send->ind[get]), sz, MPI_INT, p, SEND_ME, libComm,
              &myRecvReqs[i]);
  }

  if (inX->recv->num > 0)
    MPI_Waitall(inX->recv->num, mySendReqs, stts);

  if (inX->send->num > 0)
    MPI_Waitall(inX->send->num, myRecvReqs, stts);

  MPI_Barrier(libComm);

  free(allColumns);
  free(recvSizes);
  free(sendSizes);
  free(mySendReqs);
  free(myRecvReqs);
  free(stts);
  return;
}

/**********************************************************************/
/*send global indices of output elements, which will be folded by some other
   processor p, to processor p
*/

void informFromWhomToReceive_Columnwise(buMatrix *A, comm *outY,
                                        MPI_Comm libComm)

{
  int numProcs;
  int myId;

  int i;
  int *recvSizes, *sendSizes;
  int *allRows;
  int *ypartvec = A->outPart;
  int yCnt = A->gm;
  int numRecvs, numSends, volRecvs = 0, volSends = 0;

  MPI_Request *mySendReqs;
  MPI_Request *myRecvReqs;
  MPI_Status *stts;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  if (A->store != STORE_BY_COLUMNS) {
    printf(
        "\n\t*** A->store != BY_COLUMNS in informFromWhomToReceive_Columnwise");
    MPI_Finalize();
    exit(1041);
  }

  recvSizes = (int *)malloc(sizeof(int) * numProcs);
  memset(recvSizes, 0, sizeof(int) * numProcs);
  sendSizes = (int *)malloc(sizeof(int) * numProcs);
  memset(sendSizes, 0, sizeof(int) * numProcs);
  allRows = (int *)malloc(sizeof(int) * A->gm);

  for (i = 0; i < yCnt; i++)
    allRows[i] = -2;
  for (i = 0; i < A->n; i++) {
    int j, js = A->ia[i], je = A->ia[i + 1];
    for (j = js; j < je; j++) {
      int r = A->ja[j];
      int p = ypartvec[r];
      if (allRows[r] == -2) {
        allRows[r] = -1;
        sendSizes[p]++;
      }
    }
  }

  MPI_Alltoall(sendSizes, 1, MPI_INT, recvSizes, 1, MPI_INT, libComm);

  numRecvs = numSends = 0;
  recvSizes[myId] = 0;
  sendSizes[myId] = 0;
  for (i = 0; i < numProcs; i++) {
    volRecvs += recvSizes[i];
    volSends += sendSizes[i];
    if (recvSizes[i] != 0)
      numRecvs++;

    if (sendSizes[i] != 0)
      numSends++;
  }

  mySendReqs = (MPI_Request *)malloc(sizeof(MPI_Request) * numProcs);
  myRecvReqs = (MPI_Request *)malloc(sizeof(MPI_Request) * numProcs);
  stts = (MPI_Status *)malloc(sizeof(MPI_Status) * numProcs);

  outY->send = allocCommHandle();
  outY->send->all = (int *)malloc(sizeof(int) * (numProcs + 1));
  outY->send->num = numSends;
  if (numSends > 0)
    outY->send->lst = (int *)malloc(sizeof(int) * numSends);
  if (volSends > 0)
    outY->send->ind = (int *)malloc(sizeof(int) * volSends);

  outY->recv = allocCommHandle();
  outY->recv->all = (int *)malloc(sizeof(int) * (numProcs + 1));
  outY->recv->num = numRecvs;
  if (numRecvs > 0)
    outY->recv->lst = (int *)malloc(sizeof(int) * numRecvs);
  if (volRecvs > 0) {
    outY->recv->ind = (int *)malloc(sizeof(int) * volRecvs);
    outY->recv->val = (double *)malloc(sizeof(double) * volRecvs);
  }

  /*set pointers in send and recv*/
  numRecvs = numSends = volRecvs = volSends = 0;
  outY->recv->all[0] = outY->send->all[0] = 0;
  for (i = 0; i < numProcs; i++) {
    if (recvSizes[i] != 0) {
      outY->recv->lst[numRecvs++] = i;
      volRecvs += recvSizes[i];
    }
    outY->recv->all[i + 1] = volRecvs;
    if (sendSizes[i] != 0) {
      outY->send->lst[numSends++] = i;
      volSends += sendSizes[i];
    }
    outY->send->all[i + 1] = volSends;
    if (numRecvs > outY->recv->num) {
      printf("\nolmuyor in column.. myId=%d, num=%d, recvnum=%d", myId,
             numRecvs, outY->recv->num);
      MPI_Finalize();
      exit(1045);
    }
    if (numSends > outY->send->num) {
      printf("\nolmuyor in column.. asdasd2222");
      MPI_Finalize();
      exit(1045);
    }
  }
  /*fill up indices for send*/
  for (i = 0; i < yCnt; i++) {
    if ((allRows[i] == -1) && (ypartvec[i] != myId)) {
      int p = ypartvec[i];
      int put = outY->send->all[p];
      outY->send->all[p]++;
      outY->send->ind[put] = i;
    }
  }

  /*re-adjust send->all*/
  for (i = numProcs; i > 0; i--) {
    outY->send->all[i] = outY->send->all[i - 1];
  }
  outY->send->all[0] = 0;

  for (i = 0; i < outY->send->num; i++) {
    int p = outY->send->lst[i];
    int sz = outY->send->all[p + 1] - outY->send->all[p];
    int st = outY->send->all[p];
    MPI_Isend(&(outY->send->ind[st]), sz, MPI_INT, p, RECV_ME, libComm,
              &mySendReqs[i]);
  }

  for (i = 0; i < outY->recv->num; i++) {
    int p = outY->recv->lst[i];
    int sz = outY->recv->all[p + 1] - outY->recv->all[p];
    int get = outY->recv->all[p];
    MPI_Irecv(&(outY->recv->ind[get]), sz, MPI_INT, p, RECV_ME, libComm,
              &myRecvReqs[i]);
  }

  if (outY->send->num > 0)
    MPI_Waitall(outY->send->num, mySendReqs, stts);

  if (outY->recv->num > 0)
    MPI_Waitall(outY->recv->num, myRecvReqs, stts);

  MPI_Barrier(libComm);

  free(allRows);
  free(recvSizes);
  free(sendSizes);
  free(mySendReqs);
  free(myRecvReqs);
  free(stts);
  return;
}

/**********************************************************************/
/* note: A is of size m_by_n with nnz nonzeros. Aloc and Acpl will also be of
 * the same size where their total nnz is nnz.
 */

void split(int commStep, buMatrix *A, /*input with global indices*/
           buMatrix *Aloc, /*only local, either y local or x'are local*/
           buMatrix *Acpl,
           /*either to another processor's y entry or needs an x entry*/
           MPI_Comm libComm) {

  if (commStep == PRECOMM)
    split_Rowwise(A, Aloc, Acpl, libComm);
  else if (commStep == POSTCOMM)
    split_Colwise(A, Aloc, Acpl, libComm);
  else if (commStep == PREPOSTCMM) {

  } else {
    printf("\n\t*** commStep %d is not available", commStep);
    MPI_Finalize();
    exit(1001);
  }

  MPI_Barrier(libComm);

  return;
}
/**********************************************************************/

void split_Rowwise(buMatrix *A, buMatrix *Aloc, buMatrix *Acpl,
                   MPI_Comm libComm) {
  int i;
  int gloNnz = 0;
  int locNnz = 0;
  int myId;
  int *xpartvec;

  MPI_Comm_rank(libComm, &myId);

  xpartvec = A->inPart;
  if (A->store != STORE_BY_ROWS) {
    printf("\n\t*** A'is not stored rowwise in PRECOMM ***");
    MPI_Finalize();
    exit(1001);
  }
  Aloc->gn = Acpl->gn = A->gn;
  Aloc->gm = Acpl->gm = A->gm;

  Aloc->m = Acpl->m = A->m;
  Aloc->n = Acpl->n = A->n;

  Aloc->ia = (int *)malloc(sizeof(int) * (Aloc->m + 1));
  Acpl->ia = (int *)malloc(sizeof(int) * (Acpl->m + 1));

  memset(Aloc->ia, 0, sizeof(int) * (Aloc->m + 1));
  memset(Acpl->ia, 0, sizeof(int) * (Acpl->m + 1));

  locNnz = gloNnz = 0;
  for (i = 0; i < A->m; i++) {
    int glo = 0;
    int je = A->ia[i + 1];
    int j = A->ia[i];
    for (; j < je; j++) {
      int col = A->ja[j];
      if (xpartvec[col] != myId)
        gloNnz++;
      else
        locNnz++;
    }
    Aloc->ia[i + 1] = locNnz;
    Acpl->ia[i + 1] = gloNnz;
  }

  Aloc->store = Acpl->store = STORE_BY_ROWS;

  Aloc->nnz = locNnz;
  Aloc->ja = (int *)malloc(sizeof(int) * locNnz);
  Aloc->val = (double *)malloc(sizeof(double) * locNnz);

  Acpl->nnz = gloNnz;
  Acpl->ja = (int *)malloc(sizeof(int) * gloNnz);
  Acpl->val = (double *)malloc(sizeof(double) * gloNnz);

  Aloc->inPart = (int *)malloc(sizeof(int) * A->gn);
  Acpl->inPart = (int *)malloc(sizeof(int) * A->gn);
  Aloc->outPart = (int *)malloc(sizeof(int) * A->gm);
  Acpl->outPart = (int *)malloc(sizeof(int) * A->gm);

  gloNnz = locNnz = 0;
  for (i = 0; i < A->m; i++) {
    int j, js, je, sz;
    int glo = 0;
    je = A->ia[i + 1];
    j = js = A->ia[i];
    sz = je - js;
    for (; j < je; j++) {
      int col = A->ja[j];
      double val = A->val[j];
      if (xpartvec[col] != myId) {
        Acpl->ja[gloNnz] = col;
        Acpl->val[gloNnz] = val;
        gloNnz++;
      } else {
        Aloc->ja[locNnz] = col;
        Aloc->val[locNnz] = val;
        locNnz++;
      }
    }
  }
  memcpy(Aloc->inPart, A->inPart, sizeof(int) * A->gn);
  memcpy(Aloc->outPart, A->outPart, sizeof(int) * A->gm);
  memcpy(Acpl->inPart, A->inPart, sizeof(int) * A->gn);
  memcpy(Acpl->outPart, A->outPart, sizeof(int) * A->gm);

  /* if(myId == 1)
      {
      printMatrix(Aloc);
      }
  */
  return;
}

/**********************************************************************/
void split_Colwise(buMatrix *A, buMatrix *Aloc, buMatrix *Acpl,
                   MPI_Comm libComm) {
  int i;
  int gloNnz = 0;
  int locNnz = 0;
  int myId;
  int *ypartvec;

  MPI_Comm_rank(libComm, &myId);

  ypartvec = A->outPart;
  if (A->store != STORE_BY_COLUMNS) {
    printf("\n\t*** A'is not stored columnwise in split_Colwise ***");
    MPI_Finalize();
    exit(1001);
  }
  Aloc->gn = Acpl->gn = A->gn;
  Aloc->gm = Acpl->gm = A->gm;

  Aloc->m = Acpl->m = A->m;
  Aloc->n = Acpl->n = A->n;

  Aloc->ia = (int *)malloc(sizeof(int) * (Aloc->n + 1));
  Acpl->ia = (int *)malloc(sizeof(int) * (Acpl->n + 1));
  memset(Aloc->ia, 0, sizeof(int) * (Aloc->n + 1));
  memset(Acpl->ia, 0, sizeof(int) * (Acpl->n + 1));

  for (i = 0; i < A->n; i++) {
    int j, js, je, sz;
    int glo = 0;
    je = A->ia[i + 1];
    j = js = A->ia[i];
    sz = je - js;
    for (; j < je; j++) {
      int row = A->ja[j];
      if (ypartvec[row] != myId)
        gloNnz++;
      else
        locNnz++;
    }
    Aloc->ia[i + 1] = locNnz;
    Acpl->ia[i + 1] = gloNnz;
  }

  Aloc->store = Acpl->store = STORE_BY_COLUMNS;
  Aloc->nnz = locNnz;
  Acpl->nnz = gloNnz;
  Aloc->val = (double *)malloc(sizeof(double) * locNnz);
  Aloc->ja = (int *)malloc(sizeof(int) * locNnz);
  Acpl->ja = (int *)malloc(sizeof(int) * gloNnz);
  Acpl->val = (double *)malloc(sizeof(double) * gloNnz);
  Aloc->inPart = (int *)malloc(sizeof(int) * A->gn);
  Acpl->inPart = (int *)malloc(sizeof(int) * A->gn);
  Aloc->outPart = (int *)malloc(sizeof(int) * A->gm);
  Acpl->outPart = (int *)malloc(sizeof(int) * A->gm);

  gloNnz = locNnz = 0;
  for (i = 0; i < A->n; i++) {
    int j, js, je, sz;
    int glo = 0;
    je = A->ia[i + 1];
    j = js = A->ia[i];
    sz = je - js;
    for (; j < je; j++) {
      int row = A->ja[j];
      double val = A->val[j];
      if (ypartvec[row] != myId) {
        Acpl->ja[gloNnz] = row;
        Acpl->val[gloNnz] = val;
        gloNnz++;
      } else {
        Aloc->ja[locNnz] = row;
        Aloc->val[locNnz] = val;
        locNnz++;
      }
    }
  }
  memcpy(Aloc->inPart, A->inPart, sizeof(int) * A->gn);
  memcpy(Aloc->outPart, A->outPart, sizeof(int) * A->gm);
  memcpy(Acpl->inPart, A->inPart, sizeof(int) * A->gn);
  memcpy(Acpl->outPart, A->outPart, sizeof(int) * A->gm);
  return;
}
