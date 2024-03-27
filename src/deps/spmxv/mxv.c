/****************************************************************
 *  by: bora ucar 28/08/2002                                    *
 *   sparse matrix vector multiplication.                       *
 *                                                              *
 ****************************************************************/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "spmxv.h"

//#define DEBUG

/*
//#define DEBUG_2
*/
#define DO_COMM
#define WAIT_RECEIVES
#define MX 10

void mxv_pre(buMatrix *Aloc,
	     buMatrix *Acpl,
	     buVector *x,
	     buVector *y, int mxvTag, MPI_Comm libComm);
void mxv_post(buMatrix *Aloc,
	      buMatrix *Acpl,
	      buVector *x,
	      buVector *y, int mxvTag, MPI_Comm libComm);
void mxv_prepost(buMatrix *Aloc,
		 buMatrix *Acpl,
		 buVector *x, /*x->val begins at my first x value*/
		 buVector *y, int mxvTag, MPI_Comm libComm, MPI_Comm parentComm);
void post_comm(buMatrix *Aloc, buVector *y, int commTag, MPI_Comm libComm);

/*------------------------------------------------------------------*/
/*
 *
 * mxv computes y=Ax. It does necessary communications on x or y. 
 *      PRECOMM refers communicaion on x,
 *      POSTCOMM refers communication on y
 */   
void mxv(parMatrix *A,
	 buVector *x,
	 buVector *y, MPI_Comm parentComm)
{ 
  int partScheme = A->scheme;

  int commScheme;
  int mxvTag;
  MPI_Comm libComm;
  getLibComm(parentComm, &libComm);
  getTagMult(parentComm, &mxvTag);

  if(partScheme == PART_BY_ROWS)
    commScheme = PRECOMM;
  else if(partScheme == PART_BY_COLUMNS)
    commScheme = POSTCOMM;
  else if( (partScheme == PART_2D) || (partScheme == PART_CHK))
    commScheme = PREPOSTCMM;
  else 
    {
      printf("partScheme %d is not valid", partScheme);
      MPI_Finalize();
      exit(1216);
    }

  if(commScheme == PRECOMM) 
    {
      comm *save_xcm = x->cm;
      x->cm = A->in; 
      mxv_pre(A->loc, A->cpl, x, y, mxvTag, libComm);
      x->cm = save_xcm;
    }
  else if(commScheme == POSTCOMM)
    {
      comm *save_ycm = y->cm;
      y->cm = A->out;
      mxv_post(A->loc, A->cpl, x, y, mxvTag, libComm);
      y->cm = save_ycm;
    }
  else
    {
      comm *save_xcm = x->cm;
      comm *save_ycm = y->cm;
      x->cm = A->in; 
      y->cm = A->out;
		
      mxv_prepost(A->loc, A->cpl, x, y, mxvTag, libComm, parentComm);

      x->cm = save_xcm;
      y->cm = save_ycm;     
    }
  incrementTagMult(parentComm);
  return ;
}


void mxv_prepost(buMatrix *Aloc,
		 buMatrix *Acpl,
		 buVector *x, /*x->val begins at my first x value*/
		 buVector *y, int mxvTag, MPI_Comm libComm, MPI_Comm parentComm)
{
  /*issue isend x valuse, issue i recv x values,
    do local multiplication,
    wait any message, do cpl multiplication with received x's
    ->> mxv_pre
    Then issue isends, and I recvs on y values. adjust y.
  */
  int postCommTag;
  mxv_pre(Aloc, Acpl, x, y, mxvTag, libComm);
  incrementTagMult(parentComm);
  getTagMult(parentComm, &postCommTag); 
  post_comm(Aloc, y, postCommTag, libComm);
  return;
}


void post_comm(buMatrix *Aloc, buVector *y, int commTag, MPI_Comm libComm)
{
  int i;
  int myId;
  int numProcs;
  int recved, recvInd;
  MPI_Request *sendReqs;
  MPI_Request *recvReqs;
  MPI_Status *stts; 

  commHandle *send = y->cm->send;
  commHandle *recv = y->cm->recv;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  stts = (MPI_Status *) malloc(sizeof(MPI_Status)*numProcs);
  sendReqs = (MPI_Request*) malloc(sizeof(MPI_Request)*numProcs);
  recvReqs = (MPI_Request*) malloc(sizeof(MPI_Request)*numProcs);

  /* 1.
   * Isend at each processor boundary.*/
  for( i = 0 ; i < send->num; i++)
    {
      int p = send->lst[i];
      int ysz = send->all[p+1] - send->all[p];
      int j;
      int js = send->all[p], je = send->all[p+1];
      int yst = send->ind[js];

      MPI_Isend(&(y->val[yst]), ysz, MPI_DOUBLE, p, commTag, libComm, &(sendReqs[i]));
    }


  /* 2.
   * issue Irecv on y values.*/
  for( i = 0 ; i < recv->num; i++)
    {
      int p = recv->lst[i];
      int ysz = recv->all[p+1] - recv->all[p];
      int rind = recv->all[p];

      MPI_Irecv(&(recv->val[rind]), ysz, MPI_DOUBLE, p, commTag, libComm, &(recvReqs[i]));
    }

  /* 3.
   * recv y values and add them up.*/ 
  recved = 0;
  recvInd = -1000;
  while(recved < recv->num)
    {
      MPI_Waitany(recv->num, recvReqs, &recvInd, &stts[0]);

      if(recvInd != MPI_UNDEFINED)
	{
	  int p = stts[0].MPI_SOURCE;
	  int sz = recv->all[p+1] - recv->all[p]; 
	  int st = recv->all[p];
	  int ii;

	  recved ++;
	  recvReqs[recvInd] = MPI_REQUEST_NULL;
	  /*retrieved y from processor p. handle it*/
	  // KAMER 
	  int* prind = recv->ind + st;
	  double* prec = recv->val + st;
	  double* py = y->val;
	  for(ii = 0; ii < sz; ii++) {
	    py[*prind++] += *prec++;	     
	  }	 	 
	}
    }

  if(send->num > 0)
    MPI_Waitall(send->num, sendReqs, stts);

  /*re adjust y->val and y->sz according to outpart*/
  adjustOut(Aloc, y, libComm);

  free(recvReqs);
  free(sendReqs);
  free(stts);
  return;
}

/**********************************************************************/
/*   my x portion is in x. it is enlarged to fit A->n.                */
/*   my portion is placed into suitable place.                        */
/*   y is of size A->m.                                               */
/**********************************************************************/

void mxv_pre(buMatrix *Aloc,
	     buMatrix *Acpl,
	     buVector *x, /*x->val begins at my first x value*/
	     buVector *y, int mxvTag, MPI_Comm libComm)
{
  int i;
  int numProcs;
  int myId;
  int recved;
  int recvInd;

  buVector *xe;
  MPI_Request *sendReqs;
  MPI_Request *recvReqs;
  MPI_Status *stts; 

  commHandle *send = x->cm->send;
  commHandle *recv = x->cm->recv;

  MPI_Comm_rank(libComm, &myId);
  MPI_Comm_size(libComm, &numProcs);

  xe = (buVector*) malloc(sizeof(buVector));

  /*enlarge x and put my x's into suitable places in xe*/
  adjustIn(Aloc, x, xe, libComm);

  if(y->sz != Aloc->m)
    {
      allocVectorData(y, Aloc->m);
    }

#ifdef DEBUG
  if(Aloc->store != STORE_BY_COLUMNS)
    {
      printf("\nAloc is not stored columnwise in mxv_pre");
      MPI_Finalize();
      exit(1890);
    }
  if(Acpl->store != STORE_BY_COLUMNS)
    {
      printf("\nAcpl is not stored columnwise in mxv_pre");
      MPI_Finalize();
      exit(1891);
    }

  if(y->sz != Aloc->m)
    {
      printf("\n y and Aloc is not of the same dimension in mxv_pre");
      MPI_Finalize();
      exit(1892);
    }
  if(y->sz != Acpl->m)
    {
      printf("\n y and Acpl is not of the same dimension in mxv_pre");
      MPI_Finalize();
      exit(1893);
    }
  if(xe->sz != Aloc->n)
    {
      printf("\n x and Aloc is not of the same dimension in mxv_pre %d %d", x->sz, Aloc->n);
      MPI_Finalize();
      exit(1894);
    }
  if(xe->sz != Acpl->n)
    {
      printf("\n x and Acpl is not of the same dimension in mxv_pre");
      MPI_Finalize();
      exit(1895);
    }
#endif

  stts = (MPI_Status *) malloc(sizeof(MPI_Status)*numProcs);
  sendReqs = (MPI_Request*) malloc(sizeof(MPI_Request)*numProcs);
  recvReqs = (MPI_Request*) malloc(sizeof(MPI_Request)*numProcs);

  /*  1. 
   *  (a) issue sends*/

  for(i = 0 ; i<send->num; i++)
    {
      int p = send->lst[i];
      int sz = send->all[p+1] - send->all[p];
      int st = send->all[p];
      int j, js = st, je = send->all[p+1];
#ifdef DEBUG
      if(sz <= 0)
	{
	  printf("\nI am not OK %d", myId);
	  MPI_Finalize();
	  exit(1909);
	}     
#endif

      for( j = js ; j < je; j++)
	{
	  int ind = send->ind[j];
#ifdef DEBUG
	  if(ind >= xe->sz)
	    {
	      printf("\nThis is ridiculous");
	    }
#endif
	  send->val[j] = xe->val[ind];
	}
#ifdef DO_COMM
      MPI_Isend(&(send->val[st]), sz, MPI_DOUBLE, p, mxvTag, libComm, &(sendReqs[i]));
#endif

#ifdef DEBUG
      if(st + sz > send->all[numProcs])
	{
	  printf("\nisend sz.. This is interesting");
	}
#endif
    }
  /*  1. 
   *  (b) issue recvs*/
  for(i = 0; i<recv->num; i++)
    {
      int p = recv->lst[i];
      int sz = recv->all[p+1] - recv->all[p]; 
      int st = recv->all[p];
      int ind = recv->ind[st];
#ifdef DO_COMM
      MPI_Irecv(&(xe->val[ind]), sz, MPI_DOUBLE, p, mxvTag, libComm, &(recvReqs[i]));
#endif

#ifdef DEBUG
      if(sz <= 0)
	{
	  printf("\n I am %d going to receive %d DOUBLES", myId, sz);
	  MPI_Finalize();
	  exit(675);
	}
      if(sz + ind > xe->sz)
	{
	  printf("\nThis is interesting sum");
	}
#endif
    }



#ifdef DEBUG
  if(y->sz <=0)
    {
      printf("\nolmadi  y->sz <=0");
      MPI_Finalize();
      exit(100);
    }
#endif

  //printf("kamer\n");
  for( i = 0 ; i < y->sz; i++)
    y->val[i] = 0.0;

  /*  2.
   *  Aloc multiplication */  
  int n = Aloc->n, j;
  int *ia = Aloc->ia;
  int *ja = Aloc->ja, *jja;
  double *pxe = xe->val;
  double *py = y->val;
  double *pv = Aloc->val, *jpv;
  double xeval;
  
  // KAMER 
  for(i = 0; i < n; i++)   {
    xeval = pxe[i];            
    for(j = ia[i+1] - ia[i]; j; j--) 
      py[*ja++] += xeval * *pv++;      
  }
  
  /*  3. 
   *  wait x's and do Acpl multiplication*/
#ifdef WAIT_RECEIVES
  recved = 0;
  recvInd = -1000;
  double* cplval = Acpl->val;
  int* cplia = Acpl->ia;
  int* cplja = Acpl->ja;
  int* rind = recv->ind;
  int* rall = recv->all;
  while(recved < recv->num)
    {
      MPI_Waitany(recv->num, recvReqs, &recvInd, &stts[0]);
#ifdef DEBUG
      if(recvInd == MPI_UNDEFINED)
	{
	  if(recved < recv->num)
	    {
	      printf("\nolmuyor recv in mxv_pre");
	      MPI_Finalize();
	      exit(1980);
	    }
	}
#endif
      if(recvInd != MPI_UNDEFINED)
	{
	  int p = stts[0].MPI_SOURCE;
	  int sz = rall[p+1] - rall[p]; 
	  int st =  rall[p];
	  int ind = rind[st];
	  int ii;
#ifdef DEBUG
	  if(p != recv->lst[recvInd])
	    {
	      printf("\nin mxv_pre these are not equal???");
	      MPI_Finalize();
	      exit(1198);
	    }
#endif
	  recved++;
	  recvReqs[recvInd] = MPI_REQUEST_NULL;
	  /*retrieved x from processor p. handle it*/

	  // KAMER
	  int j;
	  double xeval;
	  int *ppia = cplia + ind, *ppja;
	  double *ppxe = pxe + ind, *ppval, *pyval = y->val;
	  
	  for(ii = sz; ii; ii--) {
	    xeval = *ppxe++; 	    
	    ppval = cplval + *ppia;
	    ppja = cplja + *ppia;
	    for(j = *(ppia+1) - *ppia; j; j--) {
	      pyval[*ppja++] += xeval * *ppval++;
	    }
	  }

	  /*for(ii = ind ; ii < ie; ii++)
            {
              double xval = xe->val[ii];
              int j, js=Acpl->ia[ii], je=Acpl->ia[ii+1];
              for(j = js; j < je; j++)
                {
                  int yind = Acpl->ja[j];
                  y->val[yind] += xval * Acpl->val[j];
                }
		}*/
	}
    }
#endif


#ifdef DO_COMM
  if(send->num > 0)
    MPI_Waitall(send->num, sendReqs, stts);
#endif

  if(recv->num > 0)
    free(recvReqs);

  if(send->num > 0)
    free(sendReqs);

  free(stts);
  freeVector(xe);
  return; 
}

/**********************************************************************/

void adjustIn(buMatrix *A, buVector *x, buVector *xe, MPI_Comm libComm)
{
  int i; 
  int myId, numProcs;
  int myst;


  int index = 0;
  int sz;

  commHandle *recv = x->cm->recv; 

  MPI_Comm_rank(libComm, &myId);
  MPI_Comm_size(libComm, &numProcs);

  myst = recv->all[myId];
  sz = A->n - recv->all[numProcs];

  if(sz > 0)
    {
      xe->val = (double*) malloc(sizeof(double)*A->n);
#ifdef DEBUG
      if(x->sz > A->n)
	{
	  printf("\n\t******sacma in adjustin*******");
	}
#endif
      memcpy(&(xe->val[myst]), x->val, sizeof(double)*x->sz);
      xe->sz = A->n;

    } 
  else 
    {
      xe->val = (double*) malloc(sizeof(double)*A->n);
      xe->sz = A->n;
    }
  return;
}

/**********************************************************************/
/*        x is of size A->n.                                          */
/*        y is of size A->m. upon exit y is of size myportion.         */
/**********************************************************************/
void mxv_post(buMatrix *Aloc,
	      buMatrix *Acpl,
	      buVector *x,
	      buVector *y, int mxvTag,  MPI_Comm  libComm)
/*upon exitting, y->val points the the first y value of my portion*/
{

  int i;
  int myId;
  int numProcs;
  int recved, recvInd;
  MPI_Request *sendReqs;
  MPI_Request *recvReqs;
  MPI_Status *stts; 

  commHandle *send = y->cm->send;
  commHandle *recv = y->cm->recv;

  MPI_Comm_size(libComm, &numProcs);
  MPI_Comm_rank(libComm, &myId);

  if(y->sz != Aloc->m)
    {
      allocVectorData(y, Aloc->m);
    }

#ifdef DEBUG
  if(Aloc->store != STORE_BY_ROWS)
    {
      printf("\nAloc is not stored rowwise in mxv_post");
      MPI_Finalize();
      exit(1890);
    }
  if(Acpl->store != STORE_BY_ROWS)
    {
      printf("\nAcpl is not stored rowwise in mxv_post");
      MPI_Finalize();
      exit(1891);
    }
  if(y->sz != Aloc->m)
    {
      printf("\n y and Aloc is not of the same dimension in mxv_post %d vs %d", y->sz, Aloc->m);
      MPI_Finalize();
      exit(1892);
    }

  if(y->sz != Acpl->m)
    {
      printf("\n y and Acpl is not of the same dimension in mxv_post");
      MPI_Finalize();
      exit(1893);
    }
  if(x->sz != Aloc->n)
    {
      printf("\n x and Aloc is not of the same dimension in mxv_post");
      MPI_Finalize();
      exit(1894);
    }
  if(x->sz != Acpl->n)
    {
      printf("\n x and Acpl is not of the same dimension in mxv_post");
      MPI_Finalize();
      exit(1895);
    }
#endif

  stts = (MPI_Status *) malloc(sizeof(MPI_Status)*numProcs);
  sendReqs = (MPI_Request*) malloc(sizeof(MPI_Request)*numProcs);
  recvReqs = (MPI_Request*) malloc(sizeof(MPI_Request)*numProcs);
  
  for( i = 0 ; i < y->sz; i ++)
    y->val[i]  = 0.0;

  /* 2.
   * issue Irecv on y values.*/
  for( i = 0 ; i < recv->num; i++)
    {
      int p = recv->lst[i];
      int ysz = recv->all[p+1] - recv->all[p];
      int rind = recv->all[p];
#ifdef DEBUG
      if(ysz <= 0)
	{
	  printf("\n I am %d going to receive %d DOUBLES FROM %d", myId, ysz, p);
	  MPI_Finalize();
	  exit(675);
	}
#endif
#ifdef DEBUG_2
      printf("\n%d receiving from %d sz %d, st %d, lim %d",myId, p, ysz, rind, recv->all[numProcs]); 
#endif
#ifdef DO_COMM
      MPI_Irecv(&(recv->val[rind]), ysz, MPI_DOUBLE, p, mxvTag, libComm, &(recvReqs[i]));
#endif
    }

  // KAMER
  /* 1.
   * do Acpl multiplication, Isend at each processor boundary.*/
  double *pval = Acpl->val, *pxval = x->val, *pyval = y->val, val, *ppval;
  int *pia = Acpl->ia, *pja = Acpl->ja, *ppja, 
      *slst = send->lst, *sall = send->all, *sind = send->ind, 
      snum = send->num, r, jj, jjs, p, ysz, j, js, yst, *psind;
 
  for(i = 0; i < snum; i++) {
      p = slst[i];
      ysz = sall[p+1] - sall[p];
      js = sall[p]; 
      yst = sind[js];
      psind = sind + js;
      for(j = sall[p+1] - js; j; j--) {
	  val = 0.0;

	  r = *psind++;
	  jjs = pia[r];
	  ppval = pval + jjs;
	  ppja = pja + jjs; 

	  for(jj = pia[r+1] - jjs; jj; jj--) {
	      val += pxval[*ppja++] * *ppval++;
	  }
	  pyval[r] = val;
      }
#ifdef DEBUG_2
      printf("\n%d sending %d sz %d, st %d, lim %d", myId, p, ysz, yst, y->sz);
#endif     
#ifdef DO_COMM     
      MPI_Isend(&(y->val[yst]), ysz, MPI_DOUBLE, p, mxvTag, libComm, &(sendReqs[i]));
#endif
  }
        	
#ifdef DEBUG_2
  MPI_Barrier(libComm);
  if(myId == 0)
    printf("\nRweached local comp");
#endif

  /* 3.
   * Aloc multiplication.*/  
  // KAMER
  int m = Aloc->m, *ia = Aloc->ia, *ja = Aloc->ja, je;
  double *px = x->val, *py = y->val, *pv = Aloc->val;
  for(i = 0; i < m; i++) {
    val = 0.0;
    for (j = ia[i+1] - ia[i]; j ; j--) {
      val += *pv++ * px[*ja++];
    }
    py[i] += val;
  }
  
#ifdef DEBUG_2
  MPI_Barrier(libComm);
  if(myId == 0)
    printf("\nCompleted local comp");
#endif

  /* 4.
   * recv y values and add them up.*/ 
#ifdef DO_COMM
#ifdef WAIT_RECEIVES
  recved = 0;
  recvInd = -1000;
  while(recved < recv->num)
    {
      MPI_Waitany(recv->num, recvReqs, &recvInd, &stts[0]);
#ifdef DEBUG
      if(recvInd == MPI_UNDEFINED)
	{
	  if(recved < recv->num)
	    {
	      printf("\nolmuyor recv in mxv_pre");
	      MPI_Finalize();
	      exit(1980);
	    }
	}
#endif
      if(recvInd != MPI_UNDEFINED)
	{
	  int p = stts[0].MPI_SOURCE;
	  int sz = recv->all[p+1] - recv->all[p]; 
	  int st = recv->all[p];
	  int ii, ie = recv->all[p+1];
#ifdef DEBUG
	  if(p != recv->lst[recvInd])
	    {
	      printf("\nin mxv_post these are not equal???");
	      MPI_Finalize();
	      exit(1198);
	    }
#endif
	  recved ++;
	  recvReqs[recvInd] = MPI_REQUEST_NULL;
	  /*retrieved y from processor p. handle it*/
	  for(ii = st ; ii < ie; ii++)
	    {
	      int ind = recv->ind[ii];
#ifdef DEBUG
	      if( (ind < 0) || (ind >=y->sz))
		{
		  printf("\ninteresting ind mxv_post");
		  MPI_Finalize();
		  exit(18);
		}
#endif
	      y->val[ind] += recv->val[ii];	     
	    }
	}
    }
#else
  MPI_Barrier(libComm);

  if(recv->num > 0)
    {
      MPI_Waitall(recv->num, recvReqs, stts);
    }
#endif
#endif

#ifdef DO_COMM

  if(send->num > 0)
    MPI_Waitall(send->num, sendReqs, stts);
#endif

  /*re adjust y->val and y->sz according to outpart*/
  adjustOut(Aloc, y, libComm);

  free(recvReqs);
  free(sendReqs);
  free(stts);
  return;
}

/**********************************************************************/
/*
 * getting my portion from columnwise y=Ax;
 */

void adjustOut(buMatrix *A, buVector *y, MPI_Comm libComm)
{
  /*used in mxv_post*/
  int i; 
  int myId, numProcs;
  int myst;

  double *newVal;
  int index = 0;
  int sz;
  int higherProcsStart;
  commHandle *send = y->cm->send; 

  MPI_Comm_rank(libComm, &myId);
  MPI_Comm_size(libComm, &numProcs);
  myst = send->all[myId];

#ifdef DEBUG
  if(myst < 0)
    {
      printf("\ninteresting myst in adjustOut");
      MPI_Finalize();
      exit(197);
    }
  else
    if (myst > 0)
      {
	int ind = send->all[myId]-1;
	int whichy = send->ind[ind] ;
	if(ind != myst - 1)
	  {
	    printf("\ninteresting myst in adjustOut");
	    MPI_Finalize();
	    exit(197);
	  }
      }

#endif
  /* among the results generated those that are not sent to any processor plus some extar is mine */
  sz = A->m - send->all[numProcs];
  y->sz = sz;

  /*shift my result to the beginning of y*/
  // KAMER
  if(sz > 0 && myst != 0) {
    memmove(y->val, y->val + myst, sizeof(double) * sz);
    //  newVal = (double*) malloc(sizeof(double)*sz);
    //	memcpy(newVal, & (y->val[myst]), sizeof(double)*sz);
    /*  y is always larger than sz  */
    //	memcpy(y->val, newVal, sizeof(double)*y->sz);
    //	free(newVal);
  }
  return;
} 

