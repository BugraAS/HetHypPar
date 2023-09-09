#pragma once

#include "mpi.h"

typedef enum {
  STORE_BY_COLUMNS,
  STORE_BY_ROWS,
  PART_BY_COLUMNS,
  PART_BY_ROWS,
  PART_2D,
  PART_CHK,
  POSTCOMM,
  PRECOMM,
  PREPOSTCMM
} spmxv_const;


typedef struct{
  int *ia; 
  int *ja;
  double *val;
  int n;        /*  number of cols, independent of storage scheme*/
  int m;        /*  number of rows, independent of storage scheme*/
  int nnz;
  int gn, gm;   /*  global m n                  */
  int store;    /*  either BY_COLUMNS or BY_ROWS*/
  int *inPart;  /*  size n                      */
  int *outPart; /*  size m                      */
} buMatrix;
/*sparse m-by-n matrix with nnz nonzeros.*/


typedef struct{
  double *val; /*sometimes we will use this place to recv.
		 sometimes we will perform inplace communication*/ 
  int *ind;
  int *all;
  int *lst;
  int num;
}commHandle;

typedef struct{
  commHandle *send;
  commHandle *recv;
}comm;

typedef struct{

  /* This contains the local part of the matrix
   */
  buMatrix *loc;
  buMatrix *cpl;
  comm *in;
  comm *out;
  int scheme ;
}parMatrix;




/*
  num shows the number of processors that are in contact (recv/send) with this.
  lst[0,..num-1], ranks of procs that are in comm list.
  all[0 .. numProcs] ;like ia of sparse matrices
  ind[0..numCommItems] like ja of sparse matrices
  val[0..numCommItems] like a of sparse matrices
 */

typedef struct{
  int sz;
  double *val;
  comm *cm;
} buVector;


/* macros:
         start
*/

#define IA_SIZE(A) ( ( (A)->store == STORE_BY_ROWS ) ? ( (A)->m ) : ( (A)->n ) )

#define IA_SIZE_INC(A)       \
   if ( (A)->store == STORE_BY_ROWS )  (A)->m = (A)->m + 1 ;\
   else (A)->n = (A)->n + 1

#define OTHER_SIZE_INC(A, _sz)     \
   if ( (A)->store == STORE_BY_ROWS )  (A)->n = (A)->n + _sz ;\
   else (A)->m = (A)->m + _sz



/* macros:
         END
*/



void mxv(parMatrix *A, buVector *x, buVector *y, MPI_Comm parentComm);

double normv(buVector *v);
double norm(double *v, int n);

void scv(buVector *v, double c, buVector *w);

double dotv_div_dotv(buVector *v1, buVector *w1, buVector *v2, buVector *w2, MPI_Comm parentComm);

double dotv(buVector *v, buVector *w,MPI_Comm parentComm);
double dot(double *v, double *w, int n, MPI_Comm parentComm);

double dotLcl(buVector *v, buVector *w);

void scv(buVector *v, double c, buVector *w);

void setup(int mxCnt, int* commSteps, buMatrix **matrixChain, buMatrix **locs, buMatrix **cpls, comm **ins, comm **outs, MPI_Comm parentComm);

/**
 * int partScheme : Can be either PART\_BY\_ROWS or PART\_BY\_COLUMNS.
 * buMatrix *mtrx : Matrix to be split. It needs to be in master node.
 * buMatrix *loc  : Local part of the matrix that doesn't haev outside dependencies. I think.
 * buMatrix *cpl  : Local part of the matrix WITH outside dependencies. I think.
 * renamed first argument to partScheme from commStep. Inconsistent naming with defintion.
 */
void setupMisG(int partScheme, buMatrix *mtrx, buMatrix *loc, buMatrix *cpl, comm *ins, comm *outs, MPI_Comm parentComm);

void switchStore(buMatrix *A, buMatrix *AT);
void printSendRecvs(comm *in, int who);
void copyMatrix(buMatrix *from, buMatrix *to);
void copyCreateMatrix(buMatrix *from, buMatrix *to);
void printMatrix(buMatrix *A);
void printIntVec(int *vec, int sz);

void allocMatrix(buMatrix *A);
void freeMatrix(buMatrix *A);

int isAllZero(buVector *v);
void v_gets_v_plus_cw(buVector *v, buVector *w, double c);

void v_plus_cw(buVector *v, buVector *w, double c, buVector *z);
buVector *allocVector(int n);
void freeVector(buVector *v);

void allocVectorData(buVector *v, int sz);
void adjustOut(buMatrix *A, buVector *y, MPI_Comm libComm);
void adjustIn(buMatrix *A, buVector *x, buVector *xe, MPI_Comm libComm);

void initParLib(MPI_Comm parentComm);
void quitParLib(MPI_Comm parentComm);
void getLibComm(MPI_Comm  parentComm, MPI_Comm *libComm);

void freeComm(comm *cm);
void getTagMult(MPI_Comm parentComm, int *tagMult);
void incrementTagMult(MPI_Comm  parentComm);

comm *allocComm();
commHandle *allocCommHandle();
void inverseComm(comm *from, comm *to, MPI_Comm parentComm);
void readMatrixCoordsFromFile(int *rowCnt, int *colCnt, int *nnzCnt,
			      int **rows, int **cols, double **vals,
			      char *fname);
void readMatrixCoordinates(int **rowIndices, int **colIndices, double **val, int *nnz, 			   int *gm, int *gn, int **rpv, int **cpv,
			   char *fname,
			   char *fnameIn,
			   char *fnameOut,
			   char *fnameNnz,
			   MPI_Comm parentComm, int partScheme);

void readMatrix(buMatrix *A, char *fname, int partScheme, char *xpartfname, char *ypartfname);

void setup2D(int *rowIndices, int *colIndices, double *val, int indexCnt, /*coordinate format*/
	     buMatrix *mtrx, buMatrix *loc, buMatrix *cpl,
	     comm *ins, comm *outs,
	     MPI_Comm parentComm	     
	    );
