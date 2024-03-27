#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmxv.h"

//#define DEBUG


double dotv_div_dotv(buVector *v1, buVector *w1, buVector *v2, buVector *w2,
		     MPI_Comm parentComm)
{
 double d1 = 0.0 , d2=0.0;  
 double dots[2] = {0.0, 0.0};
 double gdots[2]= {0.0, 0.0};
 int i;
 MPI_Comm  libComm;

 getLibComm(parentComm, &libComm);
 /*get myID if necessary...

   MPI_Comm_rank(libComm, &myId); 
 */
#ifdef DEBUG
 if(v1->sz != w1->sz)
     {
     printf("\nfirst pair is not of the same size, in_dotv/dotv");
     MPI_Finalize();
     exit(1700);
     }
 if(v2->sz != w2->sz)
     {
     printf("\nsecond pair is not of the same size, in_dotv/dotv");
     MPI_Finalize();
     exit(1700);
     }
#endif
 
 for( i = 0; i <v1->sz; i++)
     {
     d1 += v1->val[i] * w1->val[i];
     }
 dots[0] = d1;
 for( i = 0; i <v2->sz; i++)
     {
     d2 += v2->val[i] * w2->val[i];
     }
 dots[1] = d2;
 
 MPI_Allreduce(dots, gdots, 2, MPI_DOUBLE, MPI_SUM, libComm);

 dots[0] = gdots[0];
 dots[1] = gdots[1];
 return (dots[0]/dots[1]);
 

}

void scv(buVector *v, double c, buVector *w)
{
 int i, n=v->sz;
#ifdef DEBUG
 if(v->sz != w->sz)
     {
     printf("\nv->sz and w->sz is not of the same size in scv");
     MPI_Finalize();
     exit(1702);
     }
#endif
 for( i = 0; i < n; i++ )
     {
     w->val[i] = c * v->val[i];
     }
 
 return;
}

double dotv(buVector *v, buVector *w, MPI_Comm parentComm)
{


#ifdef DEBUG
 if(v->sz != w->sz)
     {
     printf("\nv->sz and w->sz is not of the same size");
     MPI_Finalize();
     exit(1700);
     }
#endif
 return(dot(v->val, w->val, v->sz, parentComm));

}

double dotLcl(buVector *v, buVector *w)
{
 int i;
 double dot = 0.0;
 int n = v->sz;
#ifdef DEBUG
 if(v->sz != w->sz)
     {
     printf("\nv and w are not of the same size, in dotLcl");
     MPI_Finalize();
     exit(1701);
     }
#endif

 for( i = 0; i < n; i++)
     {
     dot += v->val[i] * w->val[i];
     }


 return dot;
}

double dot(double *v, double *w, int n, MPI_Comm parentComm)
{
 int i;
 double dot = 0.0;
 double gdot = 0.0;
 MPI_Comm  libComm;

 getLibComm(parentComm, &libComm);
 /*get myID if necessary...

   MPI_Comm_rank(libComm, &myId); 
 */
 
 for( i = 0; i < n ; i++)
     dot += v[i] * w[i];
 
 
 MPI_Allreduce(&dot, &gdot, 1, MPI_DOUBLE, MPI_SUM, libComm);
 
 dot = gdot;
 return (dot);
}

/**********************************************************************/
void freeVector(buVector *v)
{
 if( v ) 
     {
     if( v->val ) 
	 free(v->val);
     free(v);
     }
 return;

}

void allocVectorData(buVector *v, int sz)
{
 
 if(v->sz != 0)
     {
     if(v->val != NULL)
	 free(v->val);
     }
 v->sz = sz; 
 
 v->val = (double *) malloc(sizeof(double)*sz);
 if(!v->val)
     {
     printf("\nAllocation failed for vector->val");
     MPI_Finalize();
     exit(1878);
     }
 
 return;
}
buVector *allocVector(int n)
{
 buVector *v; 
 int i;

 v = (buVector *) malloc(sizeof(buVector));
 if(!v)
     {
     printf("\nAllocation failed for vector");
     MPI_Finalize();
     exit(1878);
     }

 v->sz = n;
 v->val = (double *) malloc(sizeof(double)*n);
 if(!v->val)
     {
     printf("\nAllocation failed for vector->val");
     MPI_Finalize();
     exit(1878);
     }

 for( i = 0 ; i < n ; i++)
     v->val[i] = 0.0;
 return(v);
}
/**********************************************************************/
void printIntVec(int *vec, int sz)
{
 int i;
 printf("\n");
 for( i = 0 ; i < sz; i++)
     printf(" %d", vec[i]);
 printf("\n");
}
void printDblVec(double *vec, int sz)
{
 int i;
 printf("\n");
 for( i = 0 ; i < sz; i++)
     printf(" %lf", vec[i]);
 printf("\n");
}

int isAllZero(buVector *v)
{

 int i, n=v->sz;
 int s = 0;

 int isZero;
 
 double myEps = 1.0e-6;
 for( i = 0; i < n; i++)
     s += ( (v->val[i] < 0.0 + myEps) && ((v->val[i] > 0.0 - myEps))) ? 0 : 1;
 
 isZero = !s;
 return ( isZero );

}

/**********************************************************************/
void v_gets_v_plus_cw(buVector *v, buVector *w, double c)
{
 int i;
 int n=v->sz;
#ifdef DEBUG
 if(v->sz != w->sz)
     {
     printf("\nv and w are not of the same size in v_get_v_plus_cw");
     MPI_Finalize();
     exit(1989);
     }
#endif
 for( i = 0 ; i < n ; i++) 
      v->val[i] = v->val[i] + c * w->val[i];
 return;

}

/**********************************************************************/
/* z[i] = v[i] + c w[i];
 * sets z size;
 */
void v_plus_cw(buVector *v, buVector *w, double c, buVector *z)
{

 int i; 

 int n = v->sz;
#ifdef DEBUG
 if(v->sz != w->sz)
     {
     printf("\nv and w are not of the same size in v_plus_cw");
     MPI_Finalize();
     exit(1989);
     }
 
#endif

  for( i = 0 ; i < n ; i++) 
      z->val[i] = v->val[i] + c*w->val[i];

  z->sz = n;

  return;
}

/***********************************************************************/

void vcopy_vv(buVector *from, buVector *to)
{
#ifdef DEBUG
 if(from->sz != to->sz)
     {
     printf("\nfrom->sz(%d) != to->sz in vcopy_vv (%d)", from->sz, to->sz);
     MPI_Finalize();
     exit(1500);
     }
#endif
 memcpy(to->val, from->val, sizeof(double)*from->sz);

 return;
}
