#include "partition.h"
#include "common.h"
#include "mmio.h"
#include "patoh.h"
#include "stdio.h"
#include <stdlib.h>
#include <time.h> 

CSC COO_to_CSC(COO *in) {
  CSC out = {.I = (int *)malloc(sizeof(int) * in->nnz),
             .J = (int *)malloc(sizeof(int) * (in->n + 1)),
             .val = (double *)malloc(sizeof(double) * in->nnz),
             .nnz = in->nnz,
             .m = in->m,
             .n = in->n};
  if (out.I == NULL || out.J == NULL || out.val == NULL) {
    return (CSC){0};
  }
  out.J[0] = 0;
  out.J[out.n] = out.nnz;
  int i, col;
  for (i = 0, col = 0; i < out.nnz; i++) {
    if (in->J[i] != col)
      out.J[col++ + 1] = i;
    out.I[i] = in->I[i];
    out.val[i] = in->val[i];
  }
  int last = out.J[out.n];
  if (last != out.nnz) {
    out.J[out.n] = out.nnz;
  }
  return out;
}

CSC ReadSparseMatrix(char *fname) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int M, N, nz;

  if ((f = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "Could not open matrix file\n");
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  if (mm_is_complex(matcode) | !mm_is_matrix(matcode) | !mm_is_sparse(matcode) |
      mm_is_symmetric(matcode)) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
    exit(1);
  }

  /* reseve memory for matrices */

  COO smatrix = {.I = (int *)malloc(nz * sizeof(int)),
                 .J = (int *)malloc(nz * sizeof(int)),
                 .val = (double *)malloc(nz * sizeof(double)),
                 .nnz = nz,
                 .m = M,
                 .n = N};

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (int i = 0; i < nz; i++) {
    if (fscanf(f, "%d %d %lg\n", &smatrix.I[i], &smatrix.J[i],
               &smatrix.val[i]) != 3) {
      fprintf(stderr, "Error reading matrix\n");
      exit(EXIT_FAILURE);
    }
    smatrix.I[i]--; /* adjust from 1-based to 0-based */
    smatrix.J[i]--;
  }
  fclose(f);

  CSC cscmatrix = {0};
  if ((cscmatrix = COO_to_CSC(&smatrix)).nnz == 0)
    exit(EXIT_FAILURE);
  freeSparseMatrix(&smatrix);

  return cscmatrix;
}

static void nnzDistribution(int *cweights, int numpart, int nrow, int *partvec,char* resultFName) {

  int nnzCount[numpart];
  for (int i = 0; i < numpart; i++)
    nnzCount[i] = 0;

  for (int i = 0; i < nrow; i++) {

    nnzCount[partvec[i]] += cweights[i];
  }

 //-----------------------Result file ----------------------------------   
  FILE* fptr = fopen(resultFName,"a") ; 

  if (fptr == NULL) {
     printf("Result file cannot opened!!") ;
     exit(-1) ;
  }
  else  {

    time_t mytime;
    mytime = time(NULL) ;
    
    fprintf(fptr,"%s\n",ctime(&mytime)) ;
    fprintf(fptr,"Number of Non-zero Count : \n") ;
    for (int i = 0; i < numpart; i++) {
        printf("[%d] %d ", i, nnzCount[i]);
        fprintf(fptr," [%d] : %d ",i,nnzCount[i]) ;
    }
  }
  fclose(fptr) ;
  //--------------------- Ending Result file -----------------------------
  printf("\n");
}

int *CalcPartVec(int nparts, CSC *cscmatrix,char * fName, double final_imbal, int seed , char* resultFName) {

  CSR csrmatrix = {};
  if ((csrmatrix = CSC_to_CSR(cscmatrix)).nnz == 0)
    exit(EXIT_FAILURE);

  CSC csctmatrix = SparseTranspose(*cscmatrix); /* test needed */
  int *cweights = CalcWeights(&csctmatrix);

  // PATOH STARTS HERE
  // =========================================================================================================

  int nweights[csrmatrix.n];
  for (int i = 0; i < csrmatrix.n; i++)
    nweights[i] = 1;

  PaToH_Parameters args = {0};
  PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);
  args._k = nparts;
  
  // =========================================================================================================
  // Final imbalance and counstant seed  
  args.final_imbal = final_imbal ; 
  args.seed = seed ;
  // =========================================================================================================

  int *partvec =
      malloc(sizeof(int) * csrmatrix.n); // contains the resulting partition
  int partweights[args._k]; // contains the weights of the partition
  int cut;

  float targetweigths[nparts] ; 


  // Open a file in read mode
  FILE *fptr;
  if((fptr = fopen(fName, "r")) == NULL) {
    fprintf(stderr, "file can't be opened %s file \n",fName) ;
    exit(-1) ;
  }
  
  int i=0 ;
  while (fscanf(fptr, "%f ",targetweigths + i++)  == 1) ;
        
  fclose(fptr) ;
  
  
  PaToH_Alloc(&args, csrmatrix.n, csrmatrix.m, 1, cweights, nweights,
              csrmatrix.I, csrmatrix.J);

  PaToH_Part(&args, csrmatrix.n, csrmatrix.m, 1, 0, cweights, nweights,
             csrmatrix.I, csrmatrix.J, targetweigths, partvec, partweights,
             &cut);

  nnzDistribution(cweights, nparts, cscmatrix->m, partvec,resultFName); // print nnz per part 
                                                        
  
  
  printf("cut : %d",cut) ;
  
  // --------------------------- open result file for write cut size -----------------------
   FILE* file = fopen(resultFName,"a") ; 

  if (file == NULL) {
     printf("Result file cannot opened!!") ;
     exit(-1) ;
  }
  else  {
    fprintf(file,"\n\ncut  : %d \n",cut) ;
  }
  fclose(file) ; 
   
  // --------------------------- closing file ---------------------------------------------- 




  free(cweights);
  freeSparseMatrix(&csrmatrix);
  PaToH_Free();
  // PATOH ENDS HERE
  // ============================================================================================
  return partvec;
}
