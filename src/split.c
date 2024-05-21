#include "split.h"
#include "common.h"
#include "util.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>


//Taken From G2G
// A iterative binary search function. It returns location
// of x in given array arr[l..r] if present, otherwise -1
int binarySearch(int arr[], int l, int r, int x)
{
    // the loop will run till there are elements in the
    // subarray as l > r means that there are no elements to
    // consider in the given subarray
    while (l <= r) {
 
        // calculating mid point
        int m = l + (r - l) / 2;
 
        // Check if x is present at mid
        if (arr[m] == x) {
            return m;
        }
 
        // If x greater than ,, ignore left half
        if (arr[m] < x) {
            l = m + 1;
        }
 
        // If x is smaller than m, ignore right half
        else {
            r = m - 1;
        }
    }
 
    // if we reach here, then element was not present
    return -1;
}


static inline uint32_t* boolArray_new(int nItems){
  uint32_t* res;
  int nelems = ceil(nItems/32.0);
  CALLOC_ARRAY(res, nelems);
  return res;
}

static inline int boolArray_get(uint32_t* array, int index){
  return (array[index/32]) & (1 << (index % 32));
}

static inline void boolArray_set(uint32_t* array, int index, int val){
  uint32_t mask = -1;
  mask ^= 1 << (index % 32);
  array[index/32] &= mask;
  array[index/32] |= (val) << (index % 32);
}

// find ~a1 & a2
// store result into a2
static inline void boolArray_diff(uint32_t* a1, uint32_t* a2, int nItems){
  int nelems = ceil(nItems/32.0);
  for (size_t i = 0; i < nelems; i++)
  {
    a2[i] &= ~(a1[i]);
  }
}

static inline int boolArray_isEmpty(uint32_t* array, int nItems){
  int flag = 1;
  int nelems = ceil(nItems/32.0);
  for (size_t i = 0; i < nelems; i++)
  {
    flag &= !array[i];
  }
  return flag;
}

// size is an output
static inline int* boolArray_toIndexList(uint32_t* array, int nItems, int* size){
  int count = 0;
  int nelems = ceil(nItems/32.0);
  for (size_t i = 0; i < nelems; i++)
  {
    uint32_t val = array[i];
    if(!val) continue;
    for (size_t j = 0; j < 32; j++)
    {
      if((val) & (1 << (j))) count++;
    }
  }

  *size = count;

  int* res = NULL;

  //early exit
  if(count == 0){
    return res;
  }

  ALLOC_ARRAY(res, count);
  count = 0;
  for (size_t i = 0; i < nelems; i++)
  {
    uint32_t val = array[i];
    if(!val) continue;
    for (size_t j = 0; j < 32; j++)
    {
      if((val) & (1 << (j))) res[count++] = j + i*32;
    }
  }
  
  return res;
}

// Assume Idx always has a match in mp
// Note for future: binary searchh is viable here
static inline int findPermIdx(int* mp, int mp_size, int Idx ){
  int res = binarySearch(mp, 0, mp_size-1, Idx);
  // for (size_t i = 0; i < mp_size; i++)
  // {
  //   if(mp[i] == Idx)
  //     return i;
  // }
  // ABORT("Idx: %d has no counterpart in mp", Idx) // 414 not found in 0
  // DEBUGLOG("Idx: %d has no counterpart in mp", Idx)
  // TODO: Enable line below
  return res;
}

static inline void fillLocal(CSR* loc, int* locp, const CSR big){
  for (size_t i = 0; i < loc->m; i++){
    int loci = loc->I[i];
    int bigi = big.I[locp[i]];
    int ncols = big.I[locp[i]+1] - bigi; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++)
    {
      int Idx = findPermIdx(locp, loc->m, big.J[bigi+ j]);
      if (Idx == -1) ABORT("Idx: %d has no counterpart in mp", Idx)
      loc->J[loci + j] = Idx;
      loc->val[loci + j] = big.val[bigi + j];
    }
  }
}

// new function that takes a completely empty loc (except loc.m)
static inline void fillPart(CSR* part, int* perm, uint32_t* mask_loc, const CSR big){

  CALLOC_ARRAY(part->I, part->m + 1);
  part->nnz = 0;
  
  // dry run to fill loc.I and count nnz
  for (size_t i = 0; i < part->m; i++){
    int loci = part->I[i];

    int bigi = perm[i];
    int bigj = big.I[bigi];
    int ncols = big.I[perm[i]+1] - bigj; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++)
    {
      if(!boolArray_get(mask_loc, big.J[bigj + j])) continue;

      part->nnz++;
      loci++;

      // int Idx = findPermIdx(perm, part->m, big.J[bigj+ j]);
      // if (Idx == -1) ABORT("Idx: %d has no counterpart in mp", Idx)
      // loc->J[loci + j] = Idx;
      // loc->val[loci + j] = big.val[bigj + j];
    }
    part->I[i+1] = loci;
  }

  if(part->I[part->m] != part->nnz)
    ABORT("Number of non-zeroes (%d) in part doesnt equal last element (%d) in compressed index vector", part->nnz, part->I[part->m])

  ALLOC_ARRAY(part->J, part->nnz);
  ALLOC_ARRAY(part->val, part->nnz);

  int count = 0;

  for (size_t i = 0; i < part->m; i++)
  {
    int loci = part->I[i];

    int bigi = perm[i];
    int bigj = big.I[bigi];
    int ncols = big.I[perm[i]+1] - bigj; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++)
    {
      if(!boolArray_get(mask_loc, big.J[bigj + j])) continue;
      int Idx = findPermIdx(perm, part->m, big.J[bigj+ j]);
      if (Idx == -1) ABORT("Idx: %d has no counterpart in mp", Idx)
    
      part->J[count] = Idx;
      part->val[count] = big.val[bigj + j];
      count++;
    }
  }

  if(count != part->nnz) ABORT("Count (%d) does not match nnz (%d)", count, part->nnz)
}

//Simplifying assumptions:
// - Matrix is split into 2 pieces. (CPU GPU)
// - shared matrix is empty (cutsize is 0)
// - split is row wise
// - partition is same on rows as columns
SPLIT_CSR* cleanSplit(CSR big, int* partvec){

  SPLIT_CSR* res;
  ALLOC_ARRAY(res, 2);
  res[0] = (SPLIT_CSR){};  // GPU 
  res[1] = (SPLIT_CSR){};  // CPU 

  /*
   * Things to do:
   *  - Count how many rows and non-zeros are owned by each CPU-GPU.
   *  - Calculate the "transpose" of partvec.
   */

  for( size_t i=0; i < big.m; i++){
    int proc = partvec[i];
    res[proc].loc.m++;
    // res[proc].loc.n++;
    res[proc].loc.nnz += (big.I[i+1] - big.I[i]);   //  CSR matrix file  
  }
  ALLOC_ARRAY(res[0].locp, res[0].loc.m);  // loc p permutation index 
  ALLOC_ARRAY(res[1].locp, res[1].loc.m);

  ALLOC_ARRAY(res[0].loc.I, res[0].loc.m + 1);
  ALLOC_ARRAY(res[0].loc.J, res[0].loc.nnz);
  ALLOC_ARRAY(res[0].loc.val, res[0].loc.nnz);
  ALLOC_ARRAY(res[1].loc.I, res[1].loc.m + 1);
  ALLOC_ARRAY(res[1].loc.J, res[1].loc.nnz);
  ALLOC_ARRAY(res[1].loc.val, res[1].loc.nnz);

  {
    res[0].loc.I[0] = 0;   // 
    res[1].loc.I[0] = 0;
    for( size_t i=0; i < big.m; i++){
      int proc = partvec[i];
      int* n = &(res[proc].loc.n);
      int* I = res[proc].loc.I;
      res[proc].locp[(*n)++] = i;
      I[*n] =  (big.I[i+1] - big.I[i]) + I[*n -1];
    }
  }

  if (res[0].loc.m != res[0].loc.n)
    ABORT(
      "split sanity check failed: id:%d, loc.m:%d, loc.n:%d",
      0, res[0].loc.m, res[0].loc.n)

  if (res[1].loc.m != res[1].loc.n)
    ABORT(
      "split sanity check failed: id:%d, loc.m:%d, loc.n:%d",
      1, res[1].loc.m, res[1].loc.n)

  fillLocal( &(res[0].loc), res[0].locp, big);
  fillLocal( &(res[1].loc), res[1].locp, big);

  return res;
}

//Simplifying assumptions:
// - big matrix is square
// - Matrix is split into 2 pieces. (CPU GPU)
// - split is row wise
// - partition is same on rows as columns
SPLIT_CSR* sparseSplit(CSR big, int* partvec){

  SPLIT_CSR* res;
  ALLOC_ARRAY(res, 2);
  res[0] = (SPLIT_CSR){};  // GPU 
  res[1] = (SPLIT_CSR){};  // CPU

  uint32_t* mask_loc[2] = {
    boolArray_new(big.n),
    boolArray_new(big.n)
  };

  for( size_t i=0; i < big.m; i++){
    int proc = partvec[i];
    boolArray_set(mask_loc[proc], i, 1);
    res[proc].loc.m++;
  }

  // Potential sanity check: mmask_loc[0] and mask_loc[1] should be opposites

  res[0].locp = boolArray_toIndexList(mask_loc[0], big.n, &(res[0].loc.n));
  res[1].locp = boolArray_toIndexList(mask_loc[1], big.n, &(res[1].loc.n));

  if(res[0].loc.n != res[0].loc.m)
    ABORT(
    "Sanity check failed on matrix split. Number of set booleans (%d) is different than the number of rows (%d) on 0",
    res[0].loc.n, res[0].loc.m
  );
  if(res[1].loc.n != res[1].loc.m)
    ABORT(
    "Sanity check failed on matrix split. Number of set booleans (%d) is different than the number of rows (%d) on 1",
    res[1].loc.n, res[1].loc.m
  );

  uint32_t* mask_shr[2] = {
    boolArray_new(big.n),
    boolArray_new(big.n)
  };

  for (size_t i = 0; i < big.m; i++)
  {
    int proc = partvec[i];
    for (size_t j = big.I[i]; j < big.I[i+1]; j++)
      boolArray_set(mask_shr[proc], big.J[j], 1);
  }
  boolArray_diff(mask_loc[0], mask_shr[0], big.m);
  boolArray_diff(mask_loc[1], mask_shr[1], big.m);

  res[0].shrp = boolArray_toIndexList(mask_shr[0], big.n, &(res[0].shr.n));
  res[0].shr.m = res[0].loc.m;
  res[1].shrp = boolArray_toIndexList(mask_shr[1], big.n, &(res[1].shr.n));
  res[1].shr.m = res[1].loc.m;

  fillPart(&(res[0].loc), res[0].locp, mask_loc[0], big);
  if(res[0].shr.n != 0)
    fillPart(&(res[0].shr), res[0].shrp, mask_shr[0], big);

  fillPart(&(res[1].loc), res[1].locp, mask_loc[1], big);
  if(res[1].shr.n != 0)
    fillPart(&(res[1].shr), res[1].shrp, mask_shr[1], big);

  FREE_AND_NULL(mask_loc[0]);
  FREE_AND_NULL(mask_loc[1]);

  FREE_AND_NULL(mask_shr[0]);
  FREE_AND_NULL(mask_shr[1]);

  return res;
}