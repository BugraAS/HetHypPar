#pragma once

#include <stdio.h>
#include <stdlib.h>

#define ABORT(message, args...)                                                \
  {                                                                            \
    fprintf(stderr, "ERROR: " message "\n", ##args);                           \
    exit(EXIT_FAILURE);                                                        \
  }

#ifndef NDEBUG
#define DEBUGLOG(message, args...)                                             \
  { fprintf(stderr, "DEBUG: " message "\n", ##args); }
#else
#define DEBUGLOG(message, args...)
#endif // !NDEBUG

// ==============================================================
// Below code is taken from git source code
// ==============================================================

static inline size_t st_mult(size_t a, size_t b) {
  // ignore safety - bugra
  // if (unsigned_mult_overflows(a, b))
  // 	die("size_t overflow: %"PRIuMAX" * %"PRIuMAX,
  // 	    (uintmax_t)a, (uintmax_t)b);
  return a * b;
}

#define FREE_AND_NULL(p) do { free(p); (p) = NULL; } while (0)

#define ALLOC_ARRAY(x, alloc) (x) = malloc(st_mult(sizeof(*(x)), (alloc)))
#define CALLOC_ARRAY(x, alloc) (x) = calloc((alloc), sizeof(*(x)))
#define REALLOC_ARRAY(x, alloc)                                                \
  (x) = realloc((x), st_mult(sizeof(*(x)), (alloc)))

#define alloc_nr(x) (((x) + 16) * 3 / 2)

/**
 * Dynamically growing an array using realloc() is error prone and boring.
 *
 * Define your array with:
 *
 * - a pointer (`item`) that points at the array, initialized to `NULL`
 *   (although please name the variable based on its contents, not on its
 *   type);
 *
 * - an integer variable (`alloc`) that keeps track of how big the current
 *   allocation is, initialized to `0`;
 *
 * - another integer variable (`nr`) to keep track of how many elements the
 *   array currently has, initialized to `0`.
 *
 * Then before adding `n`th element to the item, call `ALLOC_GROW(item, n,
 * alloc)`.  This ensures that the array can hold at least `n` elements by
 * calling `realloc(3)` and adjusting `alloc` variable.
 *
 * ------------
 * sometype *item;
 * size_t nr;
 * size_t alloc
 *
 * for (i = 0; i < nr; i++)
 * 	if (we like item[i] already)
 * 		return;
 *
 * // we did not like any existing one, so add one
 * ALLOC_GROW(item, nr + 1, alloc);
 * item[nr++] = value you like;
 * ------------
 *
 * You are responsible for updating the `nr` variable.
 *
 * If you need to specify the number of elements to allocate explicitly
 * then use the macro `REALLOC_ARRAY(item, alloc)` instead of `ALLOC_GROW`.
 *
 * Consider using ALLOC_GROW_BY instead of ALLOC_GROW as it has some
 * added niceties.
 *
 * DO NOT USE any expression with side-effect for 'x', 'nr', or 'alloc'.
 */
#define ALLOC_GROW(x, nr, alloc)                                               \
  do {                                                                         \
    if ((nr) > alloc) {                                                        \
      if (alloc_nr(alloc) < (nr))                                              \
        alloc = (nr);                                                          \
      else                                                                     \
        alloc = alloc_nr(alloc);                                               \
      REALLOC_ARRAY(x, alloc);                                                 \
    }                                                                          \
  } while (0)
