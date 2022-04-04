#include "EXTERN.h"
#include "perl.h"
#include "pdl.h"
#include "pdlcore.h"

#define PDL PDL_LinearAlgebra_Real
extern Core *PDL;

#define PDL_LA_CALL_SV(val1p, val2p, sv_func) \
  dSP ; \
  long  retval; \
  int count; \
  ENTER ; \
  SAVETMPS ; \
  PUSHMARK(sp) ; \
  XPUSHs(sv_2mortal(newSVnv((double ) *val1p))); \
  XPUSHs(sv_2mortal(newSVnv((double ) *val2p))); \
  PUTBACK ; \
  count = perl_call_sv(sv_func, G_SCALAR); \
  SPAGAIN; \
  if (count != 1) \
    croak("Error calling perl function\n"); \
  retval = (long ) POPl ;  /* Return value */ \
  PUTBACK ; \
  FREETMPS ; \
  LEAVE ; \
  return retval;

/* replace BLAS one so don't terminate on bad input */
int xerbla_(char *sub, int *info) { return 0; }

#define SEL_FUNC(letter, type) \
  static SV* letter ## select_func; \
  void letter ## select_func_set(SV* func) { \
    letter ## select_func = func; \
  } \
  PDL_Long letter ## select_wrapper(type *wr, type *wi) \
  { \
    PDL_LA_CALL_SV(wr, wi, letter ## select_func) \
  }
SEL_FUNC(f, float)
SEL_FUNC(d, double)

#define PDL_LA_CALL_GSV(val1p, val2p, val3p, sv_func) \
  dSP ; \
  long  retval; \
  int count; \
  ENTER ; \
  SAVETMPS ; \
  PUSHMARK(sp) ; \
  XPUSHs(sv_2mortal(newSVnv((double)  *val1p))); \
  XPUSHs(sv_2mortal(newSVnv((double)  *val2p))); \
  XPUSHs(sv_2mortal(newSVnv((double)  *val3p))); \
  PUTBACK ; \
  count = perl_call_sv(sv_func, G_SCALAR); \
  SPAGAIN; \
  if (count != 1) \
    croak("Error calling perl function\n"); \
  retval = (long ) POPl ;  /* Return value */ \
  PUTBACK ; \
  FREETMPS ; \
  LEAVE ; \
  return retval;

#define GSEL_FUNC(letter, type) \
  static SV* letter ## gselect_func; \
  void letter ## gselect_func_set(SV* func) { \
    letter ## gselect_func = func; \
  } \
  PDL_Long letter ## gselect_wrapper(type *zr, type *zi, type *d) \
  { \
    PDL_LA_CALL_GSV(zr, zi, d, letter ## gselect_func) \
  }
GSEL_FUNC(f, float)
GSEL_FUNC(d, double)

#define TRACE(letter, type) \
  type letter ## trace(int n, type *mat) { \
    PDL_Indx i; \
    type sum = mat[0]; \
    for (i = 1; i < n; i++) \
          sum += mat[i*(n+1)]; \
    return sum; \
  }
TRACE(f, float)
TRACE(d, double)
