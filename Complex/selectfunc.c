#include "EXTERN.h"
#include "perl.h"
#include "pdl.h"
#include "pdlcore.h"

#define PDL PDL_LinearAlgebra_Complex
extern Core *PDL;

#define PDL_LA_COMPLEX_INIT_PUSH(pdl, type, valp, svpdl) \
   pdl = PDL->pdlnew(); \
   PDL->setdims (pdl, dims, 2); \
   pdl->datatype = type; \
   pdl->data = valp; \
   pdl->state |= PDL_DONTTOUCHDATA | PDL_ALLOCATED; \
   ENTER;   SAVETMPS;   PUSHMARK(sp); \
    svpdl = sv_newmortal(); \
    PDL->SetSV_PDL(svpdl, pdl); \
    svpdl = sv_bless(svpdl, bless_stash); /* bless in PDL::Complex  */ \
    XPUSHs(svpdl); \
   PUTBACK;

#define PDL_LA_COMPLEX_UNINIT(pdl) \
   PDL->setdims (pdl, odims, 0); \
   pdl->state &= ~ (PDL_ALLOCATED |PDL_DONTTOUCHDATA); \
   pdl->data=NULL;

#define PDL_LA_CALL_SV(type, valp, sv_func) \
   dSP; \
   int count; \
   long ret; \
   SV *pdl1; \
   HV *bless_stash; \
   pdl *pdl; \
   PDL_Indx odims[1]; \
   PDL_Indx dims[] = {2,1}; \
   bless_stash = gv_stashpv("PDL::Complex", 0); \
   PDL_LA_COMPLEX_INIT_PUSH(pdl, type, valp, pdl1) \
   count = perl_call_sv(sv_func, G_SCALAR); \
   SPAGAIN; \
   if (count !=1) \
      croak("Error calling perl function\n"); \
   /* For pdl_free */ \
   odims[0] = 0; \
   PDL_LA_COMPLEX_UNINIT(pdl) \
   ret = (long ) POPl ; \
   PUTBACK ;   FREETMPS ;   LEAVE ; \
   return ret;

/* replace BLAS one so don't terminate on bad input */
int xerbla_(char *sub, int *info) { return 0; }

#define SEL_FUNC(letter, type, pdl_type) \
  static SV* letter ## select_func; \
  void letter ## select_func_set(SV* func) { \
    letter ## select_func = func; \
  } \
  PDL_Long letter ## select_wrapper(type *p) \
  { \
    PDL_LA_CALL_SV(pdl_type, p, letter ## select_func) \
  }
SEL_FUNC(f, float, PDL_F)
SEL_FUNC(d, double, PDL_D)

#define PDL_LA_CALL_GSV(type, val1p, val2p, sv_func) \
   dSP; \
   int count; \
   long ret; \
   SV *svpdl1, *svpdl2; \
   HV *bless_stash; \
   pdl *pdl1, *pdl2; \
   PDL_Indx odims[1]; \
   PDL_Indx dims[] = {2,1}; \
   bless_stash = gv_stashpv("PDL::Complex", 0); \
   PDL_LA_COMPLEX_INIT_PUSH(pdl1, type, val1p, svpdl1) \
   PDL_LA_COMPLEX_INIT_PUSH(pdl2, type, val2p, svpdl2) \
   count = perl_call_sv(sv_func, G_SCALAR); \
   SPAGAIN; \
   if (count !=1) \
      croak("Error calling perl function\n"); \
   /* For pdl_free */ \
   odims[0] = 0; \
   PDL_LA_COMPLEX_UNINIT(pdl1) \
   PDL_LA_COMPLEX_UNINIT(pdl2) \
   ret = (long ) POPl ; \
   PUTBACK ;   FREETMPS ;   LEAVE ; \
   return ret;

#define GSEL_FUNC(letter, type, pdl_type) \
  static SV* letter ## gselect_func; \
  void letter ## gselect_func_set(SV* func) { \
    letter ## gselect_func = func; \
  } \
  PDL_Long letter ## gselect_wrapper(type *p, type *q) \
  { \
    PDL_LA_CALL_GSV(pdl_type, p, q, letter ## gselect_func) \
  }
GSEL_FUNC(f, float, PDL_F)
GSEL_FUNC(d, double, PDL_D)

#define TRACE(letter, type) \
  void c ## letter ## trace(int n, void *a1, void *a2) { \
    type *mat = a1, *res = a2; \
    PDL_Indx i; \
    res[0] = mat[0]; \
    res[1] = mat[1]; \
    for (i = 1; i < n; i++) \
    { \
          res[0] += mat[(i*(n+1))*2]; \
          res[1] += mat[(i*(n+1))*2+1]; \
    } \
  }
TRACE(f, float)
TRACE(d, double)
