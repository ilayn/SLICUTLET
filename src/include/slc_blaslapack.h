#ifndef SLC_BLASLAPACK_H
#define SLC_BLASLAPACK_H

#include <stdint.h>
// Local numeric aliases (i32/i64/f32/f64/c64/c128)
#include "types.h"

// Actual build configuration that lives in the build directory and generated from the .in template.
#include "slc_config.h"

/**
 *
 *  Minimal portability shim
 *    - sl_int resolves to 32 or 64-bit integer for BLAS/LAPACK sizes based on SLC_ILP64.
 *    - SLC_F77_FUNC resolves Fortran symbol name mangling detected at configure time.
 *    - lapack_logical is fixed 32-bit to model Fortran LOGICAL (hopefully) in a portable manner.
 *
 */
#if defined(SLC_ILP64) && SLC_ILP64
    typedef int64_t sl_int;
#else
    typedef int32_t sl_int;
#endif

#if defined(SLC_FC_LOWER_US) && SLC_FC_LOWER_US
    #define SLC_F77_FUNC(lc,UC) lc##_
#elif defined(SLC_FC_LOWER) && SLC_FC_LOWER
    #define SLC_F77_FUNC(lc,UC) lc
#elif defined(SLC_FC_UPPER) && SLC_FC_UPPER
    #define SLC_F77_FUNC(lc,UC) UC
#else
    // Default to lowercase with trailing underscore
    #define SLC_F77_FUNC(lc,UC) lc##_
#endif

#ifndef SLC_F77_FUNC_US
    #define SLC_F77_FUNC_US(lc,UC) SLC_F77_FUNC(lc,UC)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t lapack_logical;

/**
 *
 * IMPORTANT:
 *
 * An evil hack to skip typing hundreds of sl_int in the prototypes below.
 * We define "int" to be "sl_int" for the scope of the following prototypes.
 * We then undefine it at the end of prototypes.
 *
 * Until BLAS/LAPACK is rewritten in another language, we need these silly
 * games to keep the interface manageable for different symbol mangling
 * rules and also for 64-bit LAPACK integer support.
 *
 * Callbacks (e.g., DGGES SELCTG) should return lapack_logical and accept
 * pointer arguments by reference, matching Fortran ABI expectations.
 *
 */

#define int sl_int

/**
 *
 * From this point on, until it is undefined again, the "int" must be avoided outside BLAS/LAPACK prototypes.
 *
 */

// BLAS routines
void   SLC_F77_FUNC(dcopy, DCOPY)(const int* n, const f64* dx, const int* incx, f64* dy, const int* incy);
void   SLC_F77_FUNC(dscal, DSCAL)(const int* n, const f64* a, f64* x, const int* incx);
void   SLC_F77_FUNC(dswap, DSWAP)(const int* n, f64* x, const int* incx, f64* y, const int* incy);
f64    SLC_F77_FUNC(dnrm2, DNRM2)(const int* n, const f64* x, const int* incx);
int    SLC_F77_FUNC(idamax, IDAMAX)(const int* n, const f64* x, const int* incx);
void   SLC_F77_FUNC(dgemv, DGEMV)(const char* trans, const int* m, const int* n, const f64* alpha, const f64* a, const int* lda, const f64* x, const int* incx, const f64* beta, f64* y, const int* incy);
void   SLC_F77_FUNC(dgemm, DGEMM)(const char* transa, const char* transb, const int* m, const int* n, const int* k, const f64* alpha, const f64* a, const int* lda, const f64* b, const int* ldb, const f64* beta, f64* c, const int* ldc);
void   SLC_F77_FUNC(zgemm, ZGEMM)(const char* transa, const char* transb, const int* m, const int* n, const int* k, const c128* alpha, const c128* a, const int* lda, const c128* b, const int* ldb, const c128* beta, c128* c, const int* ldc);
void   SLC_F77_FUNC(dtrsm, DTRSM)(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const f64* alpha, const f64* a, const int* lda, f64* b, const int* ldb);


// LAPACK routines
void   SLC_F77_FUNC(dgetrf, DGETRF)(const int* m, const int* n, f64* a, const int* lda, int* ipiv, int* info);
void   SLC_F77_FUNC(dgetri, DGETRI)(const int* n, f64* a, const int* lda, const int* ipiv, f64* work, const int* lwork, int* info);
void   SLC_F77_FUNC(dgetrs, DGETRS)(const char* trans, const int* n, const int* nrhs, const f64* a, const int* lda, const int* ipiv, f64* b, const int* ldb, int* info);
void   SLC_F77_FUNC(dgecon, DGECON)(const char* norm, const int* n, const f64* a, const int* lda, const f64* anorm, f64* rcond, f64* work, int* iwork, int* info);
f64    SLC_F77_FUNC(dlange, DLANGE)(const char* norm, const int* m, const int* n, const f64* a, const int* lda, f64* work);
void   SLC_F77_FUNC(dlacpy, DLACPY)(const char* uplo, const int* m, const int* n, const f64* a, const int* lda, f64* b, const int* ldb);
void   SLC_F77_FUNC(dlaset, DLASET)(const char* uplo, const int* m, const int* n, const f64* alpha, const f64* beta, f64* a, const int* lda);
void   SLC_F77_FUNC(dlascl, DLASCL)(const char* type, const int* kl, const int* ku, const f64* cfrom, const f64* cto, const int* m, const int* n, f64* a, const int* lda, int* info);
void   SLC_F77_FUNC(dlarf, DLARF)(const char* side, const int* m, const int* n, const f64* v, const int* incv, const f64* tau, f64* c, const int* ldc, f64* work);
void   SLC_F77_FUNC(dlarfg, DLARFG)(const int* n, f64* alpha, f64* x, const int* incx, f64* tau);
void   SLC_F77_FUNC(dlaic1, DLAIC1)(const int* job, const int* j, const f64* x, const f64* sest, const f64* w, const f64* gamma, f64* sestpr, f64* s, f64* c);
void   SLC_F77_FUNC(drscl, DRSCL)(const int* n, const f64* sa, f64* sx, const int* incx);
void   SLC_F77_FUNC(dgges, DGGES)(const char* jobvsl, const char* jobvsr, const char* sort, lapack_logical (*selctg)(const f64*, const f64*, const f64*), const int* n, f64* a, const int* lda, f64* b, const int* ldb, int* sdim, f64* alphar, f64* alphai, f64* beta, f64* vsl, const int* ldvsl, f64* vsr, const int* ldvsr, f64* work, const int* lwork, lapack_logical* bwork, int* info);
f64    SLC_F77_FUNC(dlamch, DLAMCH)(const char* cmach);
void   SLC_F77_FUNC(dlabad, DLABAD)(f64* small, f64* large);
void   SLC_F77_FUNC(dlapmt, DLAPMT)(const int* forwrd, const int* m, const int* n, f64* x, const int* ldx, const int* k);
void   SLC_F77_FUNC(dormqr, DORMQR)(const char* side, const char* trans, const int* m, const int* n, const int* k, const f64* a, const int* lda, const f64* tau, f64* c, const int* ldc, f64* work, const int* lwork, int* info);
void   SLC_F77_FUNC(dorgqr, DORGQR)(const int* m, const int* n, const int* k, f64* a, const int* lda, const f64* tau, f64* work, const int* lwork, int* info);

/**
 * End of evil hack.
 */

#undef int

/**
 *
 * int is a clean word again.
 *
*/


/* Simple alias macros for use at call sites */
#define SLC_DCOPY   SLC_F77_FUNC(dcopy, DCOPY)
#define SLC_DGEMM   SLC_F77_FUNC(dgemm,  DGEMM)
#define SLC_ZGEMM   SLC_F77_FUNC(zgemm,  ZGEMM)
#define SLC_DGEMV   SLC_F77_FUNC(dgemv,  DGEMV)
#define SLC_DTRSM   SLC_F77_FUNC(dtrsm,  DTRSM)
#define SLC_DSCAL   SLC_F77_FUNC(dscal,  DSCAL)
#define SLC_DSWAP   SLC_F77_FUNC(dswap,  DSWAP)
#define SLC_DNRM2   SLC_F77_FUNC(dnrm2,  DNRM2)
#define SLC_IDAMAX  SLC_F77_FUNC(idamax, IDAMAX)

#define SLC_DGETRF  SLC_F77_FUNC(dgetrf, DGETRF)
#define SLC_DGETRI  SLC_F77_FUNC(dgetri, DGETRI)
#define SLC_DGETRS  SLC_F77_FUNC(dgetrs, DGETRS)
#define SLC_DGECON  SLC_F77_FUNC(dgecon, DGECON)
#define SLC_DLANGE  SLC_F77_FUNC(dlange, DLANGE)
#define SLC_DLACPY  SLC_F77_FUNC(dlacpy, DLACPY)
#define SLC_DLASET  SLC_F77_FUNC(dlaset, DLASET)
#define SLC_DLASCL  SLC_F77_FUNC(dlascl, DLASCL)
#define SLC_DLARF   SLC_F77_FUNC(dlarf,  DLARF)
#define SLC_DLARFG  SLC_F77_FUNC(dlarfg, DLARFG)
#define SLC_DLAIC1  SLC_F77_FUNC(dlaic1, DLAIC1)
#define SLC_DRSCL   SLC_F77_FUNC(drscl,  DRSCL)

#define SLC_DGGES   SLC_F77_FUNC(dgges,  DGGES)
#define SLC_DLAMCH  SLC_F77_FUNC(dlamch, DLAMCH)
#define SLC_DLABAD  SLC_F77_FUNC(dlabad, DLABAD)
#define SLC_DLAPMT  SLC_F77_FUNC(dlapmt, DLAPMT)
#define SLC_DORMQR  SLC_F77_FUNC(dormqr, DORMQR)
#define SLC_DORGQR  SLC_F77_FUNC(dorgqr, DORGQR)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLC_BLASLAPACK_H */
