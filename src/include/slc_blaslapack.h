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
 * From this point on, until it is undefined again, the i-word must be avoided outside BLAS/LAPACK prototypes.
 *
 */

// BLAS routines
void   SLC_F77_FUNC(dcopy, DCOPY)(int* n, f64* dx, int* incx, f64* dy, int* incy);
void   SLC_F77_FUNC(dscal, DSCAL)(int* n, f64* a, f64* x, int* incx);
void   SLC_F77_FUNC(dswap, DSWAP)(int* n, f64* x, int* incx, f64* y, int* incy);
f64    SLC_F77_FUNC(dnrm2, DNRM2)(int* n, f64* x, int* incx);
int    SLC_F77_FUNC(idamax, IDAMAX)(int* n, f64* x, int* incx);
void   SLC_F77_FUNC(dgemv, DGEMV)(char* trans, int* m, int* n, f64* alpha, f64* a, int* lda, f64* x, int* incx, f64* beta, f64* y, int* incy);
void   SLC_F77_FUNC(dgemm, DGEMM)(char* transa, char* transb, int* m, int* n, int* k, f64* alpha, f64* a, int* lda, f64* b, int* ldb, f64* beta, f64* c, int* ldc);
void   SLC_F77_FUNC(zgemm, ZGEMM)(char* transa, char* transb, int* m, int* n, int* k, c128* alpha, c128* a, int* lda, c128* b, int* ldb, c128* beta, c128* c, int* ldc);


// LAPACK routines
void   SLC_F77_FUNC(dgetrf, DGETRF)(int* m, int* n, f64* a, int* lda, int* ipiv, int* info);
void   SLC_F77_FUNC(dgetri, DGETRI)(int* n, f64* a, int* lda, int* ipiv, f64* work, int* lwork, int* info);
void   SLC_F77_FUNC(dgecon, DGECON)(char* norm, int* n, f64* a, int* lda, f64* anorm, f64* rcond, f64* work, int* iwork, int* info);
f64    SLC_F77_FUNC(dlange, DLANGE)(char* norm, int* m, int* n, f64* a, int* lda, f64* work);
void   SLC_F77_FUNC(dlacpy, DLACPY)(char* uplo, int* m, int* n, f64* a, int* lda, f64* b, int* ldb);
void   SLC_F77_FUNC(dlarf, DLARF)(char* side, int* m, int* n, f64* v, int* incv, f64* tau, f64* c, int* ldc, f64* work);
void   SLC_F77_FUNC(dlarfg, DLARFG)(int* n, f64* alpha, f64* x, int* incx, f64* tau);
void   SLC_F77_FUNC(dlaic1, DLAIC1)(int* job, int* j, f64* x, f64* sest, f64* w, f64* gamma, f64* sestpr, f64* s, f64* c);
void   SLC_F77_FUNC(drscl, DRSCL)(int* n, f64* sa, f64* sx, int* incx);
void   SLC_F77_FUNC(dgges, DGGES)(char* jobvsl, char* jobvsr, char* sort, lapack_logical (*selctg)(f64*, f64*, f64*), int* n, f64* a, int* lda, f64* b, int* ldb, int* sdim, f64* alphar, f64* alphai, f64* beta, f64* vsl, int* ldvsl, f64* vsr, int* ldvsr, f64* work, int* lwork, lapack_logical* bwork, int* info);

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
#define SLC_DSCAL   SLC_F77_FUNC(dscal,  DSCAL)
#define SLC_DSWAP   SLC_F77_FUNC(dswap,  DSWAP)
#define SLC_DNRM2   SLC_F77_FUNC(dnrm2,  DNRM2)
#define SLC_IDAMAX  SLC_F77_FUNC(idamax, IDAMAX)

#define SLC_DGETRF  SLC_F77_FUNC(dgetrf, DGETRF)
#define SLC_DGETRI  SLC_F77_FUNC(dgetri, DGETRI)
#define SLC_DGECON  SLC_F77_FUNC(dgecon, DGECON)
#define SLC_DLANGE  SLC_F77_FUNC(dlange, DLANGE)
#define SLC_DLACPY  SLC_F77_FUNC(dlacpy, DLACPY)
#define SLC_DLARF   SLC_F77_FUNC(dlarf,  DLARF)
#define SLC_DLARFG  SLC_F77_FUNC(dlarfg, DLARFG)
#define SLC_DLAIC1  SLC_F77_FUNC(dlaic1, DLAIC1)
#define SLC_DRSCL   SLC_F77_FUNC(drscl,  DRSCL)

#define SLC_DGGES   SLC_F77_FUNC(dgges,  DGGES)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLC_BLASLAPACK_H */
