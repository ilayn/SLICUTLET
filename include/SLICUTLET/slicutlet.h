#ifndef SLICUTLET_H
#define SLICUTLET_H

// Public API header for SLICUTLET: exposes selected translated SLICOT routines.

#include <stddef.h>
#include <float.h>

// Basic numeric typedefs used by the API
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

void ab04md(const i32 typ, const i32 n, const i32 m, const i32 p, const f64 alpha, const f64 beta, f64 *a, const i32 lda, f64 *b, const i32 ldb, f64 *c, const i32 ldc, f64 *d, const i32 ldd, i32 *iwork, f64 *dwork, const i32 ldwork, i32 *info);
void ab05md(const i32 uplo, const i32 over, const i32 n1, const i32 m1, const i32 p1, const i32 n2, const i32 p2, const f64* a1, const i32 lda1, const f64* b1, const i32 ldb1, const f64* c1, const i32 ldc1, const f64* d1, const i32 ldd1, const f64* a2, const i32 lda2, const f64* b2, const i32 ldb2, const f64* c2, const i32 ldc2, const f64* d2, const i32 ldd2, i32* n, f64* a, const i32 lda, f64* b, const i32 ldb, f64* c, const i32 ldc, f64* d, const i32 ldd, f64* dwork, const i32 ldwork, i32* info);
void ab05nd(const i32 over, const i32 n1, const i32 m1, const i32 p1, const i32 n2, const f64 alpha, const f64* a1, const i32 lda1, const f64* b1, const i32 ldb1, const f64* c1, const i32 ldc1, const f64* d1, const i32 ldd1, const f64* a2, const i32 lda2, const f64* b2, const i32 ldb2, const f64* c2, const i32 ldc2, const f64* d2, const i32 ldd2, i32* n, f64* a, const i32 lda, f64* b, const i32 ldb, f64* c, const i32 ldc, f64* d, const i32 ldd, i32* iwork, f64* dwork, const i32 ldwork, i32* info);
void ab07nd(const i32 n, const i32 m, f64 *a, const i32 lda, f64 *b, const i32 ldb, f64 *c, const i32 ldc, f64 *d, const i32 ldd, f64* rcond, i32 *iwork, f64 *dwork, const i32 ldwork, i32* info);

void mb03oy(const i32 m, const i32 n, f64* a, const i32 lda, const f64 rcond, const f64 svlmax, i32* rank, f64* sval, i32* jpvt, f64* tau, f64* dwork, i32* info);
void mc01td(const i32 dico, i32* dp, f64* p, i32* stable, i32* nz, f64* dwork, i32* iwarn, i32* info);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
