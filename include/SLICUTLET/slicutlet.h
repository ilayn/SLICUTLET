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
void ab01nd(const i32 jobz, const i32 n, const i32 m, f64* a, const i32 lda, f64* b, const i32 ldb, i32* ncont, i32* indcon, i32* nblk, f64* z, const i32 ldz, f64* tau, const f64 tol, i32* iwork, f64* dwork, const i32 ldwork, i32* info);
void ab05md(const i32 uplo, const i32 over, const i32 n1, const i32 m1, const i32 p1, const i32 n2, const i32 p2, const f64* a1, const i32 lda1, const f64* b1, const i32 ldb1, const f64* c1, const i32 ldc1, const f64* d1, const i32 ldd1, const f64* a2, const i32 lda2, const f64* b2, const i32 ldb2, const f64* c2, const i32 ldc2, const f64* d2, const i32 ldd2, i32* n, f64* a, const i32 lda, f64* b, const i32 ldb, f64* c, const i32 ldc, f64* d, const i32 ldd, f64* dwork, const i32 ldwork, i32* info);
void ab05nd(const i32 over, const i32 n1, const i32 m1, const i32 p1, const i32 n2, const f64 alpha, const f64* a1, const i32 lda1, const f64* b1, const i32 ldb1, const f64* c1, const i32 ldc1, const f64* d1, const i32 ldd1, const f64* a2, const i32 lda2, const f64* b2, const i32 ldb2, const f64* c2, const i32 ldc2, const f64* d2, const i32 ldd2, i32* n, f64* a, const i32 lda, f64* b, const i32 ldb, f64* c, const i32 ldc, f64* d, const i32 ldd, i32* iwork, f64* dwork, const i32 ldwork, i32* info);
void ab07nd(const i32 n, const i32 m, f64 *a, const i32 lda, f64 *b, const i32 ldb, f64 *c, const i32 ldc, f64 *d, const i32 ldd, f64* rcond, i32 *iwork, f64 *dwork, const i32 ldwork, i32* info);

void ma01ad(const f64 xr, const f64 xi, f64* yr, f64* yi);
void ma01bd(const f64 base, const f64 logbase, const int k, const int* s, const f64* a, const int inca, f64* alpha, f64* beta, i32* scale);
void ma01bz(const f64 base, const int k, const int* s, const c128* a, const int inca, c128* alpha, c128* beta, i32* scale);
i32  ma01cd(const f64 a, const i32 ia, const f64 b, const i32 ib);
void ma01dd(const f64 ar1, const f64 ai1, const f64 ar2, const f64 ai2, const f64 eps, const f64 safemin, f64* d);
void ma01dz(const f64 ar1, const f64 ai1, const f64 b1, const f64 ar2, const f64 ai2, const f64 b2, const f64 eps, const f64 safemin, f64* d1, f64* d2, i32* iwarn);

void ma02ad(const i32 job, const i32 m, const i32 n, f64* a, const i32 lda, f64* b, const i32 ldb);
void ma02az(const i32 trans, const i32 job, const i32 m, const i32 n, const c128* a, const i32 lda, c128* b, const i32 ldb);
void ma02bd(const i32 side, const i32 m, const i32 n, f64* a, const i32 lda);
void ma02bz(const i32 side, const i32 m, const i32 n, c128* a, const i32 lda);
void ma02cd(const i32 n, const i32 kl, const i32 ku, f64* a, const i32 lda);
void ma02cz(const i32 n, const i32 kl, const i32 ku, c128* a, const i32 lda);
void ma02dd(const i32 job, const i32 uplo, const i32 n, f64* a, const i32 lda, f64* ap);
void ma02ed(const i32 uplo, const i32 n, f64* a, const i32 lda);
void ma02es(const i32 uplo, const i32 n, f64* a, const i32 lda);
void ma02ez(const i32 uplo, const i32 trans, const i32 skew, const i32 n, c128* a, const i32 lda);
void ma02fd(f64* x1, f64* x2, f64* c, f64* s, i32* info);
void ma02gd(const i32 n, f64* a, const i32 lda, const i32 k1, const i32 k2, const i32* ipiv, const i32 incx);
void ma02gz(const i32 n, c128* a, const i32 lda, const i32 k1, const i32 k2, const i32* ipiv, const i32 incx);
i32  ma02hd(const i32 job, const i32 m, const i32 n, const f64 diag, const f64* a, const i32 lda);
i32  ma02hz(const i32 job, const i32 m, const i32 n, const c128 diag, const c128* a, const i32 lda);
f64  ma02id(const i32 typ, const i32 norm, const i32 n, const f64* a, const i32 lda, const f64* qg, const i32 ldqg, f64* dwork);
f64  ma02iz(const i32 typ, const i32 norm, const i32 n, const c128* a, const i32 lda, const c128* qg, const i32 ldqg, f64* dwork);
f64  ma02jd(const i32 ltran1, const i32 ltran2, const i32 n, const f64* q1, const i32 ldq1, const f64* q2, const i32 ldq2, f64* res, const i32 ldres);
f64  ma02jz(const i32 ltran1, const i32 ltran2, const i32 n, const c128* q1, const i32 ldq1, const c128* q2, const i32 ldq2, c128* res, const i32 ldres);
f64  ma02md(const i32 norm, const i32 uplo, const i32 n, const f64* a, const i32 lda, f64* dwork);
f64  ma02mz(const i32 norm, const i32 uplo, const i32 n, const c128* a, const i32 lda, f64* dwork);
void ma02nz(const i32 uplo, const i32 trans, const i32 skew, const i32 n, const i32 k, const i32 l, c128* a, const i32 lda);
i32  ma02od(const i32 skew, const i32 m, const f64* a, const i32 lda, const f64* de, const i32 ldde);
i32  ma02oz(const i32 skew, const i32 m, const c128* a, const i32 lda, const c128* de, const i32 ldde);
void ma02pd(const i32 m, const i32 n, const f64* a, const i32 lda, i32* nzr, i32* nzc);
void ma02pz(const i32 m, const i32 n, const c128* a, const i32 lda, i32* nzr, i32* nzc);
void ma02rd( const i32 id, const i32 n, f64* d, f64* e, i32* info);
f64  ma02sd(const i32 m, const i32 n, const f64* a, const i32 lda);

// Commented out routines below are already available in BLAS/LAPACK under different names. See their docstrings for details.
// void mb01kd(const i32 uplo, const i32 trans, const i32 n, const i32 k, const f64 alpha, const f64* a, const i32 lda, f64* b, const i32 ldb, const f64 beta, f64* c, const i32 ldc, i32* info);
void mb01ld(const i32 uplo, const i32 trans, const i32 m, const i32 n, const i32 k, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* a, const i32 lda, f64* x, const i32 ldx, f64* dwork, const i32 ldwork, i32* info);
// void mb01md(const i32 uplo, const i32 n, const f64 alpha, const f64* a, const i32 lda, const f64* x, const i32 incx, const f64 beta, f64* y, const i32 incy, i32* info);
// void mb01nd();
void mb01oc(const i32 uplo, const i32 trans, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* h, const i32 ldh, f64* x, const i32 ldx, i32* info);
void mb01od(const i32 uplo, const i32 trans, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, f64* h, const i32 ldh, f64* x, const i32 ldx, f64* e, const i32 lde, f64* dwork, const i32 ldwork, i32* info);
void mb01oe(const i32 uplo, const i32 trans, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* h, const i32 ldh, f64* e, const i32 lde);
void mb01oh(const i32 uplo, const i32 trans, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* h, const i32 ldh, f64* a, const i32 lda);
void mb01oo(const i32 uplo, const i32 trans, const i32 n, const f64* h, const i32 ldh, const f64* x, const i32 ldx, const f64* e, const i32 lde, f64* p, const i32 ldp, i32* info);
void mb01os(const i32 uplo, const i32 trans, const i32 n, const f64* h, const i32 ldh, const f64* x, const i32 ldx, f64* p, const i32 ldp, i32* info);
void mb01ot(const i32 uplo, const i32 trans, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* e, const i32 lde, const f64* t, const i32 ldt, i32* info);
void mb01pd(const i32 scun, const i32 type, const i32 m, const i32 n, const i32 kl, const i32 ku, const f64 anrm, const i32 nbl, const i32* nrows, f64* a, const i32 lda, i32* info);
void mb01qd(const i32 type, const i32 m, const i32 n, const i32 kl, const i32 ku, const f64 cfrom, const f64 cto, const i32 nbl, const i32* nrows, f64* a, const i32 lda, i32* info);
void mb01rb(const i32 side, const i32 uplo, const i32 trans, const i32 m, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, f64* a, const i32 lda, const f64* b, const i32 ldb, i32* info);
void mb01rd(const i32 uplo, const i32 trans, const i32 m, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* a, const i32 lda, f64* x, const i32 ldx, f64* dwork, const i32 ldwork, i32* info);
void mb01rh(const i32 uplo, const i32 trans, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, f64* h, const i32 ldh, f64* x, const i32 ldx, f64* dwork, const i32 ldwork, i32* info);
void mb01rt(const i32 uplo, const i32 trans, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* e, const i32 lde, f64* x, const i32 ldx, f64* dwork, const i32 ldwork, i32* info);
void mb01ru(const i32 uplo, const i32 trans, const i32 m, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* a, const i32 lda, f64* x, const i32 ldx, f64* dwork, const i32 ldwork, i32* info);
void mb01rw(const i32 uplo, const i32 trans, const i32 m, const i32 n, f64* a, const i32 lda, const f64* z, const i32 ldz, f64* dwork, i32* info);
void mb01rx(const i32 side, const i32 uplo, const i32 trans, const i32 m, const i32 n, const f64 alpha, const f64 beta, f64* r, const i32 ldr, const f64* a, const i32 lda, const f64* b, const i32 ldb, i32* info);
void mb01ry(const i32 side, const i32 uplo, const i32 trans, const i32 m, const f64 alpha, const f64 beta, f64* r, const i32 ldr, f64* h, const i32 ldh, const f64* b, const i32 ldb, f64* dwork, i32* info);
void mb01sd(const i32 jobs, const i32 m, const i32 n, f64* a, const i32 lda, const f64* r, const f64* c);
void mb01ss(const i32 jobs, const i32 uplo, const i32 n, f64* a, const i32 lda, const f64* d);
void mb01td(const i32 n, const f64 *a, const i32 lda, f64 *b, const i32 ldb, f64 *dwork, i32 *info);
void mb01ud(const i32 side, const i32 trans, const i32 m, const i32 n, const f64 alpha, f64* h, const i32 ldh, const f64* a, const i32 lda, f64* b, const i32 ldb, i32* info);
void mb01uw(const i32 side, const i32 trans, const i32 m, const i32 n, const f64 alpha, f64 *h, const i32 ldh, f64 *a, const i32 lda, f64 *dwork, const i32 ldwork, i32 *info);
void mb01ux(const i32 side, const i32 uplo, const i32 trans, const i32 m, const i32 n, const f64 alpha, const f64 *t, const i32 ldt, f64 *a, const i32 lda, f64 *dwork, const i32 ldwork, i32 *info);
void mb01uy(const i32 side, const i32 uplo, const i32 trans, const i32 m, const i32 n, const f64 alpha, f64 *t, const i32 ldt, const f64 *a, const i32 lda, f64 *dwork, const i32 ldwork, i32 *info);
void mb01uz(const i32 side, const i32 uplo, const i32 trans, const i32 m, const i32 n, const c128 alpha, c128 *t, const i32 ldt, c128 *a, const i32 lda, c128 *zwork, const i32 lzwork, i32 *info);
void mb01vd(const i32 trana, const i32 tranb, const i32 ma, const i32 na, const i32 mb, const i32 nb, const f64 alpha, const f64 beta, const f64 *a, const i32 lda, const f64 *b, const i32 ldb, f64 *c, const i32 ldc, i32 *mc, i32 *nc, i32 *info);
void mb01wd(const i32 dico, const i32 uplo, const i32 trans, const i32 hess, const i32 n, const f64 alpha, const f64 beta, f64 *r, const i32 ldr, f64 *a, const i32 lda, f64 *t, const i32 ldt, i32 *info);
void mb01xd(const i32 uplo, const i32 n, f64* a, const i32 lda, i32 *info);
void mb01xy(const i32 uplo, const i32 n, f64* a, const i32 lda, i32 *info);
void mb01yd(const i32 uplo, const i32 trans, const i32 n, const i32 k, const i32 l, const f64 alpha, const f64 beta, const f64* a, const i32 lda, f64* c, const i32 ldc, i32* info);
void mb01zd(const i32 side, const i32 uplo, const i32 transt, const i32 diag, const i32 m, const i32 n, const i32 l, const f64 alpha, const f64* t, const i32 ldt, f64* h, const i32 ldh, i32* info);

void mb03oy(const i32 m, const i32 n, f64* a, const i32 lda, const f64 rcond, const f64 svlmax, i32* rank, f64* sval, i32* jpvt, f64* tau, f64* dwork, i32* info);
void mc01td(const i32 dico, i32* dp, f64* p, i32* stable, i32* nz, f64* dwork, i32* iwarn, i32* info);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
