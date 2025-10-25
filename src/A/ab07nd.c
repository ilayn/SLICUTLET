#include "slicutlet.h"
#include "../include/slc_blaslapack.h"

void
ab07nd(
    const i32 n,
    const i32 m,
    f64 *a,
    const i32 lda,
    f64 *b,
    const i32 ldb,
    f64 *c,
    const i32 ldc,
    f64 *d,
    const i32 ldd,
    f64* rcond,
    i32 *iwork,
    f64 *dwork,
    const i32 ldwork,
    i32* info
)
{
    // Common integer and double constants for BLAS/LAPACK calls
    i32 int1 = 1, intm1 = -1, int0 = 0;
    f64 dbl0 = 0.0, dbl1 = 1.0, dblm1 = -1.0;

    i32 ierr = 0, lquery = 0, maxwork = 0, minwork = 0;
    *info = 0;

    // Test the input scalar arguments
    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (ldd < (m > 1 ? m : 1)) {
        *info = -10;
    } else {
        lquery = (ldwork == -1);
        minwork = (4*m > 1 ? 4*m : 1);
        // Get workspace size for D inversion
        SLC_DGETRI(&m, d, &ldd, iwork, dwork, &intm1, &ierr);
        maxwork = MAX(MAX(minwork, (i32)dwork[0]), n*m);
        if ((ldwork < maxwork) && (!lquery)) {
            *info = -14;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = maxwork;
        return;
    }

    // Quick return if possible
    if (m == 0) {
        *rcond = 1.0;
        dwork[0] = 1.0;
        return;
    }

    // Factorize D
    SLC_DGETRF(&m, &m, d, &ldd, iwork, info);
    if (*info != 0) {
        *rcond = 0.0;
        return;
    }

    // Compute the reciprocal condition number of matrix D
    f64 dnorm = SLC_DLANGE("1", &m, &m, d, &ldd, dwork);
    SLC_DGECON("1", &m, d, &ldd, &dnorm, rcond, dwork, &iwork[m], &ierr);
    if (*rcond < DBL_EPSILON) {
        // Numerically singular; report M+1 but continue computations as in reference
        *info = m+1;
    }

    // Compute D inverse
    SLC_DGETRI(&m, d, &ldd, iwork, dwork, &ldwork, &ierr);
    if (n > 0) {

        // Copy b to dwork
    SLC_DLACPY("A", &n, &m, b, &ldb, dwork, &n);

        // Compute -b * d^-1
    SLC_DGEMM("N", "N", &n, &m, &m, &dblm1, dwork, &n, d, &ldd, &dbl0, b, &ldb);

        // Compute a + (-b * d^-1) * c
    SLC_DGEMM("N", "N", &n, &n, &m, &dbl1, b, &ldb, c, &ldc, &dbl1, a, &lda);

    // Compute d^-1 * c: stage C into workspace first
    SLC_DLACPY("A", &m, &n, c, &ldc, dwork, &m);
    SLC_DGEMM("N", "N", &m, &n, &m, &dbl1, d, &ldd, dwork, &m, &dbl0, c, &ldc);
    }

    // Return optimal workspace in dwork[0]
    dwork[0] = maxwork;

    return;
}
