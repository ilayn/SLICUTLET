#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ab04md(
    const i32 typ,  // 0 discrete, 1 continuous
    const i32 n,
    const i32 m,
    const i32 p,
    const f64 alpha,
    const f64 beta,
    f64 *a,
    const i32 lda,
    f64 *b,
    const i32 ldb,
    f64 *c,
    const i32 ldc,
    f64 *d,
    const i32 ldd,
    i32 *iwork,
    f64 *dwork,
    const i32 ldwork,
    i32 *info
)
{
    // Common integer and double constants for BLAS/LAPACK calls
    i32 int1 = 1, int0 = 0;
    f64 dbl1 = 1.0, dblm1 = -1.0, dbl2 = 2.0;

    i32 i = 0, ip = 0;
    i32 n_mut = n, lda_mut = lda, ldb_mut = ldb, ldc_mut = ldc, ldd_mut = ldd, ldwork_mut = ldwork;
    i32 m_mut = m, p_mut = p;
    f64 palpha = 0.0, pbeta = 0.0, ab2 = 0.0, sqrab2 = 0.0;
    *info = 0;

    // Test the input scalar arguments
    if (typ != 0 && typ != 1) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (alpha == 0.0) {
        *info = -5;
    } else if (beta == 0.0) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -12;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -14;
    } else if (ldwork < (n > 1 ? n : 1)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if (MAX(MAX(n, m), p) == 0) {
        return;
    }

    if (typ == 0) {
        // Discrete-time to continuous-time with (alpha, beta)
        palpha = alpha;
        pbeta = beta;
    } else {
        // Continuous-time to discrete-time with (alpha, beta) is
        // equivalent to discrete-time to continuous-time with
        // (-beta, -alpha), if B and C change sign
        palpha = -beta;
        pbeta = -alpha;
    }

    ab2 = palpha * pbeta * dbl2;
    sqrab2 = (ab2 >= 0.0 ? 1.0 : -1.0) * sqrt(fabs(ab2));
    if (palpha < 0.0) {
        sqrab2 = -sqrab2;
    }

    // Compute (alpha*I + A)^-1
    for (i = 0; i < n; i++) {
        a[i + i*lda] += palpha;
    }

    SLC_DGETRF(&n_mut, &n_mut, a, &lda_mut, iwork, info);

    if (*info != 0) {
        // Error return
        if (typ == 0) {
            *info = 1;
        } else {
            *info = 2;
        }
        return;
    }

    // Compute (alpha*I + A)^-1 * B
    SLC_DGETRS("N", &n_mut, &m_mut, a, &lda_mut, iwork, b, &ldb_mut, info);

    // Compute D - C * (alpha*I + A)^-1 * B
    SLC_DGEMM("N", "N", &p_mut, &m_mut, &n_mut, &dblm1, c, &ldc_mut, b, &ldb_mut, &dbl1, d, &ldd_mut);

    // Scale B by sqrt(2*alpha*beta)
    SLC_DLASCL("G", &int0, &int0, &dbl1, &sqrab2, &n_mut, &m_mut, b, &ldb_mut, info);

    // Compute sqrt(2*alpha*beta) * C * (alpha*I + A)^-1
    SLC_DTRSM("R", "U", "N", "N", &p_mut, &n_mut, &sqrab2, a, &lda_mut, c, &ldc_mut);
    SLC_DTRSM("R", "L", "N", "U", &p_mut, &n_mut, &dbl1, a, &lda_mut, c, &ldc_mut);

    // Apply column interchanges to the solution matrix
    for (i = n-1; i >= 0; i--) {
        ip = iwork[i] - 1; // LAPACK ipiv is 1-based
        if (ip != i) {
            SLC_DSWAP(&p_mut, &c[0 + i*ldc], &int1, &c[0 + ip*ldc], &int1);
        }
    }

    // Compute beta * (alpha*I + A)^-1 * (A - alpha*I) as
    // beta*I - 2*alpha*beta * (alpha*I + A)^-1
    SLC_DGETRI(&n_mut, a, &lda_mut, iwork, dwork, &ldwork_mut, info);

    f64 neg_ab2 = -ab2;
    for (i = 0; i < n; i++) {
        SLC_DSCAL(&n_mut, &neg_ab2, &a[0 + i*lda], &int1);
        a[i + i*lda] += pbeta;
    }

    return;
}
