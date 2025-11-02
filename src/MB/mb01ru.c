#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01ru(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 m,
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    const f64* a,
    const i32 lda,
    f64* x,
    const i32 ldx,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    f64 dbl1 = 1.0, dbl0 = 0.0, dbl2 = 2.0, dblhalf = 0.5;
    i32 int0 = 0;

    *info = 0;

    if (uplo != 0 && uplo != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldr < MAX(1, m)) {
        *info = -8;
    } else if (lda < 1 || (trans && lda < n) || (!trans && lda < m)) {
        *info = -10;
    } else if (ldx < MAX(1, n)) {
        *info = -12;
    } else if ((beta != 0.0 && ldwork < m*n) ||
               (beta == 0.0 && ldwork < 0)) {
        *info = -14;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (m == 0) {
        return;
    }

    if (beta == 0.0 || n == 0) {
        if (alpha == 0.0) {
            // Special case alpha = 0.
            SLC_DLASET((uplo ? "L" : "U"), &m, &m, &dbl0, &dbl0, r, &ldr);
        } else {
            // Special case beta = 0 or N = 0.
            if (alpha != 1.0) {
                SLC_DLASCL((uplo ? "L" : "U"), &int0, &int0, &dbl1, &alpha, &m, &m, r, &ldr, info);
            }
        }
        return;
    }

    // General case: beta <> 0.
    // Compute W = op( A )*T or W = T*op( A ) in DWORK, and apply the
    // updating formula (see METHOD section).
    // Workspace: need M*N.
    SLC_DSCAL(&n, &dblhalf, x, &(i32){ldx+1});

    if (trans) {
        SLC_DLACPY("F", &n, &m, a, &lda, dwork, &n);
        SLC_DTRMM("L", (uplo ? "L" : "U"), "N", "N", &n, &m, &dbl1, x, &ldx, dwork, &n);
        SLC_DSYR2K((uplo ? "L" : "U"), "T", &m, &n, &beta, dwork, &n, a, &lda, &alpha, r, &ldr);
    } else {
        SLC_DLACPY("F", &m, &n, a, &lda, dwork, &m);
        SLC_DTRMM("R", (uplo ? "L" : "U"), "N", "N", &m, &n, &dbl1, x, &ldx, dwork, &m);
        SLC_DSYR2K((uplo ? "L" : "U"), "N", &m, &n, &beta, dwork, &m, a, &lda, &alpha, r, &ldr);
    }

    SLC_DSCAL(&n, &dbl2, x, &(i32){ldx+1});

    return;
}
