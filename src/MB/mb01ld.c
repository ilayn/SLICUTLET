#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01ld(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 m,
    const i32 n,
    const i32 k,
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
    (void)k; // Unused parameter
    i32 i, j, m2;
    f64 dbl1 = 1.0, dbl0 = 0.0;
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
    } else if ((lda < 1) || ((trans == 1) && (lda < n)) || (trans == 0 && lda < m)) {
        *info = -10;
    } else if ((ldx < MAX(1, n)) || ((ldx < m) && (uplo == 0) && (ldwork < m*(n-1)))) {
        *info = -12;
    } else if ((ldwork < 0) || ((beta != 0.0) && (m > 1) && (n > 1) && (ldwork < n))) {
        *info = -14;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (m <= 0) {
        return;
    }

    m2 = MIN(2, m);
    if ((beta == 0.0) || (n <= 1)) {
        if (uplo == 0) {
            i = 1;
            j = m2;
        } else {
            i = m2;
            j = 1;
        }

        if (alpha == 0.0) {
            // Special case alpha = 0
            SLC_DLASET((uplo ? "L" : "U"), &(i32){m-1}, &(i32){m-1}, &dbl0, &dbl0, &r[i-1 + (j-1)*ldr], &ldr);
        } else {
            // Special case beta = 0 or n <= 1
            if (alpha != 1.0) {
                SLC_DLASCL((uplo ? "L" : "U"), &int0, &int0, &dbl1, &alpha, &(i32){m-1}, &(i32){m-1}, &r[i-1 + (j-1)*ldr], &ldr, info);
            }
        }
        return;
    }

    // General case beta != 0.
    if (uplo == 0) {
        i = 1;
        j = m2;
    } else {
        i = m2;
        j = 1;
    }

    if (!trans) {
        SLC_DLACPY("F", &m, &(i32){n-1}, &a[(i-1)*lda], &lda, dwork, &m);
        SLC_DTRMM("R", (uplo ? "L" : "U"), "N", "N", &m, &(i32){n-1}, &dbl1, &x[(i-1) + (j-1)*ldx], &ldx, dwork, &m);
        SLC_DSYR2K((uplo ? "L" : "U"), (trans ? "T" : "N"), &m, &(i32){n-1}, &beta, dwork, &m, &a[(j-1)*lda], &lda, &alpha, r, &ldr);
    } else {
        SLC_DLACPY("F", &(i32){n-1}, &m, &a[j-1], &lda, dwork, &(i32){n-1});
        SLC_DTRMM("L", (uplo ? "L" : "U"), "N", "N", &(i32){n-1}, &m, &dbl1, &x[(i-1) + (j-1)*ldx], &ldx, dwork, &(i32){n-1});
        SLC_DSYR2K((uplo ? "L" : "U"), (trans ? "T" : "N"), &m, &(i32){n-1}, &beta, &a[i-1], &lda, dwork, &(i32){n-1}, &alpha, r, &ldr);
    }

    return;
}
