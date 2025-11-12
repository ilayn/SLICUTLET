#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01rx(
    const i32 side,  // 0: left, 1: right
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
    const f64* b,
    const i32 ldb,
    i32* info
)
{
    i32 j;
    f64 dbl1 = 1.0, dbl0 = 0.0;
    i32 int1 = 1, int0 = 0;

    *info = 0;
    if (side != 0 && side != 1) {
        *info = -1;
    } else if (uplo != 0 && uplo != 1) {
        *info = -2;
    } else if (trans != 0 && trans != 1) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldr < MAX(1, m)) {
        *info = -9;
    } else if (lda < 1 ||
               (((side == 0 && trans == 0) || (side == 1 && trans == 1)) && lda < m) ||
               (((side == 0 && trans == 1) || (side == 1 && trans == 0)) && lda < n)) {
        *info = -11;
    } else if (ldb < 1 ||
               (side == 0 && ldb < n) ||
               (side == 1 && ldb < m)) {
        *info = -13;
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
    // Compute the required triangle of (1) or (2) using BLAS 2 operations.
    if (side == 0) {
        if (!uplo) {
            if (trans) {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("T", &n, &(i32){j+1}, &beta, a, &lda, &b[j*ldb], &int1, &alpha, &r[j*ldr], &int1);
                }
            } else {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("T", &(i32){j+1}, &n, &beta, a, &lda, &b[j*ldb], &int1, &alpha, &r[j*ldr], &int1);
                }
            }
        } else {
            if (trans) {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("T", &n, &(i32){m-j}, &beta, &a[j*lda], &lda, &b[j*ldb], &int1, &alpha, &r[j + j*ldr], &int1);
                }
            } else {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("T", &(i32){m-j}, &n, &beta, &a[j], &lda, &b[j*ldb], &int1, &alpha, &r[j + j*ldr], &int1);
                }
            }
        }
    } else {
        if (!uplo) {
            if (trans) {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("N", &(i32){j+1}, &n, &beta, b, &ldb, &a[j], &lda, &alpha, &r[j*ldr], &int1);
                }
            } else {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("N", &(i32){j+1}, &n, &beta, b, &ldb, &a[j*lda], &int1, &alpha, &r[j*ldr], &int1);
                }
            }
        } else {
            if (trans) {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("N", &(i32){m-j}, &n, &beta, &b[j], &ldb, &a[j], &lda, &alpha, &r[j + j*ldr], &int1);
                }
            } else {
                for (j = 0; j < m; j++) {
                    SLC_DGEMV("N", &(i32){m-j}, &n, &beta, &b[j], &ldb, &a[j*lda], &int1, &alpha, &r[j + j*ldr], &int1);
                }
            }
        }
    }

    return;
}
