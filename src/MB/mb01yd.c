#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01yd(
    const i32 uplo,   // 0: upper, 1: lower
    const i32 trans,  // 0: no transpose, 1: transpose
    const i32 n,
    const i32 k,
    const i32 l,
    const f64 alpha,
    const f64 beta,
    const f64* a,
    const i32 lda,
    f64* c,
    const i32 ldc,
    i32* info
)
{
    i32 i, j, m, ncola, nrowa;
    f64 temp;
    f64 dbl1 = 1.0, dbl0 = 0.0;
    i32 int1 = 1, int0 = 0;

    *info = 0;

    if (trans) {
        nrowa = k;
        ncola = n;
    } else {
        nrowa = n;
        ncola = k;
    }

    if (uplo == 0) {
        m = nrowa;
    } else {
        m = ncola;
    }

    if (uplo != 0 && uplo != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (k < 0) {
        *info = -4;
    } else if (l < 0 || l > MAX(0, m-1)) {
        *info = -5;
    } else if (lda < MAX(1, nrowa)) {
        *info = -9;
    } else if (ldc < MAX(1, n)) {
        *info = -11;
    }
    if (*info != 0) {
        return;
    }

    // Quick return, if possible.
    if ((n == 0) || (((alpha == 0.0) || (k == 0)) && (beta == 1.0))) {
        return;
    }

    if (alpha == 0.0) {
        if (beta == 0.0) {
            // Special case when both alpha = 0 and beta = 0.
            SLC_DLASET((uplo ? "L" : "U"), &n, &n, &dbl0, &dbl0, c, &ldc);
        } else {
            // Special case alpha = 0.
            SLC_DLASCL((uplo ? "L" : "U"), &int0, &int0, &dbl1, &beta, &n, &n, c, &ldc, info);
        }
        return;
    }

    // General case: alpha <> 0.
    if (!trans) {
        // Form  C := alpha*A*A' + beta*C.
        if (uplo == 0) {
            for (j = 0; j < n; j++) {
                if (beta == 0.0) {
                    for (i = 0; i <= j; i++) {
                        c[i + j*ldc] = 0.0;
                    }
                } else if (beta != 1.0) {
                    SLC_DSCAL(&(i32){j+1}, &beta, &c[j*ldc], &int1);
                }

                for (m = MAX(0, j-l); m < k; m++) {
                    SLC_DAXPY(&(i32){MIN(j+1, l+m+1)}, &(f64){alpha*a[j + m*lda]},
                              &a[m*lda], &int1, &c[j*ldc], &int1);
                }
            }

        } else {
            for (j = 0; j < n; j++) {
                if (beta == 0.0) {
                    for (i = j; i < n; i++) {
                        c[i + j*ldc] = 0.0;
                    }
                } else if (beta != 1.0) {
                    SLC_DSCAL(&(i32){n-j}, &beta, &c[j + j*ldc], &int1);
                }

                for (m = 0; m < MIN(j+l+1, k); m++) {
                    SLC_DAXPY(&(i32){n-j}, &(f64){alpha*a[j + m*lda]},
                              &a[j + m*lda], &int1, &c[j + j*ldc], &int1);
                }
            }
        }

    } else {
        // Form  C := alpha*A'*A + beta*C.
        if (uplo == 0) {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    temp = alpha * SLC_DDOT(&(i32){MIN(j+l+1, k)}, &a[i*lda], &int1, &a[j*lda], &int1);
                    if (beta == 0.0) {
                        c[i + j*ldc] = temp;
                    } else {
                        c[i + j*ldc] = temp + beta * c[i + j*ldc];
                    }
                }
            }

        } else {
            for (j = 0; j < n; j++) {
                for (i = j; i < n; i++) {
                    m = MAX(0, i-l);
                    temp = alpha * SLC_DDOT(&(i32){k-m}, &a[m + i*lda], &int1, &a[m + j*lda], &int1);
                    if (beta == 0.0) {
                        c[i + j*ldc] = temp;
                    } else {
                        c[i + j*ldc] = temp + beta * c[i + j*ldc];
                    }
                }
            }
        }
    }

    return;
}
