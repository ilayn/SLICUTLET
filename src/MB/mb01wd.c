#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01yd(const i32, const i32, const i32, const i32, const i32, const f64, const f64, const f64*, const i32, f64*, const i32, i32*);
void mb01zd(const i32, const i32 uplo, const i32, const i32, const i32, const i32, const i32, const f64, const f64*, const i32, f64*, const i32, i32*);

void mb01wd(
    const i32 dico,  // 0: continuous, 1: discrete
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 hess,  // 0: General, 1: Hessenberg
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64 *r,
    const i32 ldr,
    f64 *a,
    const i32 lda,
    f64 *t,
    const i32 ldt,
    i32 *info
)
{
    const f64 dbl0 = 0.0, dbl1 = 1.0;
    const i32 int0 = 0, int1 = 1;
    i32* info2;
    char* side, negtrans;

    *info = 0;

    if (dico != 0 && dico != 1) {
        *info = -1;
    } else if (uplo != 0 && uplo != 1) {
        *info = -2;
    } else if (trans != 0 && trans != 1) {
        *info = -3;
    } else if (hess != 0 && hess != 1) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldr < MAX(1, n)) {
        *info = -9;
    } else if (lda < MAX(1, n)) {
        *info = -11;
    } else if (ldt < MAX(1, n)) {
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n == 0) {
        return;
    }

    if (alpha == 0.0) {
        if (beta == 0.0) {
            // Special case alpha = 0 and beta = 0.
            SLC_DLASET((uplo ? "L" : "U"), &n, &n, &dbl0, &dbl0, r, &ldr);
        } else {
            // Special case alpha = 0.
            if (beta != 1.0) {
                SLC_DLASCL((uplo ? "L" : "U"), &int0, &int0, &dbl1, &beta, &n, &n, r, &ldr, info);
            }
        }
        return;
    }

    // General case: alpha <> 0.
    // Compute (in A) T*A, if TRANS = 'N', or
    //                A*T, otherwise.
    if (trans == 1) {
        side = "R";
        negtrans = "N";
    } else {
        side = "L";
        negtrans = "T";
    }

    if ((hess) && (n > 2)) {
        mb01zd(1, uplo, 0, 0, n, n, 1, 1.0, t, ldt, a, ldt, info);
    } else {
        SLC_DTRMM(side, (uplo ? "L" : "U"), "N", "N", &n, &n, &dbl1, t, &ldt, a, &lda);
    }


    if (dico == 0) {
        // Compute (in A) alpha*T'*T*A, if TRANS = 'N', or
        //                alpha*A*T*T', otherwise.
        if ((hess) && (n > 2)) {
            mb01zd(1, uplo, 0, 0, n, n, 1, alpha, t, ldt, a, ldt, info);
        } else {
            SLC_DTRMM(side, (uplo ? "L" : "U"), negtrans, "N", &n, &n, &alpha, t, &ldt, a, &lda);
        }

        if (uplo == 0) {
            if (beta == 0.0) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j*ldr] = a[i + j*lda] + a[j + i*lda];
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j*ldr] = a[i + j*lda] + a[j + i*lda] + beta*r[i + j*ldr];
                    }
                }
            }
        } else {
            if (beta == 0.0) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j*ldr] = a[i + j*lda] + a[j + i*lda];
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j*ldr] = a[i + j*lda] + a[j + i*lda] + beta*r[i + j*ldr];
                    }
                }
            }
        }
    } else {
        // Compute (in R) alpha*A'*T'*T*A + beta*R, if TRANS = 'N', or
        //                alpha*A*T*T'*A' + beta*R, otherwise.

        if ((hess) && (n > 2)) {
            mb01yd(uplo, 1 - trans, n, n, 1, alpha, beta, a, lda, r, ldr, &info2);
        } else {
            SLC_DSYRK((uplo ? "L" : "U"), negtrans, &n, &n, &alpha, a, &lda, &beta, r, &ldr);
        }

        // Compute (in R) -alpha*T'*T + R, if TRANS = 'N', or
        //                -alpha*T*T' + R, otherwise.
        mb01yd(uplo, 1 - trans, n, n, 0, -alpha, 1.0, t, ldt, r, ldr, &info2);
    }

    return;
}
