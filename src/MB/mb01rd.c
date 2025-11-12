#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01rd(
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
    i32 j, jwork, ldw, nrowa;
    f64 dbl1 = 1.0, dbl0 = 0.0, dblhalf = 0.5;
    i32 int1 = 1, int0 = 0;
    const char* ntran;

    *info = 0;

    if (trans) {
        nrowa = n;
        ntran = "N";
    } else {
        nrowa = m;
        ntran = "T";
    }

    ldw = MAX(1, m);

    if (uplo != 0 && uplo != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldr < ldw) {
        *info = -8;
    } else if (lda < MAX(1, nrowa)) {
        *info = -10;
    } else if (ldx < MAX(1, n)) {
        *info = -12;
    } else if ((beta != 0.0 && ldwork < MAX(1, m*n)) ||
               (beta == 0.0 && ldwork < 1)) {
        *info = -14;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    SLC_DSCAL(&n, &dblhalf, x, &(i32){ldx+1});
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

    // General case: beta <> 0. Efficiently compute
    //    _
    //    R = alpha*R + beta*op( A )*X*op( A )',
    //
    // as described in the Method section.
    //
    // Compute W = beta*op( A )*T in DWORK.
    // Workspace: need M*N.
    if (trans) {
        jwork = 0;
        for (j = 0; j < n; j++) {
            SLC_DCOPY(&m, &a[j], &lda, &dwork[jwork], &int1);
            jwork += ldw;
        }
    } else {
        SLC_DLACPY("F", &m, &n, a, &lda, dwork, &ldw);
    }

    SLC_DTRMM("R", (uplo ? "L" : "U"), "N", "N", &m, &n, &beta, x, &ldx, dwork, &ldw);

    // Compute Y = alpha*V + W*op( A )' in R. First, set to zero the
    // strictly triangular part of R not specified by UPLO. That part
    // will then contain beta*stri( B ).
    if (alpha != 0.0) {
        if (m > 1) {
            if (!uplo) {
                SLC_DLASET("L", &(i32){m-1}, &(i32){m-1}, &dbl0, &dbl0, &r[1], &ldr);
            } else {
                SLC_DLASET("U", &(i32){m-1}, &(i32){m-1}, &dbl0, &dbl0, &r[ldr], &ldr);
            }
        }
        SLC_DSCAL(&m, &dblhalf, r, &(i32){ldr+1});
    }

    SLC_DGEMM("N", ntran, &m, &m, &n, &dbl1, dwork, &ldw, a, &lda, &alpha, r, &ldr);

    // Add the term corresponding to B', with B = op( A )*T*op( A )'.
    if (!uplo) {
        for (j = 0; j < m; j++) {
            SLC_DAXPY(&(i32){j+1}, &dbl1, &r[j], &ldr, &r[j*ldr], &int1);
        }
    } else {
        for (j = 0; j < m; j++) {
            SLC_DAXPY(&(i32){j+1}, &dbl1, &r[j*ldr], &int1, &r[j], &ldr);
        }
    }

    return;
}
