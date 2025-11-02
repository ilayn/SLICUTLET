#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01ux(
    const i32 side,
    const i32 uplo,
    const i32 trans,
    const i32 m,
    const i32 n,
    const f64 alpha,
    const f64 *t,
    const i32 ldt,
    f64 *a,
    const i32 lda,
    f64 *dwork,
    const i32 ldwork,
    i32 *info
)
{
    const f64 dbl0 = 0.0, dbl1 = 1.0;
    const i32 int0 = 0, int1 = 1;

    *info = 0;

    i32 k = (side == 0 ? m : n);
    i32 workmin = MAX(1, 2 * (k - 1));
    i32 lquery = (ldwork == -1);

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
    } else if (ldt < MAX(1, k)) {
        *info = -8;
    } else if (lda < MAX(1, m)) {
        *info = -10;
    } else if (alpha != dbl0 && MIN(m, n) > 0) {
        if (lquery) {
            i32 workopt;
            if (side == 0) {
                workopt = (m / 2) * n + m - 1;
            } else {
                workopt = (n / 2) * m + n - 1;
            }
            workopt = MAX(workopt, workmin);
            dwork[0] = (f64)workopt;
            return;
        } else if (ldwork < workmin) {
            dwork[0] = (f64) workmin;
            *info = -12;
        }
    }

    if (*info != 0) {
        return;
    }

    if (MIN(m, n) == 0) {
        return;
    }

    if (alpha == dbl0) {
        SLC_DLASET("F", &m, &n, &dbl0, &dbl0, a, &lda);
        return;
    }

    // Save and count off-diagonal entries of T.
    if (uplo == 0) {
        i32 cnt = k - 1;
        i32 inc = ldt + 1;
        if (cnt > 0) {
            SLC_DCOPY(&cnt, &t[1 + 0 * ldt], &inc, dwork, &int1);
        }
    } else {
        i32 cnt = k - 1;
        i32 inc = ldt + 1;
        if (cnt > 0) {
            SLC_DCOPY(&cnt, &t[0 + 1 * ldt], &inc, dwork, &int1);
        }
    }

    i32 noff = 0;
    for (i32 i = 0; i < k - 1; i++) {
        if (dwork[i] != dbl0) { noff++; }
    }

    // Compute optimal workspace.
    i32 workopt;
    if (side == 0) {
        workopt = noff * n + m - 1;
    } else {
        workopt = noff * m + n - 1;
    }

    i32 psav = k;
    i32 xdif = (trans == 1) ? 1 : 0;
    if (uplo != 0) xdif = 1 - xdif;
    if (side != 0) xdif = 1 - xdif;

    // Save relevant parts of A in the workspace and compute one of
    // the matrix products
    //   A : = alpha*op( triu( T ) ) * A, or
    //   A : = alpha*A * op( triu( T ) ),
    // involving the upper/lower triangle of T.

    i32 pdw = psav;
    if (side == 0) {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m - 1; i++) {
                if (dwork[i] != dbl0) {
                    dwork[pdw] = a[(i + xdif) + j * lda];
                    pdw++;
                }
            }
        }
    } else {
        for (i32 j = 0; j < n - 1; ++j) {
            if (dwork[j] != dbl0) {
                i32 cnt = m;
                SLC_DCOPY(&cnt, &a[0 + (j + xdif) * lda], &int1, &dwork[pdw], &int1);
                pdw += m;
            }
        }
    }

    SLC_DTRMM((side == 0 ? "L" : "R"), (uplo == 0 ? "U" : "L"), (trans == 0 ? "N" : "T"), "N", &m, &n, &alpha, t, &ldt, a, &lda);

    pdw = psav;
    xdif = 1 - xdif;

    if (side == 0) {
        for (i32 j = 0; j < n; ++j) {
            for (i32 i = 0; i < m - 1; ++i) {
                f64 temp = dwork[i];
                if (temp != dbl0) {
                    a[(i + xdif) + j * lda] += alpha * temp * dwork[pdw];
                    ++pdw;
                }
            }
        }
    } else {
        for (i32 j = 0; j < n - 1; ++j) {
            f64 temp = dwork[j] * alpha;
            if (temp != dbl0) {
                i32 cnt = m;
                SLC_DAXPY(&cnt, &temp, &dwork[pdw], &int1, &a[0 + (j + xdif) * lda], &int1);
                pdw += m;
            }
        }
    }

    dwork[0] = (f64)(MAX(workopt, workmin));

    return;
}
