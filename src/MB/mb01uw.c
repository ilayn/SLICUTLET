#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01uw(
    const i32 side,
    const i32 trans,
    const i32 m,
    const i32 n,
    const f64 alpha,
    const f64 *h,
    const i32 ldh,
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

    if (side != 0 && side != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldh < 1 || (side == 0 && ldh < m) || (side == 1 && ldh < n)) {
        *info = -7;
    } else if (lda < MAX(1, m)) {
        *info = -9;
    } else if (ldwork < 0 ||
               (alpha != dbl0 && MIN(m, n) > 0 &&
                ((side == 0 && ldwork < m - 1) ||
                 (side == 1 && ldwork < n - 1)))) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (MIN(m, n) == 0) {
        return;
    }

    if (side == 0) {
        if (m == 1) {
            f64 scale = alpha * h[0];
            SLC_DSCAL(&n, &scale, a, &lda);
            return;
        }
    } else {
        if (n == 1) {
            f64 scale = alpha * h[0];
            SLC_DSCAL(&m, &scale, a, &int1);
            return;
        }
    }

    if (alpha == dbl0) {
        SLC_DLASET("F", &m, &n, &dbl0, &dbl0, a, &lda);
        return;
    }

    /* Save A to workspace and multiply */
    SLC_DLACPY("F", &m, &n, a, &lda, dwork, &m);
    SLC_DTRMM((side == 0 ? "L" : "R"), "U", (trans == 0 ? "N" : "T"), "N", &m, &n, &alpha, h, &ldh, a, &lda);

    if (side == 0) {
        if (m > 2) {
            SLC_DSWAP(&(i32){m - 2}, &h[2 + 1 * ldh], &(i32){ldh + 1}, &h[2 + 0 * ldh], &int1);
        }

        if (trans == 1) {
            i32 jw = 0;
            for (i32 j = 0; j < n; ++j) {
                jw++;
                for (i32 i = 0; i < m - 1; ++i) {
                    a[i + j * lda] += alpha * h[(i + 1) + 0 * ldh] * dwork[jw];
                    jw++;
                }
            }
        } else {
            i32 jw = -1;
            for (i32 j = 0; j < n; ++j) {
                jw++;
                for (i32 i = 1; i < m; ++i) {
                    a[i + j * lda] += alpha * h[i + 0 * ldh] * dwork[jw];
                    jw++;
                }
            }
        }

        if (m > 2) {
            i32 cnt = m - 2;
            i32 inc = ldh + 1;
            SLC_DSWAP(&cnt, &h[2 + 1 * ldh], &inc, &h[2 + 0 * ldh], &int1);
        }

    } else {
        if (trans == 1) {
            for (i32 j = 0; j < n - 1; ++j) {
                if (h[(j + 1) + j * ldh] != dbl0) {
                    f64 coeff = alpha * h[(j + 1) + j * ldh];
                    SLC_DAXPY(&m, &coeff, &dwork[j * m], &int1, &a[(j + 1) * lda], &int1);
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; ++j) {
                if (h[(j + 1) + j * ldh] != dbl0) {
                    f64 coeff = alpha * h[(j + 1) + j * ldh];
                    SLC_DAXPY(&m, &coeff, &dwork[(j + 1) * m], &int1, &a[j * lda], &int1);
                }
            }
        }
    }
}
