#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01ud(
    const i32 side,  // 0: left, 1: right
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 m,
    const i32 n,
    const f64 alpha,
    const f64* h,
    const i32 ldh,
    const f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    i32* info
)
{
    const f64 dbl0 = 0.0;
    const i32 int1 = 1;
    *info = 0;

    if ((side != 0) && (side != 1)) {
        *info = -1;
    } else if ((trans != 0) && (trans != 1)) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldh < 1 || ((side == 0) && ldh < m) || ((side == 1) && ldh < n)) {
        *info = -7;
    } else if (lda < MAX(1, m)) {
        *info = -9;
    } else if (ldb < MAX(1, m)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (MIN(m, n) == 0) {
        return;
    }

    if (alpha == dbl0) {
        SLC_DLASET("F", &m, &n, &dbl0, &dbl0, b, &ldb);
        return;
    }

    SLC_DLACPY("F", &m, &n, a, &lda, b, &ldb);
    SLC_DTRMM((side == 0 ? "L" : "R"), "U", (trans == 0 ? "N" : "T"), "N", &m, &n, &alpha, h, &ldh, b, &ldb);

    if (side == 0) {
        if (m > 2) {
            SLC_DSWAP(&(i32){m - 2}, &h[2 + ldh], &(i32){ldh + 1}, &h[2], &int1);
        }

        if (trans == 1) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m - 1; i++) {
                    b[i + j * ldb] += alpha * h[(i + 1)] * a[(i + 1) + j * lda];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 1; i < m; i++) {
                    b[i + j * ldb] += alpha * h[i] * a[(i - 1) + j * lda];
                }
            }
        }

        if (m > 2) {
            SLC_DSWAP(&(i32){m - 2}, &h[2 + 1 * ldh], &(i32){ldh + 1}, &h[2], &int1);
        }

    } else {
        if (trans == 1) {
            for (i32 j = 0; j < n - 1; ++j) {
                if (h[(j + 1) + j * ldh] != dbl0) {
                    SLC_DAXPY(&m, &(f64){alpha * h[(j + 1) + j * ldh]}, &a[j * lda], &int1, &b[(j + 1) * ldb], &int1);
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; ++j) {
                if (h[(j + 1) + j * ldh] != dbl0) {
                    SLC_DAXPY(&m, &(f64){alpha * h[(j + 1) + j * ldh]}, &a[(j + 1) * lda], &int1, &b[j * ldb], &int1);
                }
            }
        }
    }

    return;
}
