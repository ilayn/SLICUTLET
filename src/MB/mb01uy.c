#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01uy(
    const i32 side,
    const i32 uplo,
    const i32 trans,
    const i32 m,
    const i32 n,
    const f64 alpha,
    f64 *t,
    const i32 ldt,
    const f64 *a,
    const i32 lda,
    f64 *dwork,
    const i32 ldwork,
    i32 *info
)
{
    const f64 dbl0 = 0.0, dbl1 = 1.0;
    const i32 int0 = 0, int1 = 1, minmn = MIN(m, n), maxmn = MAX(m, n);
    i32 k, l;

    *info = 0;

    if (side == 0) {
        k = m;
        l = n;
    } else {
        k = n;
        l = m;
    }

    // Ensure that at least two rows or columns of A fit into the
    // workspace, if optimal workspace is required.
    i32 workmin = 1;
    if (alpha != dbl0 && minmn > 0) {
        workmin = MAX(workmin, k);
    }
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
    } else if ((ldt < MAX(1, k)) || (side == 1) && (ldt < n)) {
        *info = -8;
    } else if (lda < MAX(1, m)) {
        *info = -10;
    } else if (lquery) {
        if ((alpha != dbl0) && (minmn > 0)) {
            SLC_DGEQRF(&m, &maxmn, a, &lda, dwork, dwork, &(i32){-1}, info);
            i32 workopt = MAX(workmin, MAX(2*l, (i32)dwork[0]));
            dwork[0] = (f64)workopt;
        } else {
            dwork[0] = (f64) workmin;
        }
        return;
    } else if (ldwork < workmin) {
        dwork[0] = (f64) workmin;
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (minmn == 0) {
        return;
    }

    if (alpha == dbl0) {
        SLC_DLASET("F", &m, &n, &dbl0, &dbl0, t, &ldt);
        return;
    }

    SLC_DLACPY("A", &m, &n, a, &lda, dwork, &m);
    SLC_DTRMM((side == 0 ? "L" : "R"), (uplo == 0 ? "U" : "L"), (trans == 0 ? "N" : "T"), "N", &m, &n, &alpha, t, &ldt, dwork, &m);
    SLC_DLACPY("A", &m, &n, dwork, &m, t, &ldt);

    return;
}
