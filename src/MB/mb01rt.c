#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01ot(const i32, const i32, const i32, const f64, const f64, f64*, const i32, const f64*, const i32, const f64*, const i32, i32*);

void
mb01rt(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    const f64* e,
    const i32 lde,
    f64* x,
    const i32 ldx,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    i32 j;
    f64 dbl1 = 1.0, dbl0 = 0.0, dbl2 = 2.0, dblhalf = 0.5;
    i32 int1 = 1, int0 = 0;

    *info = 0;

    if (uplo != 0 && uplo != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldr < MAX(1, n)) {
        *info = -7;
    } else if (lde < 1 || (trans && lde < n) || (!trans && lde < n)) {
        *info = -9;
    } else if (ldx < MAX(1, n)) {
        *info = -11;
    } else if ((beta != 0.0 && ldwork < n*n) ||
               (beta == 0.0 && ldwork < 0)) {
        *info = -13;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n == 0) {
        return;
    }

    if (beta == 0.0) {
        if (alpha == 0.0) {
            // Special case alpha = 0.
            SLC_DLASET((uplo ? "L" : "U"), &n, &n, &dbl0, &dbl0, r, &ldr);
        } else {
            // Special case beta = 0.
            if (alpha != 1.0) {
                SLC_DLASCL((uplo ? "L" : "U"), &int0, &int0, &dbl1, &alpha, &n, &n, r, &ldr, info);
            }
        }
        return;
    }

    // General case: beta <> 0.
    // Compute W = E*T or W = T*E in DWORK, and apply the updating
    // formula (see METHOD section).
    // Workspace: need N*N.
    SLC_DSCAL(&n, &dblhalf, x, &(i32){ldx+1});

    if (!trans) {
        if (!uplo) {
            for (j = 0; j < n; j++) {
                SLC_DCOPY(&(i32){j+1}, &x[j*ldx], &int1, &dwork[j*n], &int1);
                SLC_DTRMV("U", "N", "N", &(i32){j+1}, e, &lde, &dwork[j*n], &int1);
            }
        } else {
            for (j = 0; j < n; j++) {
                SLC_DCOPY(&(i32){j+1}, &x[j], &ldx, &dwork[j*n], &int1);
                SLC_DTRMV("U", "N", "N", &(i32){j+1}, e, &lde, &dwork[j*n], &int1);
            }
        }
    } else {
        if (!uplo) {
            for (j = 0; j < n; j++) {
                SLC_DCOPY(&(i32){j+1}, &e[j*lde], &int1, &dwork[j*n], &int1);
                SLC_DTRMV("U", "N", "N", &(i32){j+1}, x, &ldx, &dwork[j*n], &int1);
            }
        } else {
            for (j = 0; j < n; j++) {
                SLC_DCOPY(&(i32){j+1}, &e[j*lde], &int1, &dwork[j*n], &int1);
                SLC_DTRMV("L", "T", "N", &(i32){j+1}, x, &ldx, &dwork[j*n], &int1);
            }
        }
    }

    SLC_DSCAL(&n, &dbl2, x, &(i32){ldx+1});

    mb01ot(uplo, trans, n, alpha, beta, r, ldr, e, lde, dwork, n, info);

    return;
}
