#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01oe(const i32, const i32, const i32, const f64, const f64, f64*, const i32, const f64*, const i32, const f64*, const i32);

void
mb01od(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    f64* h,
    const i32 ldh,
    f64* x,
    const i32 ldx,
    const f64* e,
    const i32 lde,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    i32 i, j, j1;
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
    } else if (ldh < 1 || (trans && ldh < n) || (!trans && ldh < n)) {
        *info = -9;
    } else if (ldx < MAX(1, n)) {
        *info = -11;
    } else if ((beta != 0.0 && ldwork < n*n) ||
               (beta == 0.0 && ldwork < 0)) {
        *info = -15;
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
    // Compute W = H*U or W = U*H in DWORK, and apply the corresponding
    // updating formula (see METHOD section).
    // Workspace: need N*N.
    SLC_DSCAL(&n, &dblhalf, x, &(i32){ldx+1});

    if (!trans) {
        // For convenience, swap the subdiagonal entries in H with
        // those in the first column, and finally restore them.
        if (n > 2) {
            SLC_DSWAP(&(i32){n-2}, &h[2], &int1, &h[2+ldh], &(i32){ldh+1});
        }

        if (!uplo) {
            for (j = 0; j < n-1; j++) {
                j1 = j + 1;
                SLC_DCOPY(&(i32){j+1}, &x[j*ldx], &int1, &dwork[j*n], &int1);
                SLC_DTRMV("U", "N", "N", &(i32){j+1}, h, &ldh, &dwork[j*n], &int1);
                for (i = 1; i <= j; i++) {
                    dwork[i + j*n] = dwork[i + j*n] + h[i] * x[i-1 + j*ldx];
                }
                dwork[j1 + j*n] = h[j1] * x[j + j*ldx];
            }

            SLC_DCOPY(&n, &x[(n-1)*ldx], &int1, &dwork[(n-1)*n], &int1);
            SLC_DTRMV("U", "N", "N", &n, h, &ldh, &dwork[(n-1)*n], &int1);

            for (i = 1; i < n; i++) {
                dwork[i + (n-1)*n] = dwork[i + (n-1)*n] + h[i] * x[i-1 + (n-1)*ldx];
            }

        } else {
            for (j = 0; j < n-1; j++) {
                j1 = j + 1;
                SLC_DCOPY(&(i32){j+1}, &x[j], &ldx, &dwork[j*n], &int1);
                SLC_DTRMV("U", "N", "N", &(i32){j+1}, h, &ldh, &dwork[j*n], &int1);
                for (i = 1; i <= j; i++) {
                    dwork[i + j*n] = dwork[i + j*n] + h[i] * x[j + (i-1)*ldx];
                }
                dwork[j1 + j*n] = h[j1] * x[j + j*ldx];
            }

            SLC_DCOPY(&n, &x[n-1], &ldx, &dwork[(n-1)*n], &int1);
            SLC_DTRMV("U", "N", "N", &n, h, &ldh, &dwork[(n-1)*n], &int1);

            for (i = 1; i < n; i++) {
                dwork[i + (n-1)*n] = dwork[i + (n-1)*n] + h[i] * x[n-1 + (i-1)*ldx];
            }
        }

        if (n > 2) {
            SLC_DSWAP(&(i32){n-2}, &h[2], &int1, &h[2+ldh], &(i32){ldh+1});
        }

    } else {
        if (!uplo) {
            for (j = 0; j < n-1; j++) {
                j1 = j + 1;
                SLC_DCOPY(&(i32){j+1}, &h[j*ldh], &int1, &dwork[j*n], &int1);
                SLC_DTRMV("U", "N", "N", &(i32){j+1}, x, &ldx, &dwork[j*n], &int1);
                SLC_DAXPY(&(i32){j+1}, &h[j1 + j*ldh], &x[j1*ldx], &int1, &dwork[j*n], &int1);
                dwork[j1 + j*n] = h[j1 + j*ldh] * x[j1 + j1*ldx];
            }

            SLC_DCOPY(&n, &h[(n-1)*ldh], &int1, &dwork[(n-1)*n], &int1);
            SLC_DTRMV("U", "N", "N", &n, x, &ldx, &dwork[(n-1)*n], &int1);

        } else {
            for (j = 0; j < n-1; j++) {
                j1 = j + 1;
                SLC_DCOPY(&(i32){j+1}, &h[j*ldh], &int1, &dwork[j*n], &int1);
                SLC_DTRMV("L", "T", "N", &(i32){j+1}, x, &ldx, &dwork[j*n], &int1);
                SLC_DAXPY(&(i32){j+1}, &h[j1 + j*ldh], &x[j1], &ldx, &dwork[j*n], &int1);
                dwork[j1 + j*n] = h[j1 + j*ldh] * x[j1 + j1*ldx];
            }

            SLC_DCOPY(&n, &h[(n-1)*ldh], &int1, &dwork[(n-1)*n], &int1);
            SLC_DTRMV("L", "T", "N", &n, x, &ldx, &dwork[(n-1)*n], &int1);
        }
    }

    mb01oe(uplo, trans, n, alpha, beta, r, ldr, dwork, n, e, lde);

    // Compute W = E*U or W = U*E in DWORK, and apply the corresponding
    // updating formula (see METHOD section).
    // Workspace: need N*N.
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

    mb01oe(uplo, trans, n, dbl1, beta, r, ldr, h, ldh, dwork, n);

    SLC_DSCAL(&n, &dbl2, x, &(i32){ldx+1});

    return;
}
