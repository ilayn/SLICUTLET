#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01os(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 n,
    const f64* h,
    const i32 ldh,
    const f64* x,
    const i32 ldx,
    f64* p,
    const i32 ldp,
    i32* info
)
{
    i32 i, j, j3;
    f64 temp;
    f64 dbl1 = 1.0, dbl0 = 0.0;
    i32 int1 = 1;

    *info = 0;
    if (uplo != 0 && uplo != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldh < MAX(1, n)) {
        *info = -5;
    } else if (ldx < MAX(1, n)) {
        *info = -7;
    } else if (ldp < MAX(1, n)) {
        *info = -9;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n == 0) {
        return;
    }

    if (!trans) {
        if (!uplo) {
            // Compute  P := H*U + H*sU'.
            for (j = 0; j < n-1; j++) {
                SLC_DCOPY(&(i32){j+1}, &x[j*ldx], &int1, &p[j*ldp], &int1);
                SLC_DTRMV((uplo ? "L" : "U"), "N", "N", &(i32){j+1}, h, &ldh, &p[j*ldp], &int1);
                for (i = j+1; i < n; i++) {
                    p[i + j*ldp] = 0.0;
                }

                for (i = 1; i <= j+1; i++) {
                    p[i + j*ldp] = p[i + j*ldp] + h[i + (i-1)*ldh]*x[(i-1) + j*ldx];
                }

                for (i = j+2; i < n; i++) {
                    SLC_DAXPY(&(i32){i+1}, &x[j + (i-1)*ldx], &h[(i-1)*ldh], &int1, &p[j*ldp], &int1);
                }

                SLC_DAXPY(&n, &x[j + (n-1)*ldx], &h[(n-1)*ldh], &int1, &p[j*ldp], &int1);
            }

            SLC_DCOPY(&n, &x[(n-1)*ldx], &int1, &p[(n-1)*ldp], &int1);
            SLC_DTRMV((uplo ? "L" : "U"), "N", "N", &n, h, &ldh, &p[(n-1)*ldp], &int1);

            for (i = 1; i < n; i++) {
                p[i + (n-1)*ldp] = p[i + (n-1)*ldp] + h[i + (i-1)*ldh]*x[(i-1) + (n-1)*ldx];
            }
        } else {
            // Compute  P := H*L + H*sL'.
            // There is no contribution from sL' for the first column.
            SLC_DCOPY(&n, x, &int1, p, &int1);
            SLC_DTRMV("U", "N", "N", &n, h, &ldh, p, &int1);

            for (i = 1; i < n; i++) {
                p[i] = p[i] + h[i + (i-1)*ldh]*x[(i-1)];
            }

            for (j = 1; j < n; j++) {
                // Compute the contribution from H*sL'.
                SLC_DCOPY(&j, &x[j], &ldx, &p[j*ldp], &int1);
                SLC_DTRMV("U", "N", "N", &j, h, &ldh, &p[j*ldp], &int1);
                p[j + j*ldp] = 0.0;

                for (i = 1; i <= j; i++) {
                    p[i + j*ldp] = p[i + j*ldp] + h[i + (i-1)*ldh]*x[j + (i-1)*ldx];
                }

                // Compute the contribution from H*L.
                temp = p[j + j*ldp];
                SLC_DGEMV("N", &j, &(i32){n-j}, &dbl1, &h[j*ldh], &ldh, &x[j + j*ldx], &int1, &dbl1, &p[j*ldp], &int1);
                SLC_DCOPY(&(i32){n-j}, &x[j + j*ldx], &int1, &p[j + j*ldp], &int1);
                SLC_DTRMV("U", "N", "N", &(i32){n-j}, &h[j + j*ldh], &ldh, &p[j + j*ldp], &int1);
                p[j + j*ldp] = p[j + j*ldp] + temp;

                for (i = j+1; i < n; i++) {
                    p[i + j*ldp] = p[i + j*ldp] + h[i + (i-1)*ldh]*x[(i-1) + j*ldx];
                }
            }
        }
    } else {
        if (!uplo) {
            // Compute  P := U*H + sU'*H.
            for (j = 0; j < n-2; j++) {
                j3 = MIN(j+3, n);
                SLC_DCOPY(&(i32){j+2}, &h[j*ldh], &int1, &p[j*ldp], &int1);
                SLC_DTRMV((uplo ? "L" : "U"), "N", "N", &(i32){j+2}, x, &ldx, &p[j*ldp], &int1);
                SLC_DCOPY(&(i32){j+2}, &h[j*ldh], &int1, &p[1 + (j+1)*ldp], &int1);
                SLC_DTRMV((uplo ? "L" : "U"), "T", "N", &(i32){j+2}, &x[ldx], &ldx, &p[1 + (j+1)*ldp], &int1);
                SLC_DAXPY(&(i32){j+1}, &dbl1, &p[1 + (j+1)*ldp], &int1, &p[1 + j*ldp], &int1);
                p[(j+2) + j*ldp] = SLC_DDOT(&(i32){j+2}, &x[(j+2)*ldx], &int1, &h[j*ldh], &int1);
                SLC_DGEMV("T", &(i32){j+2}, &(i32){n-j3}, &dbl1, &x[j3*ldx], &ldx, &h[j*ldh], &int1, &dbl1, &p[j3 + j*ldp], &int1);
                p[(n-1) + j*ldp] = SLC_DDOT(&(i32){j+2}, &x[(n-1)*ldx], &int1, &h[j*ldh], &int1);
            }

            if (n == 1) {
                p[0] = x[0]*h[0];
            } else {
                SLC_DCOPY(&n, &h[(n-2)*ldh], &int1, &p[(n-2)*ldp], &int1);
                SLC_DCOPY(&n, &h[(n-1)*ldh], &int1, &p[(n-1)*ldp], &int1);
                SLC_DTRMM("L", (uplo ? "L" : "U"), "N", "N", &n, &(i32){2}, &dbl1, x, &ldx, &p[(n-2)*ldp], &ldp);

                for (i = 1; i < n; i++) {
                    SLC_DGEMV("T", &i, &(i32){2}, &dbl1, &h[(n-2)*ldh], &ldh, &x[i*ldx], &int1, &dbl1, &p[i + (n-2)*ldp], &ldp);
                }
            }
        } else {
            // Compute  P := L*H + sL'*H.
            for (j = 0; j < n-1; j++) {
                SLC_DCOPY(&(i32){j+1}, &h[j*ldh], &int1, &p[j*ldp], &int1);
                SLC_DTRMV((uplo ? "L" : "U"), "N", "N", &(i32){j+1}, x, &ldx, &p[j*ldp], &int1);
                SLC_DCOPY(&(i32){j+1}, &h[1 + j*ldh], &int1, &p[(j+1)*ldp], &int1);
                SLC_DTRMV((uplo ? "L" : "U"), "T", "N", &(i32){j+1}, &x[1], &ldx, &p[(j+1)*ldp], &int1);
                SLC_DAXPY(&(i32){j+1}, &dbl1, &p[(j+1)*ldp], &int1, &p[j*ldp], &int1);
                SLC_DGEMV("N", &(i32){n-j-1}, &(i32){j+2}, &dbl1, &x[(j+1)], &ldx, &h[j*ldh], &int1, &dbl0, &p[j+1 + j*ldp], &int1);
            }

            SLC_DCOPY(&n, &h[(n-1)*ldh], &int1, &p[(n-1)*ldp], &int1);
            SLC_DTRMV((uplo ? "L" : "U"), "N", "N", &n, x, &ldx, &p[(n-1)*ldp], &int1);

            for (i = 0; i < n-1; i++) {
                p[i + (n-1)*ldp] = p[i + (n-1)*ldp] + SLC_DDOT(&(i32){n-i-1}, &x[i+1 + i*ldx], &int1, &h[i+1 + (n-1)*ldh], &int1);
            }
        }
    }

    return;
}
