#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01oe(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    const f64* h,
    const i32 ldh,
    f64* e,
    const i32 lde,
    i32* info
)
{
    i32 i, j, j1;
    f64 temp, beta2;
    f64 dbl1 = 1.0, dbl0 = 0.0, dbl2 = 2.0;
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
    } else if (ldh < MAX(1, n)) {
        *info = -9;
    } else if (lde < MAX(1, n)) {
        *info = -11;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n == 0 || (beta == 0.0 && alpha == 1.0)) {
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

    // Start the operations.
    if (!trans) {
        // Form  R := alpha*R + beta*H*E' + beta*E*H'.
        if (!uplo) {
            beta2 = 2.0*beta;
            if (alpha == 0.0) {
                r[0] = 0.0;
            } else if (alpha != 1.0) {
                r[0] = alpha*r[0];
            }

            r[0] = r[0] + beta2*SLC_DDOT(&n, h, &ldh, e, &lde);

            for (j = 1; j < n; j++) {
                if (alpha == 0.0) {
                    for (i = 0; i <= j; i++) {
                        r[i + j*ldr] = 0.0;
                    }
                } else if (alpha != 1.0) {
                    SLC_DSCAL(&(i32){j+1}, &alpha, &r[j*ldr], &int1);
                }

                SLC_DAXPY(&j, &(f64){beta*h[j + (j-1)*ldh]}, &e[(j-1)*lde], &int1, &r[j*ldr], &int1);

                for (i = j; i < n; i++) {
                    SLC_DAXPY(&(i32){j+1}, &(f64){beta*e[j + i*lde]}, &h[i*ldh], &int1, &r[j*ldr], &int1);
                    SLC_DAXPY(&(i32){j+1}, &(f64){beta*h[j + i*ldh]}, &e[i*lde], &int1, &r[j*ldr], &int1);
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                if (alpha == 0.0) {
                    for (i = j; i < n; i++) {
                        r[i + j*ldr] = 0.0;
                    }
                } else if (alpha != 1.0) {
                    SLC_DSCAL(&(i32){n-j}, &alpha, &r[j + j*ldr], &int1);
                }

                for (i = j; i < n-1; i++) {
                    SLC_DAXPY(&(i32){i-j+2}, &(f64){beta*e[j + i*lde]}, &h[j + i*ldh], &int1, &r[j + j*ldr], &int1);
                    SLC_DAXPY(&(i32){i-j+1}, &(f64){beta*h[j + i*ldh]}, &e[j + i*lde], &int1, &r[j + j*ldr], &int1);
                }

                SLC_DAXPY(&(i32){n-j}, &(f64){beta*e[j + (n-1)*lde]}, &h[j + (n-1)*ldh], &int1, &r[j + j*ldr], &int1);
                SLC_DAXPY(&(i32){n-j}, &(f64){beta*h[j + (n-1)*ldh]}, &e[j + (n-1)*lde], &int1, &r[j + j*ldr], &int1);
            }
        }
    } else {
        // Form  R := alpha*R + beta*H'*E + beta*E'*H.
        beta2 = 2.0*beta;

        if (!uplo) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    temp = beta*(SLC_DDOT(&(i32){i+2}, &h[i*ldh], &int1, &e[j*lde], &int1) +
                                 SLC_DDOT(&(i32){i+1}, &e[i*lde], &int1, &h[j*ldh], &int1));
                    if (alpha == 0.0) {
                        r[i + j*ldr] = temp;
                    } else {
                        r[i + j*ldr] = alpha*r[i + j*ldr] + temp;
                    }
                }

                temp = beta2*SLC_DDOT(&(i32){j+1}, &h[j*ldh], &int1, &e[j*lde], &int1);
                if (alpha == 0.0) {
                    r[j + j*ldr] = temp;
                } else {
                    r[j + j*ldr] = alpha*r[j + j*ldr] + temp;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                temp = beta2*SLC_DDOT(&(i32){j+1}, &h[j*ldh], &int1, &e[j*lde], &int1);
                if (alpha == 0.0) {
                    r[j + j*ldr] = temp;
                } else {
                    r[j + j*ldr] = alpha*r[j + j*ldr] + temp;
                }
                j1 = j + 1;

                for (i = j1; i < n; i++) {
                    temp = beta*(SLC_DDOT(&(i32){j+1}, &h[i*ldh], &int1, &e[j*lde], &int1) +
                                 SLC_DDOT(&(i32){j+2}, &e[i*lde], &int1, &h[j*ldh], &int1));
                    if (alpha == 0.0) {
                        r[i + j*ldr] = temp;
                    } else {
                        r[i + j*ldr] = alpha*r[i + j*ldr] + temp;
                    }
                }
            }
        }
    }

    return;
}
