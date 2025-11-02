#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01oc(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    const f64* h,
    const i32 ldh,
    f64* x,
    const i32 ldx,
    i32* info
)
{
    i32 i, j, l;
    f64 temp1, temp2;
    f64 dbl1 = 1.0, dbl0 = 0.0;
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
    } else if (ldx < MAX(1, n)) {
        *info = -11;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n <= 0) {
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
    // Compute R = alpha*R + beta*( op( H )*X + X*op( H )' ), exploiting
    // the structure, where op( H ) is H, if UPLO = 'U', and H', if
    // UPLO = 'L'.

    if (!trans) {
        // Form  R := alpha*R + beta*( H*X + X*H' ).
        if (!uplo) {
            for (j = 0; j < n; j++) {
                if (alpha == 0.0) {
                    for (i = 0; i <= j; i++) {
                        r[i + j*ldr] = 0.0;
                    }
                } else if (alpha != 1.0) {
                    SLC_DSCAL(&(i32){j+1}, &alpha, &r[j*ldr], &int1);
                }
                i = MAX(0, j-1);
                for (l = 0; l < n; l++) {
                    if (l <= j) {
                        temp1 = x[l + j*ldx];
                    } else {
                        temp1 = x[j + l*ldx];
                    }
                    if (temp1 != 0.0) {
                        SLC_DAXPY(&(i32){MIN(l+1, j+1)}, &(f64){beta*temp1}, &h[l*ldh], &int1, &r[j*ldr], &int1);
                    }
                    if (l >= i) {
                        temp2 = h[j + l*ldh];
                        if (temp2 != 0.0) {
                            temp2 = beta*temp2;
                            SLC_DAXPY(&j, &temp2, &x[l*ldx], &int1, &r[j*ldr], &int1);
                            if (j > 0) {
                                r[j + j*ldr] = r[j + j*ldr] + temp1*temp2;
                            }
                        }
                    }
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
                for (l = MAX(0, j-1); l < n; l++) {
                    i = MIN(l+1, n);
                    if (l >= j) {
                        temp1 = beta*x[l + j*ldx];
                    } else {
                        temp1 = beta*x[j + l*ldx];
                    }
                    temp2 = beta*h[j + l*ldh];
                    SLC_DAXPY(&(i32){i-j}, &temp1, &h[j + l*ldh], &int1, &r[j + j*ldr], &int1);
                    SLC_DAXPY(&(i32){n-l}, &temp2, &x[l + l*ldx], &int1, &r[l + j*ldr], &int1);
                    if (l > j) {
                        SLC_DAXPY(&(i32){l-j}, &temp2, &x[l + j*ldx], &ldx, &r[j + j*ldr], &int1);
                    }
                }
            }
        }
    } else {
        // Form  R := alpha*R + beta*( H'*X + X*H ).
        if (!uplo) {
            for (j = 0; j < n; j++) {
                if (alpha == 0.0) {
                    for (i = 0; i <= j; i++) {
                        r[i + j*ldr] = 0.0;
                    }
                } else if (alpha != 1.0) {
                    SLC_DSCAL(&(i32){j+1}, &alpha, &r[j*ldr], &int1);
                }
                for (i = 0; i <= j; i++) {
                    for (l = 0; l < MIN(j+1, n); l++) {
                        if (l <= j) {
                            temp1 = x[l + j*ldx];
                            if (l <= i) {
                                temp2 = x[l + i*ldx];
                            } else {
                                temp2 = x[i + l*ldx];
                            }
                        } else {
                            temp1 = x[j + l*ldx];
                            temp2 = x[i + l*ldx];
                        }
                        if (l <= MIN(i+1, n)-1) {
                            r[i + j*ldr] = r[i + j*ldr] + beta*temp1*h[l + i*ldh];
                        }
                        r[i + j*ldr] = r[i + j*ldr] + beta*temp2*h[l + j*ldh];
                    }
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
                for (i = j; i < n; i++) {
                    for (l = 0; l < MIN(i+1, n); l++) {
                        if (l >= i) {
                            temp1 = x[l + j*ldx];
                            temp2 = x[l + i*ldx];
                        } else {
                            if (l >= j) {
                                temp1 = x[l + j*ldx];
                            } else {
                                temp1 = x[j + l*ldx];
                            }
                            temp2 = x[i + l*ldx];
                        }
                        r[i + j*ldr] = r[i + j*ldr] + beta*temp1*h[l + i*ldh];
                        if (l <= MIN(j+1, n)-1) {
                            r[i + j*ldr] = r[i + j*ldr] + beta*temp2*h[l + j*ldh];
                        }
                    }
                }
            }
        }
    }

    return;
}
