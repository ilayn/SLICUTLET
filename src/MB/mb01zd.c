#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01zd(
    const i32 side,   // 0: left, 1: right
    const i32 uplo,   // 0: upper, 1: lower
    const i32 transt, // 0: no transpose, 1: transpose
    const i32 diag,   // 0: non-unit, 1: unit
    const i32 m,
    const i32 n,
    const i32 l,
    const f64 alpha,
    const f64* t,
    const i32 ldt,
    f64* h,
    const i32 ldh,
    i32* info
)
{
    i32 i, i1, i2, j, k, m2, nrowt;
    f64 temp;
    f64 dbl1 = 1.0, dbl0 = 0.0;
    i32 int1 = 1;

    *info = 0;

    if (side == 0) {
        nrowt = m;
    } else {
        nrowt = n;
    }

    if (uplo == 0) {
        m2 = m;
    } else {
        m2 = n;
    }

    if (side != 0 && side != 1) {
        *info = -1;
    } else if (uplo != 0 && uplo != 1) {
        *info = -2;
    } else if (transt != 0 && transt != 1) {
        *info = -3;
    } else if (diag != 0 && diag != 1) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (l < 0 || l > MAX(0, m2-1)) {
        *info = -7;
    } else if (ldt < MAX(1, nrowt)) {
        *info = -10;
    } else if (ldh < MAX(1, m)) {
        *info = -12;
    }
    if (*info != 0) {
        return;
    }

    // Quick return, if possible.
    if (MIN(m, n) == 0) {
        return;
    }

    // Also, when alpha = 0.
    if (alpha == 0.0) {
        for (j = 0; j < n; j++) {
            if (uplo == 0) {
                i1 = 0;
                i2 = MIN(j+l, m-1);
            } else {
                i1 = MAX(0, j-l);
                i2 = m-1;
            }

            for (i = i1; i <= i2; i++) {
                h[i + j*ldh] = 0.0;
            }
        }
        return;
    }

    // Start the operations.
    if (side == 0) {
        if (!transt) {
            // Form  H := alpha*T*H.
            if (uplo == 0) {
                for (j = 0; j < n; j++) {
                    for (k = 0; k < MIN(j+l+1, m); k++) {
                        if (h[k + j*ldh] != 0.0) {
                            temp = alpha * h[k + j*ldh];
                            SLC_DAXPY(&k, &temp, &t[k*ldt], &int1, &h[j*ldh], &int1);
                            if (!diag) {
                                temp = temp * t[k + k*ldt];
                            }
                            h[k + j*ldh] = temp;
                        }
                    }
                }

            } else {
                for (j = 0; j < n; j++) {
                    for (k = m-1; k >= MAX(0, j-l); k--) {
                        if (h[k + j*ldh] != 0.0) {
                            temp = alpha * h[k + j*ldh];
                            h[k + j*ldh] = temp;
                            if (!diag) {
                                h[k + j*ldh] = h[k + j*ldh] * t[k + k*ldt];
                            }
                            SLC_DAXPY(&(i32){m-k-1}, &temp, &t[k+1 + k*ldt], &int1, &h[k+1 + j*ldh], &int1);
                        }
                    }
                }
            }

        } else {
            // Form  H := alpha*T'*H.
            if (uplo == 0) {
                for (j = 0; j < n; j++) {
                    i1 = j + l;

                    for (i = m-1; i >= 0; i--) {
                        if (i > i1) {
                            temp = SLC_DDOT(&(i32){i1+1}, &t[i*ldt], &int1, &h[j*ldh], &int1);
                        } else {
                            temp = h[i + j*ldh];
                            if (!diag) {
                                temp = temp * t[i + i*ldt];
                            }
                            temp = temp + SLC_DDOT(&i, &t[i*ldt], &int1, &h[j*ldh], &int1);
                        }
                        h[i + j*ldh] = alpha * temp;
                    }
                }

            } else {
                for (j = 0; j < MIN(m+l, n); j++) {
                    i1 = j - l;

                    for (i = 0; i < m; i++) {
                        if (i < i1) {
                            temp = SLC_DDOT(&(i32){m-i1}, &t[i1 + i*ldt], &int1, &h[i1 + j*ldh], &int1);
                        } else {
                            temp = h[i + j*ldh];
                            if (!diag) {
                                temp = temp * t[i + i*ldt];
                            }
                            temp = temp + SLC_DDOT(&(i32){m-i-1}, &t[i+1 + i*ldt], &int1, &h[i+1 + j*ldh], &int1);
                        }
                        h[i + j*ldh] = alpha * temp;
                    }
                }
            }
        }

    } else {
        if (!transt) {
            // Form  H := alpha*H*T.
            if (uplo == 0) {
                for (j = n-1; j >= 0; j--) {
                    i2 = MIN(j+l, m-1);
                    temp = alpha;
                    if (!diag) {
                        temp = temp * t[j + j*ldt];
                    }
                    SLC_DSCAL(&(i32){i2+1}, &temp, &h[j*ldh], &int1);

                    for (k = 0; k < j; k++) {
                        SLC_DAXPY(&(i32){i2+1}, &(f64){alpha*t[k + j*ldt]}, &h[k*ldh], &int1, &h[j*ldh], &int1);
                    }
                }

            } else {
                for (j = 0; j < n; j++) {
                    i1 = MAX(0, j-l);
                    temp = alpha;
                    if (!diag) {
                        temp = temp * t[j + j*ldt];
                    }
                    SLC_DSCAL(&(i32){m-i1}, &temp, &h[i1 + j*ldh], &int1);

                    for (k = j+1; k < n; k++) {
                        SLC_DAXPY(&(i32){m-i1}, &(f64){alpha*t[k + j*ldt]}, &h[i1 + k*ldh], &int1, &h[i1 + j*ldh], &int1);
                    }
                }
            }

        } else {
            // Form  H := alpha*H*T'.
            if (uplo == 0) {
                m2 = MIN(n+l, m);

                for (k = 0; k < n; k++) {
                    i1 = MIN(k+l, m-1);
                    i2 = MIN(k+l, m2-1);

                    for (j = 0; j < k; j++) {
                        if (t[j + k*ldt] != 0.0) {
                            temp = alpha * t[j + k*ldt];
                            SLC_DAXPY(&(i32){i1+1}, &temp, &h[k*ldh], &int1, &h[j*ldh], &int1);

                            for (i = i1+1; i <= i2; i++) {
                                h[i + j*ldh] = temp * h[i + k*ldh];
                            }
                        }
                    }

                    temp = alpha;
                    if (!diag) {
                        temp = temp * t[k + k*ldt];
                    }
                    if (temp != 1.0) {
                        SLC_DSCAL(&(i32){i2+1}, &temp, &h[k*ldh], &int1);
                    }
                }

            } else {
                for (k = n-1; k >= 0; k--) {
                    i1 = MAX(0, k-l);
                    i2 = MAX(0, k-l+1);
                    m2 = MIN(m-1, i2-1);

                    for (j = k+1; j < n; j++) {
                        if (t[j + k*ldt] != 0.0) {
                            temp = alpha * t[j + k*ldt];
                            SLC_DAXPY(&(i32){m-i2+1}, &temp, &h[i2 + k*ldh], &int1, &h[i2 + j*ldh], &int1);

                            for (i = i1; i <= m2; i++) {
                                h[i + j*ldh] = temp * h[i + k*ldh];
                            }
                        }
                    }

                    temp = alpha;
                    if (!diag) {
                        temp = temp * t[k + k*ldt];
                    }
                    if (temp != 1.0) {
                        SLC_DSCAL(&(i32){m-i1}, &temp, &h[i1 + k*ldh], &int1);
                    }
                }
            }
        }
    }

    return;
}
