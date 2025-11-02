#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01ry(
    const i32 side,  // 0: left, 1: right
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 m,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    f64* h,
    const i32 ldh,
    const f64* b,
    const i32 ldb,
    f64* dwork,
    i32* info
)
{
    i32 i, j;
    f64 dbl1 = 1.0, dbl0 = 0.0;
    i32 int1 = 1, int0 = 0;

    *info = 0;

    if (side != 0 && side != 1) {
        *info = -1;
    } else if (uplo != 0 && uplo != 1) {
        *info = -2;
    } else if (trans != 0 && trans != 1) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (ldr < MAX(1, m)) {
        *info = -8;
    } else if (ldh < MAX(1, m)) {
        *info = -10;
    } else if (ldb < MAX(1, m)) {
        *info = -12;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (m == 0) {
        return;
    }

    if (beta == 0.0) {
        if (alpha == 0.0) {
            // Special case when both alpha = 0 and beta = 0.
            SLC_DLASET((uplo ? "L" : "U"), &m, &m, &dbl0, &dbl0, r, &ldr);
        } else {
            // Special case beta = 0.
            if (alpha != 1.0) {
                SLC_DLASCL((uplo ? "L" : "U"), &int0, &int0, &dbl1, &alpha, &m, &m, r, &ldr, info);
            }
        }
        return;
    }

    // General case: beta <> 0.
    // Compute the required triangle of (1) or (2) using BLAS 2 operations.
    if (side == 0) {
        // To avoid repeated references to the subdiagonal elements of H,
        // these are swapped with the corresponding elements of H in the
        // first column, and are finally restored.
        if (m > 2) {
            SLC_DSWAP(&(i32){m-2}, &h[2+ldh], &(i32){ldh+1}, &h[2], &int1);
        }

        if (!uplo) {
            if (trans) {
                for (j = 0; j < m; j++) {
                    // Multiply the transposed upper triangle of the leading
                    // j-by-j submatrix of H by the leading part of the j-th
                    // column of B.
                    SLC_DCOPY(&(i32){j+1}, &b[j*ldb], &int1, dwork, &int1);
                    SLC_DTRMV("U", "T", "N", &(i32){j+1}, h, &ldh, dwork, &int1);

                    // Add the contribution of the subdiagonal of H to
                    // the j-th column of the product.
                    for (i = 0; i < MIN(j+1, m-1); i++) {
                        r[i + j*ldr] = alpha*r[i + j*ldr] + beta*(dwork[i] + h[i+1]*b[i+1 + j*ldb]);
                    }
                }
                r[m-1 + (m-1)*ldr] = alpha*r[m-1 + (m-1)*ldr] + beta*dwork[m-1];

            } else {
                for (j = 0; j < m; j++) {
                    // Multiply the upper triangle of the leading j-by-j
                    // submatrix of H by the leading part of the j-th column of B.
                    SLC_DCOPY(&(i32){j+1}, &b[j*ldb], &int1, dwork, &int1);
                    SLC_DTRMV("U", "N", "N", &(i32){j+1}, h, &ldh, dwork, &int1);

                    if (j < m-1) {
                        // Multiply the remaining right part of the leading
                        // j-by-M submatrix of H by the trailing part of the
                        // j-th column of B.
                        SLC_DGEMV("N", &(i32){j+1}, &(i32){m-j-1}, &beta, &h[(j+1)*ldh], &ldh,
                                  &b[j+1 + j*ldb], &int1, &alpha, &r[j*ldr], &int1);
                    } else {
                        SLC_DSCAL(&m, &alpha, &r[(m-1)*ldr], &int1);
                    }

                    // Add the contribution of the subdiagonal of H to
                    // the j-th column of the product.
                    r[j*ldr] = r[j*ldr] + beta*dwork[0];

                    for (i = 1; i <= j; i++) {
                        r[i + j*ldr] = r[i + j*ldr] + beta*(dwork[i] + h[i]*b[i-1 + j*ldb]);
                    }
                }
            }

        } else {
            if (trans) {
                for (j = m-1; j >= 0; j--) {
                    // Multiply the transposed upper triangle of the trailing
                    // (M-j+1)-by-(M-j+1) submatrix of H by the trailing part
                    // of the j-th column of B.
                    SLC_DCOPY(&(i32){m-j}, &b[j + j*ldb], &int1, &dwork[j], &int1);
                    SLC_DTRMV("U", "T", "N", &(i32){m-j}, &h[j + j*ldh], &ldh, &dwork[j], &int1);

                    if (j > 0) {
                        // Multiply the remaining left part of the trailing
                        // (M-j+1)-by-(j-1) submatrix of H' by the leading
                        // part of the j-th column of B.
                        SLC_DGEMV("T", &j, &(i32){m-j}, &beta, &h[j*ldh], &ldh,
                                  &b[j*ldb], &int1, &alpha, &r[j + j*ldr], &int1);
                    } else {
                        SLC_DSCAL(&m, &alpha, r, &int1);
                    }

                    // Add the contribution of the subdiagonal of H to
                    // the j-th column of the product.
                    for (i = j; i < m-1; i++) {
                        r[i + j*ldr] = r[i + j*ldr] + beta*(dwork[i] + h[i+1]*b[i+1 + j*ldb]);
                    }
                    r[m-1 + j*ldr] = r[m-1 + j*ldr] + beta*dwork[m-1];
                }

            } else {
                for (j = m-1; j >= 0; j--) {
                    // Multiply the upper triangle of the trailing
                    // (M-j+1)-by-(M-j+1) submatrix of H by the trailing
                    // part of the j-th column of B.
                    SLC_DCOPY(&(i32){m-j}, &b[j + j*ldb], &int1, &dwork[j], &int1);
                    SLC_DTRMV("U", "N", "N", &(i32){m-j}, &h[j + j*ldh], &ldh, &dwork[j], &int1);

                    // Add the contribution of the subdiagonal of H to
                    // the j-th column of the product.
                    for (i = MAX(j, 1); i < m; i++) {
                        r[i + j*ldr] = alpha*r[i + j*ldr] + beta*(dwork[i] + h[i]*b[i-1 + j*ldb]);
                    }
                }
                r[0] = alpha*r[0] + beta*dwork[0];
            }
        }

        if (m > 2) {
            SLC_DSWAP(&(i32){m-2}, &h[2+ldh], &(i32){ldh+1}, &h[2], &int1);
        }

    } else {
        // Row-wise calculations are used for H, if SIDE = 'R' and TRANS = 'T'.
        if (!uplo) {
            if (trans) {
                r[0] = alpha*r[0] + beta*SLC_DDOT(&m, b, &ldb, h, &ldh);

                for (j = 1; j < m; j++) {
                    SLC_DGEMV("N", &(i32){j+1}, &(i32){m-j+1}, &beta,
                              &b[(j-1)*ldb], &ldb, &h[j + (j-1)*ldh], &ldh,
                              &alpha, &r[j*ldr], &int1);
                }

            } else {
                for (j = 0; j < m-1; j++) {
                    SLC_DGEMV("N", &(i32){j+1}, &(i32){j+2}, &beta, b, &ldb,
                              &h[j*ldh], &int1, &alpha, &r[j*ldr], &int1);
                }
                SLC_DGEMV("N", &m, &m, &beta, b, &ldb,
                          &h[(m-1)*ldh], &int1, &alpha, &r[(m-1)*ldr], &int1);
            }

        } else {
            if (trans) {
                SLC_DGEMV("N", &m, &m, &beta, b, &ldb, h, &ldh, &alpha, r, &int1);

                for (j = 1; j < m; j++) {
                    SLC_DGEMV("N", &(i32){m-j}, &(i32){m-j+1}, &beta,
                              &b[j + (j-1)*ldb], &ldb, &h[j + (j-1)*ldh], &ldh,
                              &alpha, &r[j + j*ldr], &int1);
                }

            } else {
                for (j = 0; j < m-1; j++) {
                    SLC_DGEMV("N", &(i32){m-j}, &(i32){j+2}, &beta,
                              &b[j], &ldb, &h[j*ldh], &int1, &alpha, &r[j + j*ldr], &int1);
                }
                r[m-1 + (m-1)*ldr] = alpha*r[m-1 + (m-1)*ldr] +
                                      beta*SLC_DDOT(&m, &b[m-1], &ldb, &h[(m-1)*ldh], &int1);
            }
        }
    }

    return;
}
