#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>
#include <complex.h>

void
ma02nz(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: transpose, 1: conjugate transpose
    const i32 skew,  // 0: hermitian, 1: skew-Hermitian
    const i32 n,
    const i32 k,     // 0-indexed
    const i32 l,     // 0-indexed
    c128* a,
    const i32 lda
)
{
    if ((n == 0) || (k == l)) { return; }

    c128 t = a[k + k*lda];
    a[k + k*lda] = a[l + l*lda];
    a[l + l*lda] = t;

    if (uplo == 1) {
        // Permute the lower triangle of A.
        SLC_ZSWAP(&k, &a[k], &lda, &a[l], &lda);

        if (trans == 0) {
            if (skew == 0) {
                SLC_ZSWAP(&(i32){l - k - 1}, &a[(k+1) + k*lda], &(i32){1}, &a[l + (k+1)*lda], &lda);
            } else {
                a[l + k*lda] = -a[l + k*lda];
                for (i32 i = k + 1; i <= l - 1; i++) {
                    t            = -a[l + i*lda];
                    a[l + i*lda] = -a[i + k*lda];
                    a[i + k*lda] =  t;
                }
            }
        } else {
            if (skew == 0) {
                a[l + k*lda] = conj(a[l + k*lda]);
                for (i32 i = k + 1; i <= l - 1; i++) {
                    t            = conj(a[l + i*lda]);
                    a[l + i*lda] = conj(a[i + k*lda]);
                    a[i + k*lda] =  t;
                }
            } else {
                a[l + k*lda] = CMPLX(-creal(a[l + k*lda]), cimag(a[l + k*lda]));
                for (i32 i = k + 1; i <= l - 1; i++) {
                    t            = CMPLX(-creal(a[l + i*lda]), cimag(a[l + i*lda]));
                    a[l + i*lda] = CMPLX(-creal(a[i + k*lda]), cimag(a[i + k*lda]));
                    a[i + k*lda] =  t;
                }
            }
        }
        SLC_ZSWAP(&(i32){n - l - 1}, &a[(l+1) + k*lda], &(i32){1}, &a[(l+1) + l*lda], &(i32){1});

    } else {
        // Permute the upper triangle of A.
        SLC_ZSWAP(&k, &a[k*lda], &(i32){1}, &a[l*lda], &(i32){1});

        if (trans == 0) {
            if (skew == 0) {
                SLC_ZSWAP(&(i32){l - k - 1}, &a[k + (k+1)*lda], &lda, &a[(k+1) + l*lda], &(i32){1});
            } else {
                a[k + l*lda] = -a[k + l*lda];
                for (i32 j = k + 1; j <= l - 1; j++) {
                    t            = -a[j + l*lda];
                    a[j + l*lda] = -a[k + j*lda];
                    a[k + j*lda] =  t;
                }
            }
        } else {
            if (skew == 0) {
                a[k + l*lda] = conj(a[k + l*lda]);
                for (i32 i = k + 1; i <= l - 1; i++) {
                    t            = conj(a[i + l*lda]);
                    a[i + l*lda] = conj(a[k + i*lda]);
                    a[k + i*lda] =  t;
                }
            } else {
                a[k + l*lda] = CMPLX(-creal(a[k + l*lda]), cimag(a[k + l*lda]));
                for (i32 i = k + 1; i <= l - 1; i++) {
                    t            = CMPLX(-creal(a[i + l*lda]), cimag(a[i + l*lda]));
                    a[i + l*lda] = CMPLX(-creal(a[k + i*lda]), cimag(a[k + i*lda]));
                    a[k + i*lda] =  t;
                }
            }
        }
        SLC_ZSWAP(&(i32){n - l - 1}, &a[k + (l+1)*lda], &lda, &a[l + (l+1)*lda], &lda);
    }

    return;
}
