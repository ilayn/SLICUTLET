#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01rw(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    const f64* z,
    const i32 ldz,
    f64* dwork,
    i32* info
)
{
    i32 i, j;
    f64 dbl1 = 1.0, dbl0 = 0.0;
    i32 int1 = 1;

    *info = 0;

    if (uplo != 0 && uplo != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < MAX(1, MAX(m, n))) {
        *info = -6;
    } else if ((!trans && ldz < MAX(1, m)) ||
               (trans && ldz < MAX(1, n))) {
        *info = -8;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n == 0 || m == 0) {
        return;
    }

    if (!trans) {
        // Compute Z*A*Z'.
        if (!uplo) {
            // Compute Z*A in A (M-by-N).
            for (j = 0; j < n; j++) {
                SLC_DCOPY(&j, &a[j*lda], &int1, dwork, &int1);
                SLC_DCOPY(&(i32){n-j}, &a[j + j*lda], &lda, &dwork[j], &int1);
                SLC_DGEMV("N", &m, &n, &dbl1, z, &ldz, dwork, &int1, &dbl0, &a[j*lda], &int1);
            }

            // Compute A*Z' in the upper triangular part of A.
            for (i = 0; i < m; i++) {
                SLC_DCOPY(&n, &a[i], &lda, dwork, &int1);
                SLC_DGEMV("N", &(i32){m-i}, &n, &dbl1, &z[i], &ldz, dwork, &int1, &dbl0, &a[i + i*lda], &lda);
            }

        } else {
            // Compute A*Z' in A (N-by-M).
            for (i = 0; i < n; i++) {
                SLC_DCOPY(&i, &a[i], &lda, dwork, &int1);
                SLC_DCOPY(&(i32){n-i}, &a[i + i*lda], &int1, &dwork[i], &int1);
                SLC_DGEMV("N", &m, &n, &dbl1, z, &ldz, dwork, &int1, &dbl0, &a[i], &lda);
            }

            // Compute Z*A in the lower triangular part of A.
            for (j = 0; j < m; j++) {
                SLC_DCOPY(&n, &a[j*lda], &int1, dwork, &int1);
                SLC_DGEMV("N", &(i32){m-j}, &n, &dbl1, &z[j], &ldz, dwork, &int1, &dbl0, &a[j + j*lda], &int1);
            }
        }
    } else {
        // Compute Z'*A*Z.
        if (!uplo) {
            // Compute Z'*A in A (M-by-N).
            for (j = 0; j < n; j++) {
                SLC_DCOPY(&j, &a[j*lda], &int1, dwork, &int1);
                SLC_DCOPY(&(i32){n-j}, &a[j + j*lda], &lda, &dwork[j], &int1);
                SLC_DGEMV("T", &n, &m, &dbl1, z, &ldz, dwork, &int1, &dbl0, &a[j*lda], &int1);
            }

            // Compute A*Z in the upper triangular part of A.
            for (i = 0; i < m; i++) {
                SLC_DCOPY(&n, &a[i], &lda, dwork, &int1);
                SLC_DGEMV("T", &n, &(i32){m-i}, &dbl1, &z[i*ldz], &ldz, dwork, &int1, &dbl0, &a[i + i*lda], &lda);
            }

        } else {
            // Compute A*Z in A (N-by-M).
            for (i = 0; i < n; i++) {
                SLC_DCOPY(&i, &a[i], &lda, dwork, &int1);
                SLC_DCOPY(&(i32){n-i}, &a[i + i*lda], &int1, &dwork[i], &int1);
                SLC_DGEMV("T", &n, &m, &dbl1, z, &ldz, dwork, &int1, &dbl0, &a[i], &lda);
            }

            // Compute Z'*A in the lower triangular part of A.
            for (j = 0; j < m; j++) {
                SLC_DCOPY(&n, &a[j*lda], &int1, dwork, &int1);
                SLC_DGEMV("T", &n, &(i32){m-j}, &dbl1, &z[j*ldz], &ldz, dwork, &int1, &dbl0, &a[j + j*lda], &int1);
            }
        }
    }

    return;
}
