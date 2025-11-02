#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01xy(
    const i32 uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    i32 *info
)
{
    const i32 int1 = 1;
    const f64 dbl1 = 1.0;
    *info = 0;
    if((uplo != 0) && (uplo != 1)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if(lda < MAX(1, n)) {
        *info = -4;
    }

    if(*info != 0) {
        return;
    }

    // Quick return, if possible.
    if (n == 0) {
        return;
    }

    if (uplo == 0) {
        // Compute the product U' * U.
        a[n-1 + (n-1)*lda] = SLC_DDOT(&n, &a[(n-1)*lda], &int1, &a[(n-1)*lda], &int1);
        for (i32 i = n-2; i > 0; i--) {
            f64 aii = a[i + i*lda];
            a[i + i*lda] = SLC_DDOT(&(i32){i+1}, &a[i*lda], &int1, &a[i*lda], &int1);
            SLC_DGEMV("T", &i, &(i32){n - i - 1}, &dbl1, &a[(i+1)*lda], &lda, &a[i*lda], &int1, &aii, &a[i + (i+1)*lda], &lda);
        }

        if (n > 1) {
            f64 a11 = a[0];
            SLC_DSCAL(&n, &a11, &a[0], &lda);
        }
    } else {
        // Compute the product L * L'.
        a[n-1 + (n-1)*lda] = SLC_DDOT(&n, &a[n-1], &lda, &a[n-1], &lda);
        for (i32 i = n-2; i > 0; i--) {
            f64 aii = a[i + i*lda];
            a[i + i*lda] = SLC_DDOT(&(i32){i+1}, &a[i], &lda, &a[i], &lda);
            SLC_DGEMV("N", &(i32){n - i - 1}, &i, &dbl1, &a[i + 1], &lda, &a[i], &lda, &aii, &a[(i+1) + i*lda], &int1);
        }

        if (n > 1) {
            f64 a11 = a[0];
            SLC_DSCAL(&n, &a11, &a[0], &int1);
        }
    }

    return;
}
