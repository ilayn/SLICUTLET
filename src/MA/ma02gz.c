#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>
#include <complex.h>

void
ma02gz(
    const i32 n,
    c128* a,
    const i32 lda,
    const i32 k1,
    const i32 k2,
    const i32* ipiv,
    const i32 incx
)
{
    if ((incx == 0) || (n == 0)) {
        return;
    }

    i32 jx, jp, int1 = 1;

    // Interchange column j with column ipiv(j) for each of columns k1 through k2.
    if( incx > 0 ) {
        jx = k1;
    } else {
        jx = 1 + (1 - k2)*incx;
    }

    if (incx == 1) {
        for (i32 j = k1; j <= k2; j++) {
            jp = ipiv[j - 1];
            if (jp != j) {
                SLC_ZSWAP(&n, &a[(j - 1)*lda], &int1, &a[(jp - 1)*lda], &int1);
            }
        }
    } else if (incx > 1) {
        for (i32 j = k1; j <= k2; j++) {
            jp = ipiv[jx - 1];
            if (jp != j) {
                SLC_ZSWAP(&n, &a[(j - 1)*lda], &int1, &a[(jp - 1)*lda], &int1);
            }
            jx += incx;
        }
    } else if (incx < 0) {
        for (i32 j = k2; j >= k1; j--) {
            jp = ipiv[jx - 1];
            if (jp != j) {
                SLC_ZSWAP(&n, &a[(j - 1)*lda], &int1, &a[(jp - 1)*lda], &int1);
            }
            jx += incx;
        }
    }

    return;
}
