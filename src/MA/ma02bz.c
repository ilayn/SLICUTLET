#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <complex.h>

void
ma02bz(
    const i32 side,
    const i32 m,
    const i32 n,
    c128* a,
    const i32 lda
)
{
    i32 int1 = 1, intm1 = -1;
    if (((side == 0) || (side == 2)) && (m > 1)) {
        i32 m2 = m / 2;
        i32 k = m - m2;
        for (i32 j = 0; j < n; j++) {
            SLC_ZSWAP(&m2, &a[j*lda], &intm1, &a[k + j*lda], &int1);
        }
    }
    if (((side == 1) || (side == 2)) && (n > 1)) {
        i32 n2 = n / 2;
        i32 k = n - n2;
        i32 mlda = -lda;
        for (i32 i = 0; i < m; i++) {
            SLC_ZSWAP(&n2, &a[i], &mlda, &a[i + k*lda], &lda);
        }
    }

    return;
}
