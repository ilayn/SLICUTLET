#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02cd(
    const i32 n,
    const i32 kl,
    const i32 ku,
    f64* a,
    const i32 lda
)
{
    if (n < 2) { return; }

    i32 lda1 = lda + 1;
    i32 ldam1 = -lda1;

    // Perttranspose the kl subdiagonals
    for (i32 i = 1; i <= MIN(kl, n - 2); i++) {
        i32 i1 = (n - i) / 2;
        if (i1 > 0) {
            SLC_DSWAP(&i1, &a[i], &lda1, &a[(n - i1) + (n - i1 - i)*lda], &ldam1);
        }
    }

    // Pertranspose the KU superdiagonals.
    for (i32 i = 1; i <= MIN(ku, n - 2); i++) {
        i32 i1 = (n - i) / 2;
        if (i1 > 0) {
            SLC_DSWAP(&i1, &a[i*lda], &lda1, &a[(n - i1 - i) + (n - i1)*lda], &ldam1);
        }
    }

    // Pertranspose the diagonal.
    i32 i1 = n / 2;
    if (i1 > 0) {
        SLC_DSWAP(&i1, &a[0], &lda1, &a[(n - i1) + (n - i1)*lda], &ldam1);
    }

    return;
}
