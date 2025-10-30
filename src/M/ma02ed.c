#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02ed(
    const i32 uplo,
    const i32 n,
    f64* a,
    const i32 lda
)
{
    i32 int1 = 1;
    if (uplo == 1) {

        // Construct the upper triangle of A.
        for (i32 j = 1; j < n; j++) {
            SLC_DCOPY(&j, &a[j], &lda, &a[j*lda], &int1);
        }

    } else if (uplo == 0) {

        // Construct the lower triangle of A.
        for (i32 j = 1; j < n; j++) {
            SLC_DCOPY(&j, &a[j*lda], &int1, &a[j], &lda);
        }
    }

    return;
}
