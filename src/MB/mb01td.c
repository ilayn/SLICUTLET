#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01td(
    const i32 n,
    const f64 *a,
    const i32 lda,
    f64 *b,
    const i32 ldb,
    f64 *dwork,
    i32 *info
)
{
    const f64 dbl0 = 0.0, dbl1 = 1.0;
    const i32 int1 = 1;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (lda < MAX(n, 1)) {
        *info = -3;
    } else if (ldb < MAX(n, 1)) {
        *info = -5;
    }

    // Error return.
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n == 0) {
        return;
    } else if (n == 1) {
        b[0] *= a[0];
        return;
    }

    // Test the upper quasi-triangular structure of A and B for identity.
    for (i32 i = 0; i < n - 1; i++) {
        if (a[(i + 1) + i * lda] == dbl0) {
            if (b[(i + 1) + i * ldb] != dbl0) {
                *info = 1;
                return;
            }
        } else if (i < n - 2) {
            if (a[(i + 2) + (i + 1) * lda] != dbl0) {
                *info = 1;
                return;
            }
        }
    }

    for (i32 j = 0; j < n; j++) {
        i32 jmin = MIN(j + 1, n);
        i32 jmnm = MIN(jmin, n - 1);

        // Compute the contribution of the subdiagonal of A to the
        // j-th column of the product.

        for (i32 i = 0; i < jmnm; i++) {
            dwork[i] = a[(i + 1) + i * lda] * b[i + j * ldb];
        }

        // Multiply the upper triangle of A by the j-th column of B,
        // and add to the above result.

        SLC_DTRMV("U", "N", "N", &jmin, a, &lda, &b[j * ldb], &int1);
        SLC_DAXPY(&jmnm, &dbl1, dwork, &int1, &b[1 + j * ldb], &int1);
    }
}
