#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02dd(
    const i32 job,
    const i32 uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* ap
)
{
    i32 ij = 1, int1 = 1;

    if (job == 0) {
        if (uplo) {

            // Pack the lower triangle of A.

            for (i32 j = 1; j<= n; j++) {
                i32 nj = n - j + 1;
                SLC_DCOPY(&nj, &a[(j - 1) + (j - 1)*lda], &int1, &ap[ij - 1], &int1);
                ij += n - j + 1;
            }

        } else {

            // Pack the upper triangle of A.

            for (i32 j = 1; j<= n; j++) {
                SLC_DCOPY(&j, &a[(j - 1)*lda], &int1, &ap[ij - 1], &int1);
                ij += j;
            }

        }
    } else {
        if (uplo) {

            // Unpack the lower triangle of A.

            for (i32 j = 1; j<= n; j++) {
                i32 nj = n - j + 1;
                SLC_DCOPY(&nj, &ap[ij - 1], &int1, &a[(j - 1) + (j - 1)*lda], &int1);
                ij += n - j + 1;
            }

        } else {

            // Unpack the upper triangle of A.

            for (i32 j = 1; j<= n; j++) {
                SLC_DCOPY(&j, &ap[ij - 1], &int1, &a[(j - 1)*lda], &int1);
                ij += j;
            }
        }
    }

    return;
}
