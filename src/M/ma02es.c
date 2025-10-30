#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02es(
    const i32 uplo,
    const i32 n,
    f64* a,
    const i32 lda
)
{

    // For efficiency reasons, the parameters are not checked for errors.
    if (uplo == 1) {

        // Construct the upper triangle of A.
        for (i32 i = 1; i <= n; i++) {
            a[(i-1) + (i-1)*lda] = 0.0;
            for (i32 j = 2; j <= n; j++) {
                a[(i-1) + (j-1)*lda] = -a[(j-1) + (i-1)*lda];
            }
        }

    } else if (uplo == 0) {

        // Construct the lower triangle of A.
        for (i32 i = 1; i <= n; i++) {
            a[(i-1) + (i-1)*lda] = 0.0;
            for (i32 j = 2; j <= n; j++) {
                a[(j-1) + (i-1)*lda] = -a[(i-1) + (j-1)*lda];
            }
        }
    }

    return;
}
