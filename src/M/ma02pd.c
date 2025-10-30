#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02pd(
    const i32 m,
    const i32 n,
    const f64* a,
    const i32 lda,
    i32* nzr,
    i32* nzc
)
{
    i32 i = 0;
    *nzc = 0;
    *nzr = 0;

    if (MIN(m, n) > 0) {

        // Scan columns 1 .. n
        i = 0;
label10:
        i++;
        if (i <= n) {
            for (i32 j = 1; j <= m; j++) {
                if (a[(j-1) + (i-1)*lda] != 0.0) { goto label10; }
            }
            (*nzc)++;
            goto label10;
        }

        // Scan rows 1 .. m
        i = 0;
label30:
        i++;
        if (i <= m) {
            for (i32 j = 1; j <= n; j++) {
                if (a[(i-1) + (j-1)*lda] != 0.0) { goto label30; }
            }
            (*nzr)++;
            goto label30;
        }
    }

    return;
}
