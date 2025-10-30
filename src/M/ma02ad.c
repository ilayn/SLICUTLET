#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02ad(
    const i32 job,
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb
)
{
    if (job == 0) {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < MIN(j + 1, m); i++) {
                b[j + i*ldb] = a[i + j*lda];
            }
        }
    } else if (job == 1) {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = j; i < m; i++) {
                b[j + i*ldb] = a[i + j*lda];
            }
        }
    } else {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                b[j + i*ldb] = a[i + j*lda];
            }
        }
    }

    return;
}
