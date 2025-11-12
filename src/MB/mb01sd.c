#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01sd(
    const i32 jobs, // 0: row, 1: column, 2: both
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    const f64* r,
    const f64* c
)
{
    if (m == 0 || n == 0) {
        return;
    }

    if (jobs == 1) {
        // Column scaling
        for (i32 j = 0; j < n; j++) {
            f64 cj = c[j];
            for (i32 i = 0; i < m; i++) {
                a[i + j*lda] *= cj;
            }
        }
    } else if (jobs == 0) {
        // Row scaling
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                a[i + j*lda] *= r[i];
            }
        }
    } else if (jobs == 2) {
        // Both row and column scaling
        for (i32 j = 0; j < n; j++) {
            f64 cj = c[j];
            for (i32 i = 0; i < m; i++) {
                a[i + j*lda] *= cj * r[i];
            }
        }
    }
}
