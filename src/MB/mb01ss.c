#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01ss(
    const i32 jobs,  // 0: "D", 1: "I"
    const i32 uplo,  // 0: upper, 1: lower
    const i32 n,
    f64* a,
    const i32 lda,
    const f64* d
)
{
    if (n == 0) {
        return;
    }

    if (jobs == 0) {
        // Row and column scaling with D.
        if (uplo == 0) {
            // Upper triangular part
            for (i32 j = 0; j < n; j++) {
                f64 dj = d[j];
                for (i32 i = 0; i <= j; i++) {
                    a[i + j*lda] *= dj * d[i];
                }
            }
        } else {
            // Lower triangular part
            for (i32 j = 0; j < n; j++) {
                f64 dj = d[j];
                for (i32 i = j; i < n; i++) {
                    a[i + j*lda] *= dj * d[i];
                }
            }
        }
    } else {
        // Row and column scaling with D^{-1}.
        if (uplo == 0) {
            // Upper triangular part
            for (i32 j = 0; j < n; j++) {
                f64 dj = 1.0 / d[j];
                for (i32 i = 0; i <= j; i++) {
                    a[i + j*lda] *= dj / d[i];
                }
            }
        } else {
            // Lower triangular part
            for (i32 j = 0; j < n; j++) {
                f64 dj = d[j];
                for (i32 i = j; i < n; i++) {
                    a[i + j*lda] *= dj / d[i];
                }
            }
        }
    }

    return;
}
