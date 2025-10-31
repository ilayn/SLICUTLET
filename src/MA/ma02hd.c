#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

i32
ma02hd(
    const i32 job,
    const i32 m,
    const i32 n,
    const f64 diag,
    const f64* a,
    const i32 lda
)
{
    //    Do not check parameters, for efficiency.
    //    Quick return if possible.
    if (MIN(m, n) == 0) {
        return 0;
    }

    if (job == 0) {
        for (i32 j = 1; j <= n; ++j) {
            for (i32 i = 1; i <= MIN(j-1, m); i++) {
                if (a[(i-1) + (j-1)*lda] != 0.0) { return 0; }
            }
            if (j <= m) {
                if (a[(j-1) + (j-1)*lda] != diag) { return 0; }
            }
        }

    } else if (job == 1) {
        for (i32 j = 1; j <= MIN(m, n); j++) {
            if (a[(j-1) + (j-1)*lda] != diag) { return 0; }
            if (j < m) {
                for (i32 i = j+1; i <= m; i++) {
                    if (a[(i-1) + (j-1)*lda] != 0.0) { return 0; }
                }
            }
        }
    } else {
        for (i32 j = 1; j <= n; j++) {
            for (i32 i = 1; i <= MIN(j-1, m); i++) {
                if (a[(i-1) + (j-1)*lda] != 0.0) { return 0; }
            }
            if (j <= m) {
                if (a[(j-1) + (j-1)*lda] != diag) { return 0; }
            }
            if (j < m) {
                for (i32 i = j+1; i <= m; i++) {
                    if (a[(i-1) + (j-1)*lda] != 0.0) { return 0; }
                }
            }
        }
    }

    return 1;
}
