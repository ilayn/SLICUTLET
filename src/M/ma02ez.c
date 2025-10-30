#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>
#include <complex.h>

void
ma02ez(
    const i32 uplo,
    const i32 trans,
    const i32 skew,
    const i32 n,
    c128* a,
    const i32 lda
)
{

    // For efficiency reasons, the parameters are not checked for errors.
    if (uplo == 1) {

        // Construct the upper triangle of A.
        if (trans) {
            if (skew == 2) {
                for (i32 i = 1; i <= n; i++) {
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(i-1) + (j-1)*lda] = -a[(j-1) + (i-1)*lda];
                    }
                }
            } else {
                for (i32 i = 1; i <= n; i++) {
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(i-1) + (j-1)*lda] = a[(j-1) + (i-1)*lda];
                    }
                }
            }
        } else {
            if (skew == 0) {
                for (i32 i = 1; i <= n; i++) {
                    for (i32 j = 1; j <= n; j++) {
                        a[(i-1) + (j-1)*lda] = conj(a[(j-1) + (i-1)*lda]);
                    }
                }
            } else if (skew == 1) {
                for (i32 i = 1; i <= n; i++) {
                    a[(i-1) + (i-1)*lda] = CMPLX(creal(a[(i-1) + (i-1)*lda]), 0.0);
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(i-1) + (j-1)*lda] = conj(a[(j-1) + (i-1)*lda]);
                    }
                }
            } else {
                for (i32 i = 1; i <= n; i++) {
                    a[(i-1) + (i-1)*lda] = CMPLX(cimag(a[(i-1) + (i-1)*lda]), 0.0);
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(i-1) + (j-1)*lda] = -conj(a[(j-1) + (i-1)*lda]);
                    }
                }
            }
        }
    } else if (uplo == 0) {
        // Construct the lower triangle of A.
        if (trans) {
            if (skew == 2) {
                for (i32 i = 1; i <= n; i++) {
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(j-1) + (i-1)*lda] = -a[(i-1) + (j-1)*lda];
                    }
                }
            } else {
                for (i32 i = 1; i <= n; i++) {
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(j-1) + (i-1)*lda] = a[(i-1) + (j-1)*lda];
                    }
                }
            }
        } else {
            if (skew == 0) {
                for (i32 i = 1; i <= n; i++) {
                    for (i32 j = 1; j <= n; j++) {
                        a[(j-1) + (i-1)*lda] = conj(a[(i-1) + (j-1)*lda]);
                    }
                }
            } else if (skew == 1) {
                for (i32 i = 1; i <= n; i++) {
                    a[(i-1) + (i-1)*lda] = CMPLX(creal(a[(i-1) + (i-1)*lda]), 0.0);
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(j-1) + (i-1)*lda] = conj(a[(i-1) + (j-1)*lda]);
                    }
                }
            } else {
                for (i32 i = 1; i <= n; i++) {
                    a[(i-1) + (i-1)*lda] = CMPLX(cimag(a[(i-1) + (i-1)*lda]), 0.0);
                    for (i32 j = i + 1; j <= n; j++) {
                        a[(j-1) + (i-1)*lda] = -conj(a[(i-1) + (j-1)*lda]);
                    }
                }
            }
        }
    }

    return;
}
