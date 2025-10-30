#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>
#include <complex.h>

void
ma02az(
    const i32 trans,
    const i32 job,
    const i32 m,
    const i32 n,
    const c128* a,
    const i32 lda,
    c128* b,
    const i32 ldb
)
{
    if (trans) {
        if (job == 0) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < MIN(j, m); i++) {
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
    } else {
        if (job == 0) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < MIN(j, m); i++) {
                    b[j + i*ldb] = conj(a[i + j*lda]);
                }
            }
        } else if (job == 1) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < m; i++) {
                    b[j + i*ldb] = conj(a[i + j*lda]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m; i++) {
                    b[j + i*ldb] = conj(a[i + j*lda]);
                }
            }
        }
    }

    return;
}
