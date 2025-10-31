#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <complex.h>

i32
ma02oz(
    const i32 skew,  // 0: hamiltonian, 1: skew-Hamiltonian
    const i32 m,
    const c128* a,
    const i32 lda,
    const c128* de,
    const i32 ldde
)
{
    i32 nz = 0, i = 0;
    c128 czero = CMPLX(0.0, 0.0);

    if (m > 0) {

        // Scan columns 1 .. m
        i = 0;
label10:
        i++;
        if (i <= m) {
            for (i32 j = 1; j <= m; j++) {
                if (a[(j-1) + (i-1)*lda] != czero) { goto label10; }
            }
            for (i32 j = 1; j <= i - 1; j++) {
                if (de[(i-1) + (j-1)*ldde] != czero) { goto label10; }
            }
            if (skew == 1) {
                if (cimag(de[(i-1) + (i-1)*ldde]) != 0.0) { goto label10; }
            } else {
                if (creal(de[(i-1) + (i-1)*ldde]) != 0.0) { goto label10; }
            }
            for (i32 j = i + 1; j <= m; j++) {
                if (de[(j-1) + (i-1)*ldde] != czero) { goto label10; }
            }
            nz++;
            goto label10;
        }

        // Scan columns m+1 .. 2*m
        i = 0;
label50:
        i++;
        if (i <= m) {
            for (i32 j = 1; j <= m; j++) {
                if (a[(i-1) + (j-1)*lda] != czero) { goto label50; }
            }
            for (i32 j = 1; j <= i - 1; j++) {
                if (de[(j-1) + i*ldde] != czero) { goto label50; }
            }
            if (skew == 1) {
                if (cimag(de[(i-1) + i*ldde]) != 0.0) { goto label50; }
            } else {
                if (creal(de[(i-1) + i*ldde]) != 0.0) { goto label50; }
            }
            for (i32 j = i + 1; j <= m; j++) {
                if (de[(i-1) + j*ldde] != czero) { goto label50; }
            }
            nz++;
            goto label50;
        }
    }

    return nz;
}
