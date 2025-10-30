#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>
#include <complex.h>

f64
ma02jz(
    const i32 ltran1, // 0: no transpose, 1: transpose
    const i32 ltran2, // 0: no transpose, 1: transpose
    const i32 n,
    const c128* q1,
    const i32 ldq1,
    const c128* q2,
    const i32 ldq2,
    f64* res,
    const i32 ldres
)
{
    const c128 cdbl1 = CMPLX(1.0, 0.0), cdbl0 = CMPLX(0.0, 0.0), cdblm1 = CMPLX(-1.0, 0.0);
    if (ltran1) {
        SLC_ZGEMM("N", "C", &n, &n, &n, &cdbl1, q1, &ldq1, q1, &ldq1, &cdbl0, res, &ldres);
    } else {
        SLC_ZGEMM("C", "N", &n, &n, &n, &cdbl1, q1, &ldq1, q1, &ldq1, &cdbl0, res, &ldres);
    }
    if (ltran2) {
        SLC_ZGEMM("N", "C", &n, &n, &n, &cdbl1, q2, &ldq2, q2, &ldq2, &cdbl1, res, &ldres);
    } else {
        SLC_ZGEMM("C", "N", &n, &n, &n, &cdbl1, q2, &ldq2, q2, &ldq2, &cdbl1, res, &ldres);
    }

    // Subtract identity
    for (i32 i = 0; i < n; i++) { res[i + i*ldres] -= cdbl1; }

    f64 temp = SLC_ZLANGE("F", &n, &n, res, &ldres, NULL);
    if (ltran1 && ltran2) {
        SLC_ZGEMM("N", "C", &n, &n, &n, &cdbl1, q2, &ldq2, q1, &ldq1, &cdbl0, res, &ldres);
        SLC_ZGEMM("N", "C", &n, &n, &n, &cdbl1, q1, &ldq1, q2, &ldq2, &cdblm1, res, &ldres);
    } else if (ltran1) {
        SLC_ZGEMM("C", "C", &n, &n, &n, &cdbl1, q2, &ldq2, q1, &ldq1, &cdbl0, res, &ldres);
        SLC_ZGEMM("N", "N", &n, &n, &n, &cdbl1, q1, &ldq1, q2, &ldq2, &cdblm1, res, &ldres);
    } else if (ltran2) {
        SLC_ZGEMM("N", "N", &n, &n, &n, &cdbl1, q2, &ldq2, q1, &ldq1, &cdbl0, res, &ldres);
        SLC_ZGEMM("C", "C", &n, &n, &n, &cdbl1, q1, &ldq1, q2, &ldq2, &cdblm1, res, &ldres);
    } else {
        SLC_ZGEMM("C", "N", &n, &n, &n, &cdbl1, q2, &ldq2, q1, &ldq1, &cdbl0, res, &ldres);
        SLC_ZGEMM("C", "N", &n, &n, &n, &cdbl1, q1, &ldq1, q2, &ldq2, &cdblm1, res, &ldres);
    }
    temp = hypot(temp, SLC_ZLANGE("F", &n, &n, res, &ldres, NULL));

    return sqrt(2.0) * temp;
}

