#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

f64
ma02jd(
    const i32 ltran1, // 0: no transpose, 1: transpose
    const i32 ltran2, // 0: no transpose, 1: transpose
    const i32 n,
    const f64* q1,
    const i32 ldq1,
    const f64* q2,
    const i32 ldq2,
    f64* res,
    const i32 ldres
)
{
    f64 dbl1 = 1.0, dbl0 = 0.0, dblm1 = -1.0;

    if (ltran1) {
        SLC_DGEMM("N", "T", &n, &n, &n, &dbl1, q1, &ldq1, q1, &ldq1, &dbl0, res, &ldres);
    } else {
        SLC_DGEMM("T", "N", &n, &n, &n, &dbl1, q1, &ldq1, q1, &ldq1, &dbl0, res, &ldres);
    }

    if (ltran2) {
        SLC_DGEMM("N", "T", &n, &n, &n, &dbl1, q2, &ldq2, q2, &ldq2, &dbl1, res, &ldres);
    } else {
        SLC_DGEMM("T", "N", &n, &n, &n, &dbl1, q2, &ldq2, q2, &ldq2, &dbl1, res, &ldres);
    }

    // Subtract identity
    for (i32 i = 0; i < n; i++) { res[i + i*ldres] -= 1.0; }

    f64 temp = SLC_DLANGE("F", &n, &n, res, &ldres, NULL);
    if (ltran1 && ltran2) {
        SLC_DGEMM("N", "T", &n, &n, &n, &dbl1, q2, &ldq2, q1, &ldq1, &dbl0, res, &ldres);
        SLC_DGEMM("N", "T", &n, &n, &n, &dbl1, q1, &ldq1, q2, &ldq2, &dblm1, res, &ldres);
    } else if (ltran1) {
        SLC_DGEMM("T", "T", &n, &n, &n, &dbl1, q2, &ldq2, q1, &ldq1, &dbl0, res, &ldres);
        SLC_DGEMM("N", "N", &n, &n, &n, &dbl1, q1, &ldq1, q2, &ldq2, &dblm1, res, &ldres);
    } else if (ltran2) {
        SLC_DGEMM("N", "N", &n, &n, &n, &dbl1, q2, &ldq2, q1, &ldq1, &dbl0, res, &ldres);
        SLC_DGEMM("T", "T", &n, &n, &n, &dbl1, q1, &ldq1, q2, &ldq2, &dblm1, res, &ldres);
    } else {
        SLC_DGEMM("T", "N", &n, &n, &n, &dbl1, q2, &ldq2, q1, &ldq1, &dbl0, res, &ldres);
        SLC_DGEMM("T", "N", &n, &n, &n, &dbl1, q1, &ldq1, q2, &ldq2, &dblm1, res, &ldres);
    }
    temp = hypot(temp, SLC_DLANGE("F", &n, &n, res, &ldres, NULL));

    return sqrt(2.0) * temp;
}
