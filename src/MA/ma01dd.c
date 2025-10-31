#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma01dd(
    const f64 ar1,
    const f64 ai1,
    const f64 ar2,
    const f64 ai2,
    const f64 eps,
    const f64 safemin,
    f64* d
)
{
    // For efficiency reasons, this routine does not check the input parameters for errors.
    f64 d1, d2;
    const f64 par = 4.0 - 2.0*eps;
    const f64 big = (par / safemin > par ? par / safemin : 1.0 / safemin);

    // Quick return if possible.
    f64 mx1 = fmax(fabs(ar1), fabs(ai1));
    f64 mx2 = fmax(fabs(ar2), fabs(ai2));
    f64 mx = fmax(mx1, mx2);

    if (mx == 0.0) {
        *d = 0.0;
        return;
    } else if (mx < big) {
        if (mx2 == 0.0) {
            *d = hypot(ar1, ai1);
            return;
        } else if (mx1 == 0.0) {
            *d = hypot(ar2, ai2);
            return;
        } else {
            d1 = hypot(ar1 - ar2, ai1 - ai2);
        }
    } else {
        d1 = big;
    }

    if (mx > 1.0 / big) {
        f64 ap1 = hypot(ar1, ai1);
        f64 ap2 = hypot(ar2, ai2);
        if ((mx1 <= big) && (mx2 <= big)) {
            d2 = hypot((ar1/ap1) / ap1 - (ar2/ap2) / ap2,
                       (ai2/ap2) / ap2 - (ai1/ap1) / ap1);
        } else if (mx1 <= big) {
            d2 = 1.0 / ap1;
        } else if (mx2 <= big) {
            d2 = 1.0 / ap2;
        } else {
            d2 = 0.0;
        }
    } else {
        d2 = big;
    }

    *d = fmin(d1, d2);

    return;
}
