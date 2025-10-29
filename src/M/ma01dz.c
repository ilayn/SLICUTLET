#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma01dz(
    const f64 ar1,
    const f64 ai1,
    const f64 b1,
    const f64 ar2,
    const f64 ai2,
    const f64 b2,
    const f64 eps,
    const f64 safemin,
    f64* d1,
    f64* d2,
    i32* iwarn
)
{
    // For efficiency reasons, this routine does not check the input parameters for errors.
    *iwarn = 0;

    const f64 par = 4.0 - 2.0*eps;
    const f64 big = (par / safemin > par ? par / safemin : 1.0 / safemin);

    f64 mx1 = fmax(fabs(ar1), fabs(ai1));
    f64 mx2 = fmax(fabs(ar2), fabs(ai2));

    if (b1 == 0.0) {
        if (mx1 == 0.0) {
            *d1 = 0.0;
            *d2 = 0.0;
            *iwarn = 1;
        } else {
            if (b2 == 0.0) {
                *d1 = 0.0;
                if (mx2 == 0.0) {
                    *d2 = 0.0;
                    *iwarn = 1;
                } else {
                    *d2 = 1.0;
                }
            } else if (b2 > 1.0) {
                if (mx2 > b2 / big) {
                    *d1 = b2 / hypot(ar2, ai2);
                    *d2 = 1.0;
                } else {
                    *d1 = 1.0;
                    *d2 = 0.0;
                }
            } else if (mx2 > 0.0) {
                *d1 = b2 / hypot(ar2, ai2);
                *d2 = 1.0;
            } else {
                *d1 = 1.0;
                *d2 = 0.0;
            }
        }
    } else if (b2 == 0.0) {
        if (mx2 == 0.0) {
            *d1 = 0.0;
            *d2 = 0.0;
            *iwarn = 1;
        } else {
            if (b1 > 1.0) {
                if (mx1 > b1 / big) {
                    *d1 = b1 / hypot(ar1, ai1);
                    *d2 = 1.0;
                } else {
                    *d1 = 1.0;
                    *d2 = 0.0;
                }
            } else if (mx1 > 0.0) {
                *d1 = b1 / hypot(ar1, ai1);
                *d2 = 1.0;
            } else {
                *d1 = 1.0;
                *d2 = 0.0;
            }
        }
    } else {
        // ZERj = true means that Aj is practically 0.
        // INFj = true means that Aj is infinite.
        i32 inf1, inf2, zer1, zer2;
        f64 ap1, ap2;

        if (b1 >= 1.0) {
            inf1 = 0;
            ap1 = hypot(ar1 / b1, ai1 / b1);
            zer1 = ap1 < (1.0 / big);
        } else {
            zer1 = 0;
            inf1 = mx1 > (b1 * big);
            if (!inf1) {
                ap1 = hypot(ar1 / b1, ai1 / b1);
            }
        }

        if (b2 >= 1.0) {
            inf2 = 0;
            ap2 = hypot(ar2 / b2, ai2 / b2);
            zer2 = ap2 < (1.0 / big);
        } else {
            zer2 = 0;
            inf2 = mx2 > (b2 * big);
            if (!inf2) {
                ap2 = hypot(ar2 / b2, ai2 / b2);
            }
        }

        // A1 and/or A2 are/is (practically) 0.
        *d2 = 1.0;
        if (zer1 && zer2) {
            *d1 = 0.0;
        } else if (zer1) {
            if (!inf2) {
                *d1 = ap2;
            } else {
                *d1 = 1.0;
                *d2 = 0.0;
            }
        } else if (zer2) {
            if (!inf1) {
                *d1 = ap1;
            } else {
                *d1 = 1.0;
                *d2 = 0.0;
            }
        } else if (inf1) {
            // A1 and possibly A2 is/are practically infinite.
            if (inf2) {
                *d1 = 0.0;
            } else {
                *d1 = b2 / hypot(ar2, ai2);
            }
        } else if (inf2) {
            // A2 is practically infinite.
            *d1 = b1 / hypot(ar1, ai1);
        } else {
            // A1 and A2 are finite, representable numbers.
            f64 pr1 = ar1 / b1;
            f64 pi1 = ai1 / b1;
            f64 pr2 = ar2 / b2;
            f64 pi2 = ai2 / b2;
            *d1 = fmin(hypot(pr1 - pr2, pi1 - pi2),
                       hypot((pr1 / ap1) / ap1 - (pr2 / ap2) / ap2,
                             (pi2 / ap2) / ap2 - (pi1 / ap1) / ap1));
        }
    }

    return;
}
