#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02fd(
    f64* x1,
    f64* x2,
    f64* c,
    f64* s,
    i32* info
)
{

    if (((*x1 != 0.0) || (*x2 != 0.0)) && (fabs(*x2) >= fabs(*x1))) {
        *info = 1;
    } else {
        *info = 0;
        if (*x1 == 0.0) {
            *s = 0.0;
            *c = 1.0;
        } else {
            *s = *x2 / *x1;

            //    No overflows could appear in the next statement; underflows
            //    are possible if X2 is tiny and X1 is huge, but then
            //       abs(C) = ONE - delta,
            //    where delta is much less than machine precision.

            *c = copysign(sqrt(1.0 - *s )*sqrt(1.0 + *s), *x1);
            *x1 = (*c) * (*x1);
        }
    }

    return;
}
