#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma01ad(
    const f64 xr,
    const f64 xi,
    f64* yr,
    f64* yi
)
{

    f64 s = sqrt(0.5*(hypot(xr, xi) + fabs(xr)));
    if (xr >= 0.0) { *yr = s; }
    if (xi < 0.0) { s = -s; }
    if (xr <= 0.0) {
        *yi = s;
        if (xr != 0.0) { *yr = 0.5*(xi / s); }
    } else {
        *yi = 0.5*(xi / *yr);
    }

    return;
}
