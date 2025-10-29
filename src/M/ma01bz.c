#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma01bz(
    const f64 base,
    const i32 k,
    const i32* s,
    const c128* a,
    const i32 inca,
    c128* alpha,
    c128* beta,
    i32* scale
)
{
    i32 i, inda = 0;
    const c128 cbase = CMPLX(base, 0.0);
    *alpha = CMPLX(1.0, 0.0);
    *beta = CMPLX(1.0, 0.0);
    *scale = 0;

    for (i = 0; i < k; ++i) {
        if (s[i] == 1) {
            *alpha = (*alpha) * a[inda];
        } else {
            if (a[inda] == CMPLX(0.0, 0.0)) {
                *beta = CMPLX(0.0, 0.0);
            } else {
                *alpha = (*alpha) / a[inda];
            }
        }
        if (cabs(*alpha) == 0.0) {
            *alpha = CMPLX(0.0, 0.0);
            *scale = 0;
            if (cabs(*beta) == 0.0) { return; }
        } else {
            while (cabs(*alpha) < 1.0) {
                *alpha = (*alpha) * cbase;
                (*scale) -= 1;
            }
            while (cabs(*alpha) >= base) {
                *alpha = (*alpha) / cbase;
                (*scale) += 1;
            }
        }
        inda += inca;
    }

    return;
}
