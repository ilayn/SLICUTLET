#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma01bd(
    const f64 base,
    const f64 logbase,
    const i32 k,
    const i32* s,
    const f64* a,
    const i32 inca,
    f64* alpha,
    f64* beta,
    i32* scale
)
{
    i32 sl = 0;

    *alpha = 1.0;
    *beta = 1.0;
    *scale = 0;

    for (i32 i = 0; i < k; i++) {
        f64 temp = a[i * inca];
        if (temp != 0.0) {
            sl = (i32)(log(fabs(temp)) / logbase);
            temp = temp / base / (pow(base, sl-1));
        } else {
            sl = 0;
        }
        if (s[i] == 1) {
            *alpha = *alpha * temp;
            *scale += sl;
        } else {
            *beta = *beta * temp;
            *scale -= sl;
        }
        if (((i+1) % 10 == 0) && (i > 0)) {
            if (*alpha != 0.0) {
                sl = (i32)(log(fabs(*alpha)) / logbase);
                *scale += sl;
                *alpha = *alpha / base / (pow(base, sl-1));
            }
            if (*beta != 0.0) {
                sl = (i32)(log(fabs(*beta)) / logbase);
                *scale -= sl;
                *beta = *beta / base / (pow(base, sl-1));
            }
        }
    }

    if (*beta != 0.0) {
        *alpha = *alpha / *beta;
        *beta = 1.0;
    }
    if (*alpha == 0.0) {
        *scale = 0;
    } else {
        sl = (i32)(log(fabs(*alpha)) / logbase);
        *alpha = *alpha / base / (pow(base, sl-1));
        *scale += sl;
    }

    return;
}
