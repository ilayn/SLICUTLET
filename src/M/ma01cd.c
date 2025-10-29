#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

i32
ma01cd(
    const f64 a,
    const i32 ia,
    const f64 b,
    const i32 ib
)
{
    i32 result;
    f64 s, sa, sb;

    if ((a == 0.0) && (b == 0.0)) {
        result = 0;
    } else if (a == 0.0) {
        result = (i32)copysign(1.0, b);
    } else if (b == 0.0) {
        result = (i32)copysign(1.0, a);
    } else if (ia == ib) {
        s = a + b;
        if (s == 0.0) {
            result = 0;
        } else {
            result = (i32)copysign(1.0, s);
        }
    } else {
        sa = copysign(1.0, a);
        sb = copysign(1.0, b);
        if (sa == sb) {
            result = (i32)sa;
        } else if (ia > ib) {
            if ((log(fabs(a)) + ia - ib) >= log(fabs(b))) {
                result = (i32)sa;
            } else {
                result = (i32)sb;
            }
        } else {
            if ((log(fabs(b)) + ib - ia) >= log(fabs(a))) {
                result = (i32)sb;
            } else {
                result = (i32)sa;
            }
        }
    }

    return result;
}
