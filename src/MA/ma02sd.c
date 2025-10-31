#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

f64
ma02sd(
    const i32 m,
    const i32 n,
    const f64* a,
    const i32 lda
)
{
    if ((m <= 0) || (n <= 0)) {
        return 0.0;
    }

    f64 tmp = SLC_DLAMCH("O");
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < m; i++) {
            f64 aij = fabs(a[i + j*lda]);
            if (aij > 0.0) {
                if (aij < tmp) {
                    tmp = aij;
                }
            }
        }
    }

    return tmp;
}
