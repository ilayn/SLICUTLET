#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01xd(
    const i32 uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    i32 *info
)
{
    const f64 dbl1 = 1.0;
    const i32 int1 = 1, intm1 = -1;
    i32 i, ib, ii, nb;
    i32 upper = (uplo == 1);

    *info = 0;

    if((uplo != 0) && (uplo != 1)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if(lda < MAX(1, n)) {
        *info = -4;
    }

    if(*info != 0) {
        return;
    }

    if(n == 0)
        return;

    mb01xy(uplo, n, a, lda, info);

    return;
}
