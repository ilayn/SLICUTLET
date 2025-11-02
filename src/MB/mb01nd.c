#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01nd(
    const i32 uplo,
    const i32 n,
    const f64 alpha,
    const f64* x,
    const i32 incx,
    const f64* y,
    const i32 incy,
    f64* a,
    const i32 lda
)
{
    // Skipped as this is DSYR2 which is available in BLAS
}
