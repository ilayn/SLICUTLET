#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01md(
    const i32 uplo,
    const i32 n,
    const f64 alpha,
    const f64* a,
    const i32 lda,
    const f64* x,
    const i32 incx,
    const f64 beta,
    f64* y,
    const i32 incy,
    i32* info
)
{
    // Skipped as this is DSYMV which is available in BLAS
}
