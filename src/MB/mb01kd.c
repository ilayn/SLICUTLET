#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01kd(
    const i32 uplo,
    const i32 trans,
    const i32 n,
    const i32 k,
    const f64 alpha,
    const f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    const f64 beta,
    f64* c,
    const i32 ldc,
    i32* info
)
{
    // Skipped as this is DSYR2K which is available in BLAS
}
