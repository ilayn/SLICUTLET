#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01oo(
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 n,
    const f64* h,
    const i32 ldh,
    const f64* x,
    const i32 ldx,
    const f64* e,
    const i32 lde,
    f64* p,
    const i32 ldp,
    i32* info
)
{
    f64 dbl1 = 1.0;

    *info = 0;
    if (uplo != 0 && uplo != 1) {
        *info = -1;
    } else if (trans != 0 && trans != 1) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldh < MAX(1, n)) {
        *info = -5;
    } else if (ldx < MAX(1, n)) {
        *info = -7;
    } else if (lde < MAX(1, n)) {
        *info = -9;
    } else if (ldp < MAX(1, n)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (n == 0) {
        return;
    }

    // Compute W := H*X, if TRANS = 'N'.
    // Compute W := X*H, if TRANS = 'T'.
    mb01os(uplo, trans, n, h, ldh, x, ldx, p, ldp, info);

    // Compute P = W*E' = H*X*E', if TRANS = 'N', or
    // compute P = E'*W = E'*X*H, if TRANS = 'T'.
    SLC_DTRMM((trans ? "L" : "R"), "U", "T", "N", &n, &n, &dbl1, e, &lde, p, &ldp);

    return;
}
