#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01pd(
    const i32 scun,
    const i32 type,
    const i32 m,
    const i32 n,
    const i32 kl,
    const i32 ku,
    const f64 anrm,
    const i32 nbl,
    const i32* nrows,
    f64* a,
    const i32 lda,
    i32* info
)
{
    *info = 0;

    // Map scun to lscale (0='U', 1='S')
    i32 lscale;
    if (scun == 1) {
        lscale = 1;
    } else {
        lscale = 0;
    }

    // Map type to itype (0=G, 1=L, 2=U, 3=H, 4=B, 5=Q, 6=Z)
    i32 itype;
    if (type == 0) {        // 'G'
        itype = 0;
    } else if (type == 1) { // 'L'
        itype = 1;
    } else if (type == 2) { // 'U'
        itype = 2;
    } else if (type == 3) { // 'H'
        itype = 3;
    } else if (type == 4) { // 'B'
        itype = 4;
    } else if (type == 5) { // 'Q'
        itype = 5;
    } else {                // 'Z'
        itype = 6;
    }

    i32 mn = MIN(m, n);

    i32 isum = 0;
    if (nbl > 0) {
        for (i32 i = 0; i < nbl; i++) {
            isum = isum + nrows[i];
        }
    }

    // Test the input scalar arguments
    if ((scun != 0) && (scun != 1)) {
        *info = -1;
    } else if (itype == -1) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if ((n < 0) || (((itype == 4) || (itype == 5)) && (n != m))) {
        *info = -4;
    } else if (anrm < 0.0) {
        *info = -7;
    } else if (nbl < 0) {
        *info = -8;
    } else if ((nbl > 0) && (isum != mn)) {
        *info = -9;
    } else if ((itype <= 3) && (lda < MAX(1, m))) {
        *info = -11;
    } else if (itype >= 4) {
        if ((kl < 0) || (kl > MAX(m - 1, 0))) {
            *info = -5;
        } else if ((ku < 0) || (ku > MAX(n - 1, 0)) ||
                   (((itype == 4) || (itype == 5)) && (kl != ku))) {
            *info = -6;
        } else if (((itype == 4) && (lda < kl + 1)) ||
                   ((itype == 5) && (lda < ku + 1)) ||
                   ((itype == 6) && (lda < 2 * kl + ku + 1))) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if ((mn == 0) || (anrm == 0.0)) {
        return;
    }

    // Get machine parameters
    f64 smlnum = SLC_DLAMCH("S") / SLC_DLAMCH("P");
    f64 bignum = 1.0 / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    if (lscale) {
        // Scale A, if its norm is outside range [SMLNUM,BIGNUM]
        if (anrm < smlnum) {
            // Scale matrix norm up to SMLNUM
            mb01qd(type, m, n, kl, ku, anrm, smlnum, nbl, nrows, a, lda, info);
        } else if (anrm > bignum) {
            // Scale matrix norm down to BIGNUM
            mb01qd(type, m, n, kl, ku, anrm, bignum, nbl, nrows, a, lda, info);
        }

    } else {
        // Undo scaling
        if (anrm < smlnum) {
            mb01qd(type, m, n, kl, ku, smlnum, anrm, nbl, nrows, a, lda, info);
        } else if (anrm > bignum) {
            mb01qd(type, m, n, kl, ku, bignum, anrm, nbl, nrows, a, lda, info);
        }
    }

    return;
}
