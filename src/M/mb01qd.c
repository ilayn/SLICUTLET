#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb01qd(
    const i32 type,
    const i32 m,
    const i32 n,
    const i32 kl,
    const i32 ku,
    const f64 cfrom,
    const f64 cto,
    const i32 nbl,
    const i32* nrows,
    f64* a,
    const i32 lda,
    i32* info
)
{
    *info = 0;

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

    // Quick return if possible
    if (MIN(m, n) == 0) {
        return;
    }

    // Get machine parameters
    f64 smlnum = SLC_DLAMCH("S");
    f64 bignum = 1.0 / smlnum;

    f64 cfromc = cfrom;
    f64 ctoc = cto;
    i32 done = 0;
    f64 mul;

    while (!done) {
        f64 cfrom1 = cfromc * smlnum;
        f64 cto1 = ctoc / bignum;
        if ((fabs(cfrom1) > fabs(ctoc)) && (ctoc != 0.0)) {
            mul = smlnum;
            done = 0;
            cfromc = cfrom1;
        } else if (fabs(cto1) > fabs(cfromc)) {
            mul = bignum;
            done = 0;
            ctoc = cto1;
        } else {
            mul = ctoc / cfromc;
            done = 1;
        }

        i32 noblc = (nbl == 0);

        if (itype == 0) {
            // Full matrix
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m; i++) {
                    a[i + j * lda] = a[i + j * lda] * mul;
                }
            }

        } else if (itype == 1) {
            if (noblc) {
                // Lower triangular matrix
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < m; i++) {
                        a[i + j * lda] = a[i + j * lda] * mul;
                    }
                }

            } else {
                // Block lower triangular matrix
                i32 jfin = 0;
                for (i32 k = 0; k < nbl; k++) {
                    i32 jini = jfin;
                    jfin = jfin + nrows[k];
                    for (i32 j = jini; j < jfin; j++) {
                        for (i32 i = jini; i < m; i++) {
                            a[i + j * lda] = a[i + j * lda] * mul;
                        }
                    }
                }
            }

        } else if (itype == 2) {
            if (noblc) {
                // Upper triangular matrix
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= MIN(j, m - 1); i++) {
                        a[i + j * lda] = a[i + j * lda] * mul;
                    }
                }

            } else {
                // Block upper triangular matrix
                i32 jfin = 0;
                for (i32 k = 0; k < nbl; k++) {
                    i32 jini = jfin;
                    jfin = jfin + nrows[k];
                    if (k == nbl - 1) jfin = n;
                    for (i32 j = jini; j < jfin; j++) {
                        for (i32 i = 0; i < MIN(jfin, m); i++) {
                            a[i + j * lda] = a[i + j * lda] * mul;
                        }
                    }
                }
            }

        } else if (itype == 3) {
            if (noblc) {
                // Upper Hessenberg matrix
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i < MIN(j + 2, m); i++) {
                        a[i + j * lda] = a[i + j * lda] * mul;
                    }
                }

            } else {
                // Block upper Hessenberg matrix
                i32 jfin = 0;
                for (i32 k = 0; k < nbl; k++) {
                    i32 jini = jfin;
                    jfin = jfin + nrows[k];

                    i32 ifin;
                    if (k == nbl - 1) {
                        jfin = n;
                        ifin = n;
                    } else {
                        ifin = jfin + nrows[k + 1];
                    }

                    for (i32 j = jini; j < jfin; j++) {
                        for (i32 i = 0; i < MIN(ifin, m); i++) {
                            a[i + j * lda] = a[i + j * lda] * mul;
                        }
                    }
                }
            }

        } else if (itype == 4) {
            // Lower half of a symmetric band matrix
            i32 k3 = kl + 1;
            i32 k4 = n;
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < MIN(k3, k4 - j); i++) {
                    a[i + j * lda] = a[i + j * lda] * mul;
                }
            }

        } else if (itype == 5) {
            // Upper half of a symmetric band matrix
            i32 k1 = ku + 1;
            i32 k3 = ku;
            for (i32 j = 0; j < n; j++) {
                for (i32 i = MAX(k1 - j - 1, 0); i <= k3; i++) {
                    a[i + j * lda] = a[i + j * lda] * mul;
                }
            }

        } else if (itype == 6) {
            // Band matrix
            i32 k1 = kl + ku + 1;
            i32 k2 = kl;
            i32 k3 = 2 * kl + ku;
            i32 k4 = kl + ku + m;
            for (i32 j = 0; j < n; j++) {
                for (i32 i = MAX(k1 - j - 1, k2); i <= MIN(k3, k4 - j - 1); i++) {
                    a[i + j * lda] = a[i + j * lda] * mul;
                }
            }
        }
    }

    return;
}
