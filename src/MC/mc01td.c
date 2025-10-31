#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mc01td(
    const i32 dico, // 0 discrete, 1 continuous
    i32* dp,
    f64* p,
    i32* stable,
    i32* nz,
    f64* dwork,
    i32* iwarn,
    i32* info
)
{
    i32 int1 = 1, intm1 = -1;
    *iwarn = 0;
    *info = 0;

    // Test the input scalar arguments
    if (dico != 0 && dico != 1) {
        *info = -1;
    } else if (*dp < 0) {
        *info = -2;
    }

    if (*info != 0) {
        return;
    }

    // Trim trailing zeros of P
    while (*dp >= 0 && p[*dp] == 0.0) {
        (*dp)--;
        *iwarn += 1;
    }

    if (*dp < 0) {
        *info = 1;
        return;
    }

    // P(x) is not the zero polynomial and its degree is dp.
    if (dico) {
        // Continuous-time case
        // Compute the Routh coefficients and the number of sign changes.

        i32 ncp = *dp + 1;
        SLC_DCOPY(&ncp, p, &int1, dwork, &int1);
        *nz = 0;
        i32 k = *dp;

        // WHILE ( K > 0 and DWORK(K) non-zero) DO
        while (k > 0) {
            if (dwork[k - 1] == 0.0) {
                *info = 2;
                break;
            }
            // Fortran: ALPHA = DWORK(K+1)/DWORK(K) => C: dwork[k]/dwork[k-1]
            f64 alpha = dwork[k] / dwork[k - 1];
            if (alpha < 0.0) { (*nz)++; }
            k--;
            for (i32 j = k - 1; j >= 1; j -= 2) {
                dwork[j] = dwork[j] - alpha * dwork[j - 1];
            }
        }
        // END WHILE
    } else {
        // Discrete-time case

        // To apply [3], section 6.8, on the reciprocal of polynomial
        // P(x), the elements of the array P are copied in DWORK in reverse order.
        i32 ncp = *dp + 1;
        SLC_DCOPY(&ncp, p, &int1, dwork, &intm1);

        //                                                           K-1
        //        DWORK(K),...,DWORK(DP+1), are the coefficients of T   P(x)
        //        scaled with a factor alpha(K) in order to avoid over- or
        //        underflow,
        //                                                    i-1
        //        DWORK(i), i = 1,...,K, contains alpha(i) * T   P(0).
        //

        f64 signum = 1.0;
        *nz = 0;
        i32 k = 1;

        // WHILE ( K <= DP and INFO.EQ.0 ) DO
        while ((k <= *dp) && (*info == 0)) {
            // Compute the coefficients of T^K P(x)
            // K1 = DP - K + 2; K2 = DP + 2
            i32 k1 = *dp - k + 2;
            i32 k2 = *dp + 2;

            // ALPHA = DWORK(K-1+IDAMAX( K1, DWORK(K), 1 ))
            i32 k_off = k - 1; // zero-based base
            i32 k1_for = k1;   // length
            i32 idx = SLC_IDAMAX(&k1_for, &dwork[k_off], &int1);
            f64 alpha = dwork[k_off + (idx - 1)];
            if (alpha == 0.0) {
                *info = 2;
            } else {
                // DCOPY(K1, DWORK(K), 1, DWORK(K2), 1); DRSCL(K1, ALPHA, DWORK(K2), 1)
                SLC_DCOPY(&k1_for, &dwork[k_off], &int1, &dwork[k2 - 1], &int1);
                SLC_DRSCL(&k1_for, &alpha, &dwork[k2 - 1], &int1);

                f64 p1  = dwork[k2 - 1];
                f64 pk1 = dwork[k2 - 1 + (k1 - 1)];

                // for I = 1..K1-1: DWORK(K+I) = P1*DWORK(DP+1+I) - PK1*DWORK(K2+K1-I)
                for (i32 i = 1; i <= k1 - 1; ++i) {
                    dwork[k - 1 + i] = p1 * dwork[*dp + i] - pk1 * dwork[(k2 - 1) + (k1 - i)];
                }

                // Compute the number of unstable zeros
                k = k + 1;
                if (dwork[k - 1] == 0.0) {
                    *info = 2;
                } else {
                    signum = copysign(1.0, signum * dwork[k - 1]);
                    if (signum < 0.0) (*nz)++;
                }
            }
        }
        // END WHILE
    }

    if ((*info == 0) && (*nz == 0)) {
        *stable = 1;
    } else {
        *stable = 0;
    }

    return;
}
