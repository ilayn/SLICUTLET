#include "slicutlet.h"
#include "../include/types.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
mb03oy(
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    const f64 rcond,
    const f64 svlmax,
    i32* rank,
    f64* sval,
    i32* jpvt,
    f64* tau,
    f64* dwork,
    i32* info
)
{

    // Common integer and f64 constants for BLAS/LAPACK calls
    i32 int1 = 1;

    i32 imax = 1, imin = 2;
    i32 i, tmp_int, tmp_int2;
    f64 aii = 0.0, c1, c2, s1, s2, smax, smaxpr, smin, sminpr, temp, temp2;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < MAX(1, m)) {
        *info = -4;
    } else if (rcond < 0.0 || rcond > 1.0) {
        *info = -5;
    } else if (svlmax < 0.0) {
        *info = -6;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    i32 minmn = MIN(m, n);
    if (minmn == 0) {
        *rank = 0;
        sval[0] = 0.0;
        sval[1] = 0.0;
        sval[2] = 0.0;
        return;
    }

    f64 tolz = sqrt(DBL_EPSILON*0.5);
    i32 ismin = 0;
    i32 ismax = n - 1;

    // Initialize partial column norms and pivoting vector. The first n
    // elements of dwork store the exact column norms. The already used
    // leading part is then overwritten by the condition estimator.

    for (i32 i = 0; i < n; i++) {
        dwork[i] = SLC_DNRM2(&m, &a[i * lda], &int1);
        dwork[n + i] = dwork[i];
        jpvt[i] = i;
    }

    // Compute factorization and determine RANK using incremental condition estimation
    *rank = 0;

    while (1)
    {
        if (*rank < minmn)
        {
            i = *rank;

            // Determine ith pivot column and swap if necessary
            // IDAMAX returns 1-based index hence the extra -1
            i32 nrem = n - i;
            i32 pvt = i + SLC_IDAMAX(&nrem, &dwork[i], &int1) - 1;
            if (pvt != i)
            {
                // Swap columns pvt and i
                SLC_DSWAP(&m, &a[pvt * lda], &int1, &a[i * lda], &int1);
                i32 itemp = jpvt[pvt];
                jpvt[pvt] = jpvt[i];
                jpvt[i] = itemp;
                dwork[pvt] = dwork[i];
                dwork[n + pvt] = dwork[n + i];
            }

            // Save a(i, i) and generate elementary reflector H(i)
            if (i < m - 1) {
                aii = a[i + i * lda];
                tmp_int = m - i;
                SLC_DLARFG(&tmp_int, &a[i + i * lda], &a[(i + 1) + i * lda], &int1, &tau[i]);
            } else {
                tau[m-1] = 0.0;
            }

            if (*rank == 0) {

                // Initialize; exit if the matrix is negligible (RANK = 0)
                smax = fabs(a[0]);
                if (smax <= rcond) {
                    sval[0] = 0.0;
                    sval[1] = 0.0;
                    sval[2] = 0.0;
                    // return;
                }
                smin = smax;
                smaxpr = smax;
                sminpr = smin;
                c1 = 1.0;
                c2 = 1.0;

            } else {

                // One step of incremental condition estimation
                SLC_DLAIC1(&imin, rank, &dwork[ismin], &smin, &a[i * lda], &a[i + i * lda], &sminpr, &s1, &c1);
                SLC_DLAIC1(&imax, rank, &dwork[ismax], &smax, &a[i * lda], &a[i + i * lda], &smaxpr, &s2, &c2);

            }

            if (svlmax * rcond <= smaxpr) {
                if (svlmax * rcond <= sminpr) {
                    if (smaxpr * rcond < sminpr) {

                        // Continue factorization
                        if (i < n - 1) {

                            // Apply H(i) to A(i:m, i+1:n) from the left
                            aii = a[i + i * lda];
                            a[i + i * lda] = 1.0;
                            tmp_int = m - i;
                            tmp_int2 = n - i - 1;
                            SLC_DLARF("L", &tmp_int, &tmp_int2, &a[i + i * lda], &int1, &tau[i], &a[i + (i + 1) * lda], &lda, &dwork[2*n]);
                            a[i + i * lda] = aii;
                        }

                        // Update partial column norms
                        for (i32 j = i + 1; j < n; j++) {
                            if (dwork[j] != 0.0) {
                                temp = fabs(a[i + j * lda]) / dwork[j];
                                temp = fmax((1.0 + temp) * (1.0 - temp), 0.0);
                                temp2 = temp * pow(dwork[j] / dwork[n + j], 2);
                                if (temp2 <= tolz) {
                                    if (m - i - 1> 0) {
                                        tmp_int = m - i - 1;
                                        dwork[j] = SLC_DNRM2(&tmp_int, &a[(i + 1) + j * lda], &int1);
                                        dwork[n + j] = dwork[j];
                                    } else {
                                        dwork[j] = 0.0;
                                        dwork[n + j] = 0.0;
                                    }
                                } else {
                                    dwork[j] = dwork[j]*sqrt(temp);
                                }
                            }
                        }

                        for (i32 i = 0; i < *rank; i++) {
                            dwork[ismin + i] =  s1 * dwork[ismin + i];
                            dwork[ismax + i] =  s2 * dwork[ismax + i];
                        }

                        dwork[ismin + *rank] = c1;
                        dwork[ismax + *rank] = c2;
                        smin = sminpr;
                        smax = smaxpr;
                        (*rank)++;
                        continue;
                    }
                }
            }
        }
        break;
    }

    // Restore the changed part of the (rank+1)th column and set sval.

    if (*rank < n) {
        if (i < m - 1) {
            tmp_int = m - i - 1;
            temp = -a[i + i * lda]*tau[i];
            SLC_DSCAL(&tmp_int, &temp, &a[(i + 1) + i * lda], &int1);
            a[i + i * lda] = aii;
        }
    }

    if (*rank == 0) {
        smin = 0.0;
        sminpr = 0.0;
    }

    sval[0] = smax;
    sval[1] = smin;
    sval[2] = sminpr;

    return;
}
