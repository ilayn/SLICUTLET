#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

f64
ma02id(
    const i32 typ,   // 0:skew-Hermitian, 1:Hermitian
    const i32 norm,  // 0: 1-norm, 1: Frobenius norm, 2: infinity norm, 3: max norm
    const i32 n,
    const f64* a,
    const i32 lda,
    const f64* qg,
    const i32 ldqg,
    f64* dwork
)
{
    i32 int1 = 1;
    f64 value = 0.0, sum = 0.0, temp = 0.0;
    const f64 sq2 = sqrt(2.0);

    if (n == 0) {
        value = 0.0;
    } else if ((norm == 3) && (typ == 0)) {
        // Find max(abs(A[i, j])).
        value = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
        if (n > 1) {
            for (i32 j = 1; j <= n + 1; j++) {
                for (i32 i = 1; i <= j - 2; i++) {
                    value = fmax(value, fabs(qg[(i-1) + (j-1)*ldqg]));
                }
                for (i32 i = j+1; i <= n; i++) {
                    value = fmax(value, fabs(qg[(i-1) + (j-1)*ldqg]));
                }
            }
        }
    } else if (norm == 3) {
        // Find max(abs(a[i, j]), abs(qg[i, j])).
        i32 np1 = n + 1;
        value = fmax(SLC_DLANGE("M", &n, &n, a, &lda, dwork),
                     SLC_DLANGE("M", &n, &np1, qg, &ldqg, dwork));

    } else if (((norm == 0) || (norm == 2)) && (typ == 0)) {
        // Find the column and row sums of A (in one pass).
        value = 0.0;
        for (i32 i = 0; i < n; i++) {
            dwork[i] = 0.0;
        }

        for (i32 j = 0; j < n; j++) {
            sum = 0.0;
            for (i32 i = 0; i < n; i++) {
                temp = fabs(a[i + j*lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        // Compute the maximal absolute column sum.
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = fabs(qg[i + j*ldqg]);
                dwork[i] += temp;
                dwork[j-1] += temp;
            }
            sum = dwork[n + j];
            for (i32 i = j+1; i < n; i++) {
                f64 temp = fabs(qg[i + j*ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            value = fmax(value, sum);
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = fabs(qg[i + n*ldqg]);
            dwork[i] += temp;
            dwork[n-1] += temp;
        }
        for (i32 i = 0; i < n; i++) {
            value = fmax(value, dwork[i]);
        }

    } else if ((norm == 0) || (norm == 2)) {
        // Find the column and row sums of A (in one pass).
        value = 0.0;
        for (i32 i = 0; i < n; i++) {
            dwork[i] = 0.0;
        }

        for (i32 j = 0; j < n; j++) {
            sum = 0.0;
            for (i32 i = 0; i < n; i++) {
                temp = fabs(a[i + j*lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        // Compute the maximal absolute column sum.
        sum = dwork[n] + fabs(qg[0]);
        for (i32 i = 1; i < n; i++) {
            f64 temp = fabs(qg[i]);
            sum += temp;
            dwork[n + i] += temp;
        }
        value = fmax(value, sum);
        for (i32 j = 1; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = fabs(qg[i + j*ldqg]);
                dwork[i] += temp;
                dwork[j-1] += temp;
            }
            dwork[j-1] += fabs(qg[(j-1) + j*ldqg]);
            sum = dwork[n + j] + fabs(qg[j + j*ldqg]);
            for (i32 i = j+1; i < n; i++) {
                f64 temp = fabs(qg[i + j*ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            value = fmax(value, sum);
        }
        for (i32 i = 0; i < n - 2; i++) {
            f64 temp = fabs(qg[i + n*ldqg]);
            dwork[i] += temp;
            dwork[n-1] += temp;
        }
        dwork[n-1] += fabs(qg[(n-1) + n*ldqg]);
        for (i32 i = 0; i < n; i++) {
            value = fmax(value, dwork[i]);
        }

    } else if ((norm == 1) && (typ == 0)) {
        // Find normF(A).
        f64 scale = 0.0;
        sum = 1.0;
        for (i32 j = 0; j < n; j++) {
            SLC_DLASSQ(&n, &a[j*lda], &int1, &scale, &sum);
        }

        // Add normF(G) and normF(Q).
        SLC_DLASSQ(&(i32){n - 1}, &qg[1], &int1, &scale, &sum);
        SLC_DLASSQ(&(i32){n - 2}, &qg[2 + ldqg], &int1, &scale, &sum);
        for (i32 j = 2; j <= n - 2; j++) {
            SLC_DLASSQ(&(i32){j - 2}, &qg[j*ldqg], &int1, &scale, &sum);
            SLC_DLASSQ(&(i32){n - j - 1}, &qg[(j+1) + j*ldqg], &int1, &scale, &sum);
        }
        SLC_DLASSQ(&(i32){n - 2}, &qg[(n-1)*ldqg], &int1, &scale, &sum);
        SLC_DLASSQ(&(i32){n - 1}, &qg[n*ldqg], &int1, &scale, &sum);
        value = sq2 * scale * sqrt(sum);

    } else if (norm == 1) {
        f64 scale = 0.0;
        sum = 1.0;
        for (i32 j = 0; j < n; j++) {
            SLC_DLASSQ(&n, &a[j*lda], &int1, &scale, &sum);
        }

        f64 dscl = 0.0;
        f64 dsum = 1.0;
        SLC_DLASSQ(&int1, &qg[0], &int1, &dscl, &dsum);
        if (n > 1) {
            SLC_DLASSQ(&(i32){n - 1}, &qg[1], &int1, &scale, &sum);
        }
        for (i32 j = 1; j < n; j++) {
            SLC_DLASSQ(&(i32){j - 1}, &qg[j*ldqg], &int1, &scale, &sum);
            SLC_DLASSQ(&(i32){2}, &qg[(j-1) + j*ldqg], &int1, &dscl, &dsum);
            SLC_DLASSQ(&(i32){n - j - 1}, &qg[(j+1) + j*ldqg], &int1, &scale, &sum);
        }
        SLC_DLASSQ(&(i32){n - 1}, &qg[n*ldqg], &int1, &scale, &sum);
        SLC_DLASSQ(&int1, &qg[(n-1) + n*ldqg], &int1, &dscl, &dsum);
        value = hypot(sq2 * scale * sqrt(sum), dscl * sqrt(dsum));
    }

    return value;
}
