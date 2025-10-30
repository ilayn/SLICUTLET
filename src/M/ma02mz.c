#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>
#include <complex.h>

f64
ma02mz(
    const i32 norm, // 0: 1-norm, 1: Frobenius norm, 2: infinity norm, 3: max norm
    const i32 uplo, // 0: upper, 1: lower
    const i32 n,
    const c128* a,
    const i32 lda,
    f64* dwork
)
{
    f64 value = 0.0, sum = 0.0, absa = 0.0, scale = 0.0;

    if (n == 0) {
        value = 0.0;
    } else if (norm == 3) {
        // Find max(abs(a[i, j])).
        value = 0.0;
        if (uplo == 0) {
            for (i32 j = 1; j <= n; j++) {
                for (i32 i = 1; i <= j - 1; i++) {
                    value = fmax(value, cabs(a[(i-1) + (j-1)*lda]));
                }
                value = fmax(value, fabs(cimag(a[(j-1) + (j-1)*lda])));
            }
        } else {
            for (i32 j = 1; j <= n; j++) {
                value = fmax(value, fabs(cimag(a[(j-1) + (j-1)*lda])));
                for (i32 i = j+1; i <= n; i++) {
                    value = fmax(value, cabs(a[(i-1) + (j-1)*lda]));
                }
            }
        }

    } else if ((norm == 0) || (norm == 2)) {
        // Find normI(A) ( = norm1(A), since A is skew-Hermitian).
        value = 0.0;
        if (uplo == 0) {
            for (i32 j = 1; j <= n; j++) {
                sum = 0.0;
                for (i32 i = 1; i <= j - 1; i++) {
                    absa = cabs(a[(i-1) + (j-1)*lda]);
                    sum += absa;
                    dwork[i-1] += absa;
                }
                dwork[j-1] = sum + fabs(cimag(a[(j-1) + (j-1)*lda]));
            }
            for (i32 i = 0; i < n; i++) {
                value = fmax(value, dwork[i]);
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                dwork[i] = 0.0;
            }
            for (i32 j = 1; j <= n; j++) {
                sum = dwork[j-1] + fabs(cimag(a[(j-1) + (j-1)*lda]));
                for (i32 i = j+1; i <= n; i++) {
                    absa = cabs(a[(i-1) + (j-1)*lda]);
                    sum += absa;
                    dwork[i-1] += absa;
                }
                value = fmax(value, sum);
            }
            value = fmax(value, dwork[n-1]);
        }
    } else if (norm == 1) {
        // Find normF(A).
        scale = 0.0;
        sum = 1.0;
        if (uplo == 0) {
            for (i32 j = 2; j <= n; j++) {
                SLC_ZLASSQ(&(i32){j - 1}, &a[(j-1)*lda], &(i32){1}, &scale, &sum);
            }
        } else {
            for (i32 j = 1; j <= n-1; j++) {
                SLC_ZLASSQ(&(i32){n - j}, &a[j + (j-1)*lda], &(i32){1}, &scale, &sum);
            }
        }
        sum = 2.0 * sum;
        for (i32 i = 0; i < n; i++) {
            if (cimag(a[i + i*lda]) != 0.0) {
                absa = fabs(cimag(a[i + i*lda]));
                if (scale < absa) {
                    sum = 1.0 + sum * pow(scale / absa, 2);
                    scale = absa;
                } else {
                    sum += pow(absa / scale, 2);
                }
            }
        }
        value = scale * sqrt(sum);
    }

    return value;
}
