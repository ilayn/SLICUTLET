#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>
#include <complex.h>

f64
ma02iz(
    const i32 typ,   // 0:skew-Hamiltonian, 1:Hamiltonian
    const i32 norm,  // 0: 1-norm, 1: Frobenius norm, 2: infinity norm, 3: max norm
    const i32 n,
    const c128* a,
    const i32 lda,
    const c128* qg,
    const i32 ldqg,
    f64* dwork
)
{
    f64 value = 0.0, sum = 0.0, temp = 0.0;
    const i32 int1 = 1;
    const f64 sq2 = sqrt(2.0);

    if (n == 0) {
        value = 0.0;
    } else if ((norm == 3) && (typ == 0)) {
        // Find max(abs(A[i, j])).
        value = SLC_ZLANGE("M", &n, &n, a, &lda, dwork);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                value = fmax(value, cabs(qg[i + j*ldqg]));
            }
            value = fmax(value, fabs(cimag(qg[j + j*ldqg])));
            for (i32 i = j+1; i < n; i++) {
                value = fmax(value, cabs(qg[i + j*ldqg]));
            }
            value = fmax(value, fabs(cimag(qg[j + (j+1)*ldqg])));
        }
        for (i32 i = 0; i < n - 1; i++) {
            value = fmax(value, cabs(qg[i + n*ldqg]));
        }

    } else if (norm == 3) {
        // Find max(abs(a[i, j]), abs(qg[i, j])).
        value = SLC_ZLANGE("M", &n, &n, a, &lda, dwork);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                value = fmax(value, cabs(qg[i + j*ldqg]));
            }
            value = fmax(value, fabs(creal(qg[j + j*ldqg])));
            for (i32 i = j+1; i < n; i++) {
                value = fmax(value, cabs(qg[i + j*ldqg]));
            }
            value = fmax(value, fabs(creal(qg[j + (j+1)*ldqg])));
        }
        for (i32 i = 0; i < n - 1; i++) {
            value = fmax(value, cabs(qg[i + n*ldqg]));
        }

    } else if (((norm == 0) || (norm == 2)) && (typ == 0)) {
        // Find the column and row sums of A (in one pass).
        value = 0.0;
        for (i32 i = 0; i < n; i++) {
            dwork[i] = 0.0;
        }

        for (i32 j = 0; j < n; j++) {
            sum = 0.0;
            for (i32 i = 0; i < n; i++) {
                temp = cabs(a[i + j*lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        // Compute the maximal absolute column sum.
        sum = dwork[n] + fabs(cimag(qg[0]));
        for (i32 i = 1; i < n; i++) {
            f64 temp = cabs(qg[i]);
            sum += temp;
            dwork[n + i] += temp;
        }
        value = fmax(value, sum);
        for (i32 j = 1; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = cabs(qg[i + j*ldqg]);
                dwork[i] += temp;
                dwork[j-1] += temp;
            }
            dwork[j-1] += fabs(cimag(qg[(j-1) + j*ldqg]));
            sum = dwork[n + j] + fabs(cimag(qg[j + j*ldqg]));
            for (i32 i = j+1; i < n; i++) {
                f64 temp = cabs(qg[i + j*ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            value = fmax(value, sum);
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = cabs(qg[i + n*ldqg]);
            dwork[i] += temp;
            dwork[n-1] += temp;
        }
        dwork[n-1] += fabs(cimag(qg[(n-1) + n*ldqg]));
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
                temp = cabs(a[i + j*lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        // Compute the maximal absolute column sum.
        sum = dwork[n] + fabs(creal(qg[0]));
        for (i32 i = 1; i < n; i++) {
            f64 temp = cabs(qg[i]);
            sum += temp;
            dwork[n + i] += temp;
        }
        value = fmax(value, sum);
        for (i32 j = 1; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = cabs(qg[i + j*ldqg]);
                dwork[i] += temp;
                dwork[j-1] += temp;
            }
            dwork[j-1] += fabs(creal(qg[(j-1) + j*ldqg]));
            sum = dwork[n + j] + fabs(creal(qg[j + j*ldqg]));
            for (i32 i = j+1; i < n; i++) {
                f64 temp = cabs(qg[i + j*ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            value = fmax(value, sum);
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = cabs(qg[i + n*ldqg]);
            dwork[i] += temp;
            dwork[n-1] += temp;
        }
        dwork[n-1] += fabs(creal(qg[(n-1) + n*ldqg]));
        for (i32 i = 0; i < n; i++) {
            value = fmax(value, dwork[i]);
        }

    } else if ((norm == 1) && (typ == 0)) {
        // Find normF(A).
        f64 scale = 0.0;
        sum = 1.0;
        for (i32 j = 0; j < n; j++) {
            SLC_ZLASSQ(&n, &a[j*lda], &int1, &scale, &sum);
        }

        // Add normF(G) and normF(Q).
        f64 dscl = fabs(cimag(qg[0]));
        f64 dsum = 1.0;
        if (n > 1) {
            SLC_ZLASSQ(&(i32){n - 1}, &qg[1], &int1, &scale, &sum);
            f64 dum[2] = {cimag(qg[ldqg]), cimag(qg[1 + ldqg])};
            SLC_DLASSQ(&(i32){2}, dum, &int1, &dscl, &dsum);
        }
        SLC_ZLASSQ(&(i32){n - 2}, &qg[2 + ldqg], &int1, &scale, &sum);
        for (i32 j = 2; j < n; j++) {
            SLC_ZLASSQ(&(i32){j - 1}, &qg[j*ldqg], &int1, &scale, &sum);
            f64 dum[2] = {cimag(qg[(j-1) + j*ldqg]), cimag(qg[j + j*ldqg])};
            SLC_DLASSQ(&(i32){2}, dum, &int1, &dscl, &dsum);
            SLC_ZLASSQ(&(i32){n - j - 1}, &qg[(j+1) + j*ldqg], &int1, &scale, &sum);
        }
        if (n > 1) {
            SLC_ZLASSQ(&(i32){n - 1}, &qg[n*ldqg], &int1, &scale, &sum);
        }
        f64 dum = cimag(qg[(n-1) + n*ldqg]);
        SLC_DLASSQ(&int1, &dum, &int1, &dscl, &dsum);
        value = hypot(sq2 * scale * sqrt(sum), dscl * sqrt(dsum));

    } else if (norm == 1) {
        // Find normF(A).
        f64 scale = 0.0;
        sum = 1.0;
        for (i32 j = 0; j < n; j++) {
            SLC_ZLASSQ(&n, &a[j*lda], &int1, &scale, &sum);
        }

        // Add normF(G) and normF(Q).
        f64 dscl = fabs(creal(qg[0]));
        f64 dsum = 1.0;
        if (n > 1) {
            SLC_ZLASSQ(&(i32){n - 1}, &qg[1], &int1, &scale, &sum);
            f64 dum[2] = {creal(qg[ldqg]), creal(qg[1 + ldqg])};
            SLC_DLASSQ(&(i32){2}, dum, &int1, &dscl, &dsum);
        }
        SLC_ZLASSQ(&(i32){n - 2}, &qg[2 + ldqg], &int1, &scale, &sum);
        for (i32 j = 2; j < n; j++) {
            SLC_ZLASSQ(&(i32){j - 1}, &qg[j*ldqg], &int1, &scale, &sum);
            f64 dum[2] = {creal(qg[(j-1) + j*ldqg]), creal(qg[j + j*ldqg])};
            SLC_DLASSQ(&(i32){2}, dum, &int1, &dscl, &dsum);
            SLC_ZLASSQ(&(i32){n - j - 1}, &qg[(j+1) + j*ldqg], &int1, &scale, &sum);
        }
        if (n > 1) {
            SLC_ZLASSQ(&(i32){n - 1}, &qg[n*ldqg], &int1, &scale, &sum);
        }
        f64 dum = creal(qg[(n-1) + n*ldqg]);
        SLC_DLASSQ(&int1, &dum, &int1, &dscl, &dsum);
        value = hypot(sq2 * scale * sqrt(sum), dscl * sqrt(dsum));
    }

    return value;
}
