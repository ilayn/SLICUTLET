#include "../include/types.h"
#include "../include/slc_blaslapack.h"

void ab05nd(
    const i32 over,
    const i32 n1,
    const i32 m1,
    const i32 p1,
    const i32 n2,
    const f64 alpha,
    f64* a1,
    const i32 lda1,
    f64* b1,
    const i32 ldb1,
    f64* c1,
    const i32 ldc1,
    f64* d1,
    const i32 ldd1,
    f64* a2,
    const i32 lda2,
    f64* b2,
    const i32 ldb2,
    f64* c2,
    const i32 ldc2,
    f64* d2,
    const i32 ldd2,
    i32* n,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    f64* d,
    i32 ldd,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
) {
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 ione = 1;

    i32 ldwm1 = (1 > m1) ? 1 : m1;
    *n = n1 + n2;
    *info = 0;

    // Test the input scalar arguments
    if ((over != 0) && (over != 1)) {
        *info = -1;
    } else if (n1 < 0) {
        *info = -2;
    } else if (m1 < 0) {
        *info = -3;
    } else if (p1 < 0) {
        *info = -4;
    } else if (n2 < 0) {
        *info = -5;
    } else if (lda1 < ((1 > n1) ? 1 : n1)) {
        *info = -8;
    } else if (ldb1 < ((1 > n1) ? 1 : n1)) {
        *info = -10;
    } else if (((n1 > 0) && (ldc1 < ((1 > p1) ? 1 : p1))) ||
               ((n1 == 0) && (ldc1 < 1))) {
        *info = -12;
    } else if (ldd1 < ((1 > p1) ? 1 : p1)) {
        *info = -14;
    } else if (lda2 < ((1 > n2) ? 1 : n2)) {
        *info = -16;
    } else if (ldb2 < ((1 > n2) ? 1 : n2)) {
        *info = -18;
    } else if (((n2 > 0) && (ldc2 < ldwm1)) ||
               ((n2 == 0) && (ldc2 < 1))) {
        *info = -20;
    } else if (ldd2 < ldwm1) {
        *info = -22;
    } else if (lda < ((1 > *n) ? 1 : *n)) {
        *info = -25;
    } else if (ldb < ((1 > *n) ? 1 : *n)) {
        *info = -27;
    } else if (((*n > 0) && (ldc < ((1 > p1) ? 1 : p1))) ||
               ((*n == 0) && (ldc < 1))) {
        *info = -29;
    } else if (ldd < ((1 > p1) ? 1 : p1)) {
        *info = -31;
    } else {
        i32 ldw = (p1 * p1 > m1 * m1) ? p1 * p1 : m1 * m1;
        if (n1 * p1 > ldw) ldw = n1 * p1;
        if (over == 1) {
            if (m1 > (*n) * n2) {
                i32 temp = m1 * (m1 + 1);
                if (temp > ldw) ldw = temp;
            }
            ldw = n1 * p1 + ldw;
        }
        if (ldwork < ((1 > ldw) ? 1 : ldw)) {
            *info = -34;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if (MAX(*n, MIN(m1, p1)) == 0) {
        return;
    }

    if (p1 > 0) {
        // Form (I + alpha * D1 * D2)
        SLC_DLASET("F", &p1, &p1, &zero, &one, dwork, &p1);
        SLC_DGEMM("N", "N", &p1, &p1, &m1, &alpha, d1, &ldd1, d2, &ldd2, &one, dwork, &p1);

        // Factorize this matrix
        SLC_DGETRF(&p1, &p1, dwork, &p1, iwork, info);

        if (*info != 0) {
            return;
        }

        // Form E21 * D1
        if ((over == 1) && (ldd1 <= ldd)) {
            if (ldd1 < ldd) {
                for (i32 j = m1 - 1; j >= 0; j--) {
                    for (i32 i = p1 - 1; i >= 0; i--) {
                        d[i + j * ldd] = d1[i + j * ldd1];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &p1, &m1, d1, &ldd1, d, &ldd);
        }

        SLC_DGETRS("N", &p1, &m1, dwork, &p1, iwork, d, &ldd, info);

        if (n1 > 0) {
            // Form E21 * C1
            if (over == 1) {
                // First save C1 to workspace after the temp area
                // Temp area size: max(p1*p1, m1*m1, n1*p1)
                i32 temp_size = MAX(p1 * p1, m1 * m1);
                if (n1 * p1 > temp_size) temp_size = n1 * p1;
                i32 ldw_offset = temp_size;

                SLC_DLACPY("F", &p1, &n1, c1, &ldc1, &dwork[ldw_offset], &p1);

                // Always copy to C (whether or not ldc1 == ldc)
                // Because C needs to be initialized for DGETRS to work on it
                SLC_DLACPY("F", &p1, &n1, &dwork[ldw_offset], &p1, c, &ldc);
            } else {
                SLC_DLACPY("F", &p1, &n1, c1, &ldc1, c, &ldc);
            }

            SLC_DGETRS("N", &p1, &n1, dwork, &p1, iwork, c, &ldc, info);
        }

        // Form E12 = I - alpha * D2 * (E21 * D1)
        SLC_DLASET("F", &m1, &m1, &zero, &one, dwork, &ldwm1);
        f64 neg_alpha = -alpha;
        SLC_DGEMM("N", "N", &m1, &m1, &p1, &neg_alpha, d2, &ldd2, d, &ldd, &one, dwork, &ldwm1);
    } else {
        SLC_DLASET("F", &m1, &m1, &zero, &one, dwork, &ldwm1);
    }

    // Handle A matrix
    if ((over == 1) && (lda1 <= lda)) {
        if (lda1 < lda) {
            for (i32 j = n1 - 1; j >= 0; j--) {
                for (i32 i = n1 - 1; i >= 0; i--) {
                    a[i + j * lda] = a1[i + j * lda1];
                }
            }
        }
    } else {
        SLC_DLACPY("F", &n1, &n1, a1, &lda1, a, &lda);
    }

    if ((n1 > 0) && (m1 > 0)) {
        // Form B1 * E12
        if (over == 1) {
            // Use the blocks (1,2) and (2,2) of A as workspace
            if (n1 * m1 <= (*n) * n2) {
                // Use BLAS 3 code
                f64* a_workspace = &a[n1 * lda];
                SLC_DLACPY("F", &n1, &m1, b1, &ldb1, a_workspace, &n1);
                SLC_DGEMM("N", "N", &n1, &m1, &m1, &one, a_workspace, &n1, dwork, &ldwm1, &zero, b, &ldb);
            } else if (ldb1 < ldb) {
                for (i32 j = m1 - 1; j >= 0; j--) {
                    for (i32 i = n1 - 1; i >= 0; i--) {
                        b[i + j * ldb] = b1[i + j * ldb1];
                    }
                }

                if (m1 <= (*n) * n2) {
                    // Use BLAS 2 code
                    for (i32 j = 0; j < n1; j++) {
                        SLC_DCOPY(&m1, &b[j], &ldb, &a[n1 * lda], &ione);
                        SLC_DGEMV("T", &m1, &m1, &one, dwork, &ldwm1, &a[n1 * lda], &ione, &zero, &b[j], &ldb);
                    }
                } else {
                    // Use additional workspace
                    i32 dwork_offset = m1 * m1;
                    for (i32 j = 0; j < n1; j++) {
                        SLC_DCOPY(&m1, &b[j], &ldb, &dwork[dwork_offset], &ione);
                        SLC_DGEMV("T", &m1, &m1, &one, dwork, &ldwm1, &dwork[dwork_offset], &ione, &zero, &b[j], &ldb);
                    }
                }
            } else if (m1 <= (*n) * n2) {
                // Use BLAS 2 code
                for (i32 j = 0; j < n1; j++) {
                    SLC_DCOPY(&m1, &b1[j], &ldb1, &a[n1 * lda], &ione);
                    SLC_DGEMV("T", &m1, &m1, &one, dwork, &ldwm1, &a[n1 * lda], &ione, &zero, &b[j], &ldb);
                }
            } else {
                // Use additional workspace
                i32 dwork_offset = m1 * m1;
                for (i32 j = 0; j < n1; j++) {
                    SLC_DCOPY(&m1, &b1[j], &ldb1, &dwork[dwork_offset], &ione);
                    SLC_DGEMV("T", &m1, &m1, &one, dwork, &ldwm1, &dwork[dwork_offset], &ione, &zero, &b[j], &ldb);
                }
            }
        } else {
            SLC_DGEMM("N", "N", &n1, &m1, &m1, &one, b1, &ldb1, dwork, &ldwm1, &zero, b, &ldb);
        }
    }

    if (n2 > 0) {
        // Complete matrices B and C
        if (p1 > 0) {
            f64* b_n1 = &b[n1];
            f64* c_n1 = &c[n1 * ldc];
            SLC_DGEMM("N", "N", &n2, &m1, &p1, &one, b2, &ldb2, d, &ldd, &zero, b_n1, &ldb);
            f64 neg_alpha = -alpha;
            SLC_DGEMM("N", "N", &p1, &n2, &m1, &neg_alpha, d, &ldd, c2, &ldc2, &zero, c_n1, &ldc);
        } else if (m1 > 0) {
            f64* b_n1 = &b[n1];
            SLC_DLASET("F", &n2, &m1, &zero, &zero, b_n1, &ldb);
        }
    }

    if ((n1 > 0) && (p1 > 0)) {
        // Form upper left quadrant of A
        f64 neg_alpha = -alpha;
        SLC_DGEMM("N", "N", &n1, &p1, &m1, &neg_alpha, b, &ldb, d2, &ldd2, &zero, dwork, &n1);

        if (over == 1) {
            // Saved C1 is at offset temp_size in dwork
            // Temp area size: max(p1*p1, m1*m1, n1*p1)
            i32 temp_size = (p1 * p1 > m1 * m1) ? p1 * p1 : m1 * m1;
            if (n1 * p1 > temp_size) temp_size = n1 * p1;
            i32 ldw_offset = temp_size;
            SLC_DGEMM("N", "N", &n1, &n1, &p1, &one, dwork, &n1, &dwork[ldw_offset], &p1, &one, a, &lda);
        } else {
            SLC_DGEMM("N", "N", &n1, &n1, &p1, &one, dwork, &n1, c1, &ldc1, &one, a, &lda);
        }
    }

    if (n2 > 0) {
        // Form lower right quadrant of A
        f64* a_n1n1 = &a[n1 + n1 * lda];
        SLC_DLACPY("F", &n2, &n2, a2, &lda2, a_n1n1, &lda);
        if (m1 > 0) {
            f64 neg_alpha = -alpha;
            f64* b_n1 = &b[n1];
            SLC_DGEMM("N", "N", &n2, &n2, &m1, &neg_alpha, b_n1, &ldb, c2, &ldc2, &one, a_n1n1, &lda);
        }

        // Complete the matrix A
        f64* a_n1_0 = &a[n1];
        f64* a_0_n1 = &a[n1 * lda];
        SLC_DGEMM("N", "N", &n2, &n1, &p1, &one, b2, &ldb2, c, &ldc, &zero, a_n1_0, &lda);
        f64 neg_alpha = -alpha;
        SLC_DGEMM("N", "N", &n1, &n2, &m1, &neg_alpha, b, &ldb, c2, &ldc2, &zero, a_0_n1, &lda);
    }
}
