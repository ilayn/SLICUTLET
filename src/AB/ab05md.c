#include "slc_blaslapack.h"
#include "types.h"
#include <string.h>

void ab05md(
    const i32 uplo,
    const i32 over,
    const i32 n1,
    const i32 m1,
    const i32 p1,
    const i32 n2,
    const i32 p2,
    const f64* a1,
    const i32 lda1,
    const f64* b1,
    const i32 ldb1,
    const f64* c1,
    const i32 ldc1,
    const f64* d1,
    const i32 ldd1,
    const f64* a2,
    const i32 lda2,
    const f64* b2,
    const i32 ldb2,
    const f64* c2,
    const i32 ldc2,
    const f64* d2,
    const i32 ldd2,
    i32* n,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* dwork,
    const i32 ldwork,
    i32* info
) {

    *info = 0;
    *n = n1 + n2;

    if ((uplo != 0) && (uplo != 1)) {
        *info = -1;
    } else if ((over != 0) && (over != 1)) {
        *info = -2;
    } else if (n1 < 0) {
        *info = -3;
    } else if (m1 < 0) {
        *info = -4;
    } else if (p1 < 0) {
        *info = -5;
    } else if (n2 < 0) {
        *info = -6;
    } else if (p2 < 0) {
        *info = -7;
    } else if (lda1 < MAX(1, n1)) {
        *info = -9;
    } else if (ldb1 < MAX(1, n1)) {
        *info = -11;
    } else if (((n1 > 0) && (ldc1 < MAX(1, p1))) || ((n1 == 0) && (ldc1 < 1))) {
        *info = -13;
    } else if (ldd1 < MAX(1, p1)) {
        *info = -15;
    } else if (lda2 < MAX(1, n2)) {
        *info = -17;
    } else if (ldb2 < MAX(1, n2)) {
        *info = -19;
    } else if (((n2 > 0) && (ldc2 < MAX(1, p2))) || ((n2 == 0) && (ldc2 < 1))) {
        *info = -21;
    } else if (ldd2 < MAX(1, p2)) {
        *info = -23;
    } else if (lda < MAX(1, *n)) {
        *info = -26;
    } else if (ldb < MAX(1, *n)) {
        *info = -28;
    } else if (((*n > 0) && (ldc < MAX(1, p2))) || ((*n == 0) && (ldc < 1))) {
        *info = -30;
    } else if (ldd < MAX(1, p2)) {
        *info = -32;
    } else if (((over == 1) && (ldwork < MAX(1, p1 * MAX(MAX(n1, m1), MAX(n2, p2))))) || ((over == 0) && (ldwork < 1))) {
        *info = -34;
    }

    if (*info != 0) {
        return;
    }

    if (MAX(*n, MIN(m1, p2)) == 0) {
        return;
    }

    i32 i1, i2;
    if (uplo == 0) {
        i1 = 0;
        i2 = MIN(n1, *n - 1);
    } else {
        i1 = MIN(n2, *n - 1);
        i2 = 0;
    }

    i32 ldwn2 = MAX(1, n2);
    i32 ldwp1 = MAX(1, p1);
    i32 ldwp2 = MAX(1, p2);

    f64 zero = 0.0, one = 1.0;

    if (uplo == 0) {
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
        if (n2 > 0) {
            SLC_DLACPY("F", &n2, &n2, a2, &lda2, &a[i2 + i2 * lda], &lda);
        }
    } else {
        if ((over == 1) && (lda2 <= lda)) {
            if (lda2 < lda) {
                for (i32 j = n2 - 1; j >= 0; j--) {
                    for (i32 i = n2 - 1; i >= 0; i--) {
                        a[i + j * lda] = a2[i + j * lda2];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &n2, &n2, a2, &lda2, a, &lda);
        }
        if (n1 > 0) {
            SLC_DLACPY("F", &n1, &n1, a1, &lda1, &a[i1 + i1 * lda], &lda);
        }
    }

    if (MIN(n1, n2) > 0) {
        SLC_DLASET("F", &n1, &n2, &zero, &zero, &a[i1 + i2 * lda], &lda);
        SLC_DGEMM("N", "N", &n2, &n1, &p1, &one, b2, &ldb2, c1, &ldc1, &zero, &a[i2 + i1 * lda], &lda);
    }

    if (uplo == 0) {
        if ((over == 1) && (ldb1 <= ldb)) {
            if (ldb1 < ldb) {
                for (i32 j = m1 - 1; j >= 0; j--) {
                    for (i32 i = n1 - 1; i >= 0; i--) {
                        b[i + j * ldb] = b1[i + j * ldb1];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &n1, &m1, b1, &ldb1, b, &ldb);
        }

        if (MIN(n2, m1) > 0) {
            SLC_DGEMM("N", "N", &n2, &m1, &p1, &one, b2, &ldb2, d1, &ldd1, &zero, &b[i2 + 0 * ldb], &ldb);
        }

        if (n1 > 0) {
            if (over == 1) {
                SLC_DLACPY("F", &p1, &n1, c1, &ldc1, dwork, &ldwp1);
                SLC_DGEMM("N", "N", &p2, &n1, &p1, &one, d2, &ldd2, dwork, &ldwp1, &zero, c, &ldc);
            } else {
                SLC_DGEMM("N", "N", &p2, &n1, &p1, &one, d2, &ldd2, c1, &ldc1, &zero, c, &ldc);
            }
        }

        if (MIN(p2, n2) > 0) {
            SLC_DLACPY("F", &p2, &n2, c2, &ldc2, &c[0 + i2 * ldc], &ldc);
        }

        if (over == 1) {
            SLC_DLACPY("F", &p1, &m1, d1, &ldd1, dwork, &ldwp1);
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &one, d2, &ldd2, dwork, &ldwp1, &zero, d, &ldd);
        } else {
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &one, d2, &ldd2, d1, &ldd1, &zero, d, &ldd);
        }
    } else {
        if (over == 1) {
            SLC_DLACPY("F", &n2, &p1, b2, &ldb2, dwork, &ldwn2);
            if (MIN(n2, m1) > 0) {
                SLC_DGEMM("N", "N", &n2, &m1, &p1, &one, dwork, &ldwn2, d1, &ldd1, &zero, &b[i2 + 0 * ldb], &ldb);
            }
        } else {
            SLC_DGEMM("N", "N", &n2, &m1, &p1, &one, b2, &ldb2, d1, &ldd1, &zero, b, &ldb);
        }

        if (MIN(n1, m1) > 0) {
            SLC_DLACPY("F", &n1, &m1, b1, &ldb1, &b[i1 + 0 * ldb], &ldb);
        }

        if ((over == 1) && (ldc2 <= ldc)) {
            if (ldc2 < ldc) {
                for (i32 j = n2 - 1; j >= 0; j--) {
                    for (i32 i = p2 - 1; i >= 0; i--) {
                        c[i + j * ldc] = c2[i + j * ldc2];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &p2, &n2, c2, &ldc2, c, &ldc);
        }

        if (MIN(p2, n1) > 0) {
            SLC_DGEMM("N", "N", &p2, &n1, &p1, &one, d2, &ldd2, c1, &ldc1, &zero, &c[0 + i1 * ldc], &ldc);
        }

        if (over == 1) {
            SLC_DLACPY("F", &p2, &p1, d2, &ldd2, dwork, &ldwp2);
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &one, dwork, &ldwp2, d1, &ldd1, &zero, d, &ldd);
        } else {
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &one, d2, &ldd2, d1, &ldd1, &zero, d, &ldd);
        }
    }
}
