#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01vd(
    const i32 trana,  // 0: no transpose, 1: transpose
    const i32 tranb, // 0: no transpose, 1: transpose
    const i32 ma,
    const i32 na,
    const i32 mb,
    const i32 nb,
    const f64 alpha,
    const f64 beta,
    const f64 *a,
    const i32 lda,
    const f64 *b,
    const i32 ldb,
    f64 *c,
    const i32 ldc,
    i32 *mc,
    i32 *nc,
    i32 *info
)
{
    const f64 dbl0 = 0.0;
    const f64 dbl1 = 1.0;
    const f64 sparst = 0.8;
    const i32 int0 = 0;
    const i32 int1 = 1;

    i32 i, ic, j, jc, k, l, lc, nz;
    i32 sparse;
    f64 aij;

    *mc = ma * mb;
    *nc = na * nb;
    *info = 0;

    if (trana < 0 || trana > 1) {
        *info = -1;
    } else if (tranb < 0 || tranb > 1) {
        *info = -2;
    } else if (ma < 0) {
        *info = -3;
    } else if (na < 0) {
        *info = -4;
    } else if (mb < 0) {
        *info = -5;
    } else if (nb < 0) {
        *info = -6;
    } else if ((trana == 1 && lda < na) || lda < 1 ||
               (trana == 0 && lda < ma)) {
        *info = -10;
    } else if ((tranb == 1 && ldb < nb) || ldb < 1 ||
               (tranb == 0 && ldb < mb)) {
        *info = -12;
    } else if (ldc < MAX(1, *mc)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (*mc == 0 || *nc == 0) {
        return;
    }

    if (alpha == dbl0) {
        if (beta == dbl0) {
            dlaset("Full", mc, nc, &dbl0, &dbl0, c, &ldc);
        } else if (beta != dbl1) {
            for (j = 0; j < *nc; j++) {
                dscal(mc, &beta, &c[j*ldc], &int1);
            }
        }
        return;
    }

    jc = 0;
    nz = 0;

    for (j = 0; j < na; j++) {
        for (i = 0; i < ma; i++) {
            if (trana == 1) {
                if (a[j + i*lda] == dbl0) {
                    nz = nz + 1;
                }
            } else {
                if (a[i + j*lda] == dbl0) {
                    nz = nz + 1;
                }
            }
        }
    }

    sparse = ((f64)nz / (f64)(ma * na)) >= sparst;

    if (trana == 0 && tranb == 0) {

        if (beta == dbl0) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else if (aij == dbl1) {
                                    dcopy(&mb, &b[k*ldb], &int1, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else if (beta == dbl1) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        }

    } else if (trana == 1 && tranb == 0) {

        if (beta == dbl0) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else if (aij == dbl1) {
                                    dcopy(&mb, &b[k*ldb], &int1, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else if (beta == dbl1) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[l + k*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        }

    } else if (trana == 0 && tranb == 1) {

        if (beta == dbl0) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else if (aij == dbl1) {
                                    dcopy(&mb, &b[k], &ldb, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else if (beta == dbl1) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[i + j*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        }

    } else {

        if (beta == dbl0) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else if (aij == dbl1) {
                                    dcopy(&mb, &b[k], &ldb, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                if (aij == dbl0) {
                                    dcopy(&mb, &dbl0, &int0, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else if (beta == dbl1) {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                if (aij != dbl0) {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        } else {
            if (alpha == dbl1) {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            } else {
                if (sparse) {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                if (aij == dbl0) {
                                    dscal(&mb, &beta, &c[ic + jc*ldc], &int1);
                                } else {
                                    lc = ic;
                                    for (l = 0; l < mb; l++) {
                                        c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                        lc = lc + 1;
                                    }
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                } else {

                    for (j = 0; j < na; j++) {
                        for (k = 0; k < nb; k++) {
                            ic = 0;
                            for (i = 0; i < ma; i++) {
                                aij = alpha * a[j + i*lda];
                                lc = ic;
                                for (l = 0; l < mb; l++) {
                                    c[lc + jc*ldc] = beta * c[lc + jc*ldc] + aij * b[k + l*ldb];
                                    lc = lc + 1;
                                }
                                ic = ic + mb;
                            }
                            jc = jc + 1;
                        }
                    }

                }
            }
        }
    }

    return;
}
