#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void mb01rx(const i32 side, const i32 uplo, const i32 trans, const i32 m, const i32 n,
            const f64 alpha, const f64 beta, f64* r, const i32 ldr,
            const f64* a, const i32 lda, const f64* b, const i32 ldb, i32* info);

void
mb01rb(
    const i32 side,  // 0: left, 1: right
    const i32 uplo,  // 0: upper, 1: lower
    const i32 trans, // 0: no transpose, 1: transpose
    const i32 m,
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    i32* info
)
{
    const i32 n1l = 128, n1p = 512, n2l = 40, n2p = 128, nbs = 48;
    i32 i, ib, j, jb, mn, mx, n1, n2, nb, nbmin, nx;
    f64 d[1];
    f64 dbl1 = 1.0, dbl0 = 0.0;
    i32 int1 = 1, int0 = 0, intm1 = -1;

    *info = 0;
    if (side != 0 && side != 1) {
        *info = -1;
    } else if (uplo != 0 && uplo != 1) {
        *info = -2;
    } else if (trans != 0 && trans != 1) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldr < MAX(1, m)) {
        *info = -9;
    } else if (lda < 1 ||
               (((side == 0 && trans == 0) || (side == 1 && trans == 1)) && lda < m) ||
               (((side == 0 && trans == 1) || (side == 1 && trans == 0)) && lda < n)) {
        *info = -11;
    } else if (ldb < 1 ||
               (side == 0 && ldb < n) ||
               (side == 1 && ldb < m)) {
        *info = -13;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible.
    if (m == 0) {
        return;
    }

    if (beta == 0.0 || n == 0) {
        if (alpha == 0.0) {
            // Special case alpha = 0.
            SLC_DLASET((uplo ? "L" : "U"), &m, &m, &dbl0, &dbl0, r, &ldr);
        } else {
            // Special case beta = 0 or N = 0.
            if (alpha != 1.0) {
                SLC_DLASCL((uplo ? "L" : "U"), &int0, &int0, &dbl1, &alpha, &m, &m, r, &ldr, info);
            }
        }
        return;
    }

    // General case: beta <> 0.
    // Compute the required triangle of (1) or (2) using essentially BLAS 3 operations.
    // Find the block size using DGEQRF.
    mx = MAX(m, n);
    mn = MIN(m, n);
    SLC_DGEQRF(&mx, &mn, a, &mx, d, d, &intm1, info);
    nb = ((i32)d[0])/mn/8*8;

    if (nb > 1 && nb < m) {
        // Determine when to cross over from blocked to unblocked code.
        nx = MAX(0, SLC_ILAENV(&(i32){3}, "DGEQRF", " ", &mx, &mn, &intm1, &intm1));
        if (nx < m) {
            // Determine the minimum value of NB.
            nbmin = MAX(2, SLC_ILAENV(&(i32){2}, "DGEQRF", " ", &mx, &mn, &intm1, &intm1));
        }
    } else {
        nx = 0;
        nbmin = 2;
    }

    if (nb >= nbmin && nb < m && nx < m) {
        // Use blocked code initially.
        if (trans && side == 0) {
            if (nb <= nbs) {
                n1 = MIN(MAX(n1l, nb), n);
            } else {
                n1 = MIN(n1p, n);
            }
        } else {
            if (nb <= nbs) {
                n2 = MIN(MAX(n2l, nb), n);
            } else {
                n2 = MIN(n2p, n);
            }
        }

        for (i = 0; i < m - nx; i += nb) {
            ib = i + nb;
            jb = MIN(m - i, nb);

            // Compute the current diagonal block and the needed off-diagonal part
            // of the current block row, if UPLO = 'U', or block column, if UPLO = 'L'.
            if (trans) {
                if (side == 0) {
                    mb01rx(side, uplo, trans, jb, n1, alpha, beta,
                           &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i*ldb], ldb, info);
                    for (j = n1; j < n; j += n1) {
                        mb01rx(side, uplo, trans, jb, MIN(n1, n - j),
                               dbl1, beta, &r[i + i*ldr], ldr, &a[j + i*lda], lda,
                               &b[j + i*ldb], ldb, info);
                    }
                    if (ib <= m) {
                        if (!uplo) {
                            SLC_DGEMM((trans ? "T" : "N"), "N", &jb, &(i32){m - ib + 1},
                                      &n, &beta, &a[i*lda], &lda, &b[ib*ldb], &ldb,
                                      &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM((trans ? "T" : "N"), "N", &(i32){m - ib + 1}, &jb,
                                      &n, &beta, &a[ib*lda], &lda, &b[i*ldb], &ldb,
                                      &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                } else {
                    mb01rx(side, uplo, trans, jb, n2, alpha, beta,
                           &r[i + i*ldr], ldr, &a[i], lda, &b[i], ldb, info);
                    for (j = n2; j < n; j += n2) {
                        mb01rx(side, uplo, trans, jb, MIN(n2, n - j),
                               dbl1, beta, &r[i + i*ldr], ldr, &a[i + j*lda], lda,
                               &b[i + j*ldb], ldb, info);
                    }
                    if (ib <= m) {
                        if (!uplo) {
                            SLC_DGEMM("N", (trans ? "T" : "N"), &jb, &(i32){m - ib + 1},
                                      &n, &beta, &b[i], &ldb, &a[ib], &lda,
                                      &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM("N", (trans ? "T" : "N"), &(i32){m - ib + 1}, &jb,
                                      &n, &beta, &b[ib], &ldb, &a[i], &lda,
                                      &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                }
            } else {
                if (side == 0) {
                    mb01rx(side, uplo, trans, jb, n2, alpha, beta,
                           &r[i + i*ldr], ldr, &a[i], lda, &b[i*ldb], ldb, info);
                    for (j = n2; j < n; j += n2) {
                        mb01rx(side, uplo, trans, jb, MIN(n2, n - j),
                               dbl1, beta, &r[i + i*ldr], ldr, &a[i + j*lda], lda,
                               &b[j + i*ldb], ldb, info);
                    }
                    if (ib <= m) {
                        if (!uplo) {
                            SLC_DGEMM((trans ? "T" : "N"), "N", &jb, &(i32){m - ib + 1},
                                      &n, &beta, &a[i], &lda, &b[ib*ldb], &ldb,
                                      &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM((trans ? "T" : "N"), "N", &(i32){m - ib + 1}, &jb,
                                      &n, &beta, &a[ib], &lda, &b[i*ldb], &ldb,
                                      &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                } else {
                    mb01rx(side, uplo, trans, jb, n2, alpha, beta,
                           &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i], ldb, info);
                    for (j = n2; j < n; j += n2) {
                        mb01rx(side, uplo, trans, jb, MIN(n2, n - j),
                               dbl1, beta, &r[i + i*ldr], ldr, &a[j + i*lda], lda,
                               &b[i + j*ldb], ldb, info);
                    }
                    if (ib <= m) {
                        if (!uplo) {
                            SLC_DGEMM("N", (trans ? "T" : "N"), &jb, &(i32){m - ib + 1},
                                      &n, &beta, &b[i], &ldb, &a[ib*lda], &lda,
                                      &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM("N", (trans ? "T" : "N"), &(i32){m - ib + 1}, &jb,
                                      &n, &beta, &b[ib], &ldb, &a[i*lda], &lda,
                                      &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                }
            }
        }
    } else {
        i = 0;
        n1 = n;
        n2 = n;
    }

    // Use unblocked code to compute the last or only block.
    if (i < m) {
        if (trans) {
            if (side == 0) {
                mb01rx(side, uplo, trans, m - i, n1, alpha, beta,
                       &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i*ldb], ldb, info);
                for (j = n1; j < n; j += n1) {
                    mb01rx(side, uplo, trans, m - i, MIN(n1, n - j),
                           dbl1, beta, &r[i + i*ldr], ldr, &a[j + i*lda], lda,
                           &b[j + i*ldb], ldb, info);
                }
            } else {
                mb01rx(side, uplo, trans, m - i, n2, alpha, beta,
                       &r[i + i*ldr], ldr, &a[i], lda, &b[i], ldb, info);
                for (j = n2; j < n; j += n2) {
                    mb01rx(side, uplo, trans, m - i, MIN(n2, n - j),
                           dbl1, beta, &r[i + i*ldr], ldr, &a[i + j*lda], lda,
                           &b[i + j*ldb], ldb, info);
                }
            }
        } else {
            if (side == 0) {
                mb01rx(side, uplo, trans, m - i, n2, alpha, beta,
                       &r[i + i*ldr], ldr, &a[i], lda, &b[i*ldb], ldb, info);
                for (j = n2; j < n; j += n2) {
                    mb01rx(side, uplo, trans, m - i, MIN(n2, n - j),
                           dbl1, beta, &r[i + i*ldr], ldr, &a[i + j*lda], lda,
                           &b[j + i*ldb], ldb, info);
                }
            } else {
                mb01rx(side, uplo, trans, m - i, n2, alpha, beta,
                       &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i], ldb, info);
                for (j = n2; j < n; j += n2) {
                    mb01rx(side, uplo, trans, m - i, MIN(n2, n - j),
                           dbl1, beta, &r[i + i*ldr], ldr, &a[j + i*lda], lda,
                           &b[i + j*ldb], ldb, info);
                }
            }
        }
    }

    return;
}
