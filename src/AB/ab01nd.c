#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ab01nd(
    const i32 jobz,
    const i32 n,
    const i32 m,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    i32* ncont,
    i32* indcon,
    i32* nblk,
    f64* z,
    const i32 ldz,
    f64* tau,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    // Common integer and double constants for BLAS/LAPACK calls
    i32 int1 = 1, int0 = 0;
    f64 dbl1 = 1.0, dbl0 = 0.0;

    // Local scalars
    i32 iqr, itau, j, mcrt, nbl, ncrt, ni, nj, rank = 0, wrkopt;
    f64 anorm, bnorm, fnrm, toldef;
    f64 sval[3];

    // Initialize
    *info = 0;

    // Test the input scalar arguments
    if ((jobz != 0) && (jobz != 1) && (jobz != 2)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (lda < MAX(1, n)) {
        *info = -5;
    } else if (ldb < MAX(1, n)) {
        *info = -7;
    } else if ((ldz < 1) || ((jobz > 0) && (ldz < n))) {
        *info = -12;
    } else if (ldwork < MAX(MAX(1, n), 3*m)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    *ncont = 0;
    *indcon = 0;

    // Quick return if possible
    if (MIN(n, m) == 0) {
        if (n > 0) {
            if (jobz == 2) {
                SLC_DLASET("F", &n, &n, &dbl0, &dbl1, z, &ldz);
            } else if (jobz == 1) {
                SLC_DLASET("F", &n, &n, &dbl0, &dbl0, z, &ldz);
                SLC_DLASET("F", &n, &int1, &dbl0, &dbl0, tau, &n);
            }
        }
        dwork[0] = dbl1;
        return;
    }

    // Calculate the absolute norms of A and B (used for scaling)
    anorm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    bnorm = SLC_DLANGE("M", &n, &m, b, &ldb, dwork);

    // Return if matrix B is dbl0
    if (bnorm == dbl0) {
        if (jobz == 2) {
            SLC_DLASET("F", &n, &n, &dbl0, &dbl1, z, &ldz);
        } else if (jobz == 1) {
            SLC_DLASET("F", &n, &n, &dbl0, &dbl0, z, &ldz);
            SLC_DLASET("F", &n, &int1, &dbl0, &dbl0, tau, &n);
        }
        dwork[0] = dbl1;
        return;
    }

    // Scale (if needed) the matrices A and B
    mb01pd(1, 0, n, n, int0, int0, anorm, int0, nblk, a, lda, info);
    mb01pd(1, 0, n, m, int0, int0, bnorm, int0, nblk, b, ldb, info);

    // Compute the Frobenius norm of [ B  A ] (used for rank estimation)
    fnrm = SLC_DLANGE("F", &n, &m, b, &ldb, dwork);

    toldef = tol;
    if (toldef <= dbl0) {
        // Use the default tolerance in controllability determination
        f64 eps = DBL_EPSILON;
        toldef = ((f64)(n * n)) * eps;
    }

    wrkopt = 1;
    ni = 0;
    itau = 0;
    ncrt = n;
    mcrt = m;
    iqr = 0;

    // Main loop
    while (1) {
        //    Rank-revealing QR decomposition with column pivoting.
        //    The calculation is performed in NCRT rows of B starting from
        //    the row IQR (initialized to 0 and then set to rank(B)).
        //    Workspace: 3*MCRT.

        mb03oy(ncrt, mcrt, &b[iqr], ldb, toldef, fnrm, &rank, sval, iwork, &tau[itau], dwork, info);

        if (rank != 0) {
            nj = ni;
            ni = *ncont;
            *ncont = *ncont + rank;
            nblk[*indcon] = rank;
            *indcon += 1;

            // Premultiply and postmultiply the appropriate block row and block column of A by Q' and Q
            SLC_DORMQR("L", "T", &ncrt, &ncrt, &rank, &b[iqr], &ldb, &tau[itau], &a[ni + ni*lda], &lda, dwork, &ldwork, info);
            wrkopt = MAX(wrkopt, (i32)dwork[0]);
            SLC_DORMQR("R", "N", &n, &ncrt, &rank, &b[iqr], &ldb, &tau[itau], &a[ni*lda], &lda, dwork, &ldwork, info);
            wrkopt = MAX(wrkopt, (i32)dwork[0]);

            // If required, save transformations
            // Fortran: Z(NI+2, ITAU) from B(IQR+1, 1) -> C: z[(ni+1) + itau*ldz] from b[iqr + 0*ldb]
            if ((jobz > 0) && (ncrt > 1)) {
                i32 ncrtm1 = ncrt - 1;
                i32 min_rank = MIN(rank, ncrt - 1);
                SLC_DLACPY("L", &ncrtm1, &min_rank, &b[iqr+1], &ldb, &z[(ni + 1) + itau*ldz], &ldz);
            }

            // Zero the subdiagonal elements of the current matrix
            if (rank > 1) {
                i32 rank_m1 = rank - 1;
                SLC_DLASET("L", &rank_m1, &rank_m1, &dbl0, &dbl0, &b[iqr + 1], &ldb);
            }

            // Backward permutation of the columns of B or A
            if (*indcon == 1) {
                for (i32 i = 0; i < m; i++) { iwork[i] += 1; }
                SLC_DLAPMT(&int0, &rank, &m, &b[iqr+1], &ldb, iwork);
                for (i32 i = 0; i < m; i++) { iwork[i] -= 1; }
                iqr = rank;
                fnrm = SLC_DLANGE("F", &n, &n, a, &lda, dwork);
            } else {
                // iwork is 0-indexed from mb03oy; copy from B to A with column permutation
                // Fortran: A(NI+1, NJ+IWORK(J)) -> C: a[ni + (nj + iwork[j])*lda]
                for (j = 0; j < mcrt; j++) {
                    SLC_DCOPY(&rank, &b[iqr + j*ldb], &int1, &a[ni + (nj + iwork[j])*lda], &int1);
                }
            }

            itau += rank;
            if (rank != ncrt) {
                mcrt = rank;
                ncrt -= rank;
                SLC_DLACPY("G", &ncrt, &mcrt, &a[*ncont + ni*lda], &lda, &b[iqr], &ldb);
                SLC_DLASET("G", &ncrt, &mcrt, &dbl0, &dbl0, &a[*ncont + ni*lda], &lda);
                continue;  // GO TO 10
            }
        }
        break;  // Exit main loop
    }

    // If required, accumulate transformations
    if (jobz == 2) {
        i32 max_itau = MAX(1, itau - 1);
        SLC_DORGQR(&n, &n, &max_itau, z, &ldz, tau, dwork, &ldwork, info);
        wrkopt = MAX(wrkopt, (i32)dwork[0]);
    }

    // Annihilate the trailing blocks of B
    if (n - 1 > iqr) {
        i32 n_trail = n - iqr;
        SLC_DLASET("G", &n_trail, &m, &dbl0, &dbl0, &b[iqr], &ldb);
    }

    // Annihilate the trailing elements of TAU, if JOBZ = 'F'
    if (jobz == 1) {
        for (j = itau; j < n; j++) {
            tau[j - 1] = dbl0;
        }
    }

    // Undo scaling of A and B
    if (*indcon < n) {
        nbl = *indcon + 1;
        nblk[nbl - 1] = n - *ncont;
    } else {
        nbl = 0;
    }
    mb01pd(0, 3, n, n, 0, 0, anorm, nbl, nblk, a, lda, info);
    mb01pd(0, 0, nblk[0], m, 0, 0, bnorm, 0, nblk, b, ldb, info);

    // Set optimal workspace dimension
    dwork[0] = (f64)wrkopt;
    return;
}
