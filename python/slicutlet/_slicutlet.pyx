# cython: language_level=3
# distutils: language=c
"""
Cython wrapper for SLICUTLET C library

This module provides thin wrappers around the C functions, handling
array conversions and output packaging.
"""

cimport numpy as cnp
import numpy as np

cnp.import_array()

# Import C function declarations
cdef extern from "SLICUTLET/slicutlet.h":

    void mc01td(
        const int dico,
        int* dp,
        double* p,
        int* stable,
        int* nz,
        double* dwork,
        int* iwarn,
        int* info
    ) nogil

    void ab07nd(
        const int n,
        const int m,
        double* a,
        const int lda,
        double* b,
        const int ldb,
        double* c,
        const int ldc,
        double* d,
        const int ldd,
        double* rcond,
        int* iwork,
        double* dwork,
        const int ldwork,
        int* info
    ) nogil

    void mb03oy(
        const int m,
        const int n,
        double* a,
        const int lda,
        const double rcond,
        const double svlmax,
        int* rank,
        double* sval,
        int* jpvt,
        double* tau,
        double* dwork,
        int* info
    ) nogil


# Python wrappers
def py_mc01td(dico, p):
    """
    Check stability of a real polynomial.

    Determines whether a polynomial with real coefficients is stable,
    either in continuous-time (roots in left half-plane) or discrete-time
    (roots inside unit circle).

    Parameters
    ----------
    dico : {'C', 'D', 0, 1}
        Indicates continuous-time ('C' or 1) or discrete-time ('D' or 0).
    p : array_like, shape (n+1,)
        Polynomial coefficients in increasing powers: p[0] + p[1]*x + ... + p[n]*x^n

    Returns
    -------
    stable : bool
        True if polynomial is stable, False otherwise.
    nz : int
        Number of unstable zeros (right half-plane or outside unit circle).
    dp : int
        Actual degree after trimming trailing zeros.
    iwarn : int
        Warning indicator. Number of trailing zeros trimmed.
    info : int
        0 on success, >0 for warnings/errors:
        1: zero polynomial
        2: inconclusive (zeros near boundary)

    Examples
    --------
    >>> # Stable continuous-time: (s+1)(s+2) = s^2 + 3s + 2
    >>> p = [2.0, 3.0, 1.0]
    >>> stable, nz, dp, iwarn, info = mc01td('C', p)
    >>> stable
    True
    >>> nz
    0
    """
    # Parse dico argument
    cdef int dico_int
    if isinstance(dico, str):
        if dico.upper() == 'C':
            dico_int = 1
        elif dico.upper() == 'D':
            dico_int = 0
        else:
            raise ValueError(f"dico must be 'C' or 'D', got {dico}")
    else:
        dico_int = int(dico)
        if dico_int not in (0, 1):
            raise ValueError(f"dico must be 0 or 1, got {dico_int}")

    # Convert input to contiguous array
    cdef cnp.ndarray[double, ndim=1, mode='c'] p_arr = \
        np.ascontiguousarray(p, dtype=np.float64)

    if p_arr.shape[0] == 0:
        raise ValueError("Polynomial must have at least one coefficient")

    cdef int dp = p_arr.shape[0] - 1
    cdef int stable, nz, iwarn, info
    cdef cnp.ndarray[double, ndim=1, mode='c'] dwork = \
        np.zeros(2*dp + 2, dtype=np.float64)

    # Make a copy since mc01td may modify the polynomial
    cdef cnp.ndarray[double, ndim=1, mode='c'] p_copy = p_arr.copy()

    mc01td(dico_int, &dp, &p_copy[0], &stable, &nz, &dwork[0], &iwarn, &info)

    return bool(stable), int(nz), int(dp), int(iwarn), int(info)


def py_ab07nd(n, m, A, B, C, D):
    """
    Elementary operations on a 2x2 partitioned matrix [A B; C D].

    Computes the inverse of D and the transformed system matrices:
    A_new = A - B*D^{-1}*C,  B_new = -B*D^{-1},  C_new = D^{-1}*C

    Parameters
    ----------
    n : int
        Order of square matrix A.
    m : int
        Number of columns of B and rows of C.
    A : array_like, shape (n, n)
        Input matrix A (will be modified to A_new).
    B : array_like, shape (n, m)
        Input matrix B (will be modified to B_new).
    C : array_like, shape (m, n)
        Input matrix C (will be modified to C_new).
    D : array_like, shape (m, m)
        Input matrix D (will be modified to D^{-1}).

    Returns
    -------
    A : ndarray, shape (n, n)
        Transformed A matrix.
    B : ndarray, shape (n, m)
        Transformed B matrix.
    C : ndarray, shape (m, n)
        Transformed C matrix.
    D : ndarray, shape (m, m)
        Inverse of input D.
    rcond : float
        Reciprocal condition number of D.
    info : int
        0 on success, m+1 if D is numerically singular.
    """
    cdef int n_int = int(n)
    cdef int m_int = int(m)

    if n_int < 0:
        raise ValueError(f"n must be >= 0, got {n_int}")
    if m_int < 0:
        raise ValueError(f"m must be >= 0, got {m_int}")

    # Convert and copy inputs (function modifies them in-place)
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] A_arr = \
        np.asfortranarray(A, dtype=np.float64).copy()
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] B_arr = \
        np.asfortranarray(B, dtype=np.float64).copy()
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] C_arr = \
        np.asfortranarray(C, dtype=np.float64).copy()
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] D_arr = \
        np.asfortranarray(D, dtype=np.float64).copy()

    # Validate shapes
    if A_arr.shape[0] != n_int or A_arr.shape[1] != n_int:
        raise ValueError(f"A must be ({n_int}, {n_int}), got ({A_arr.shape[0]}, {A_arr.shape[1]})")
    if B_arr.shape[0] != n_int or B_arr.shape[1] != m_int:
        raise ValueError(f"B must be ({n_int}, {m_int}), got ({B_arr.shape[0]}, {B_arr.shape[1]})")
    if C_arr.shape[0] != m_int or C_arr.shape[1] != n_int:
        raise ValueError(f"C must be ({m_int}, {n_int}), got ({C_arr.shape[0]}, {C_arr.shape[1]})")
    if D_arr.shape[0] != m_int or D_arr.shape[1] != m_int:
        raise ValueError(f"D must be ({m_int}, {m_int}), got ({D_arr.shape[0]}, {D_arr.shape[1]})")

    cdef int lda = max(1, n_int)
    cdef int ldb = max(1, n_int)
    cdef int ldc = max(1, m_int)
    cdef int ldd = max(1, m_int)
    cdef int ldwork = max(1, 4*m_int if m_int > 0 else 1)

    cdef double rcond
    cdef int info
    cdef cnp.ndarray[int, ndim=1, mode='c'] iwork = \
        np.zeros(m_int if m_int > 0 else 1, dtype=np.intc)
    cdef cnp.ndarray[double, ndim=1, mode='c'] dwork = \
        np.zeros(ldwork, dtype=np.float64)

    ab07nd(n_int, m_int,
           &A_arr[0, 0], lda,
           &B_arr[0, 0], ldb,
           &C_arr[0, 0], ldc,
           &D_arr[0, 0], ldd,
           &rcond,
           &iwork[0], &dwork[0], ldwork,
           &info)

    return A_arr, B_arr, C_arr, D_arr, float(rcond), int(info)


def py_mb03oy(A, rcond, svlmax):
    """
    Rank-revealing QR factorization with incremental condition estimation.

    Computes a QR factorization with column pivoting of a real m-by-n matrix A,
    determining the rank based on tolerance rcond and maximum singular value svlmax.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Matrix to be factorized (will be modified to contain R).
    rcond : float
        Tolerance for rank determination. Should be in [0, 1].
    svlmax : float
        Estimate of largest singular value. Must be >= 0.

    Returns
    -------
    A : ndarray, shape (m, n)
        Upper trapezoidal factor R and Householder vectors.
    rank : int
        Computed rank of the matrix.
    sval : ndarray, shape (3,)
        Singular value estimates: [max, min, min'].
    jpvt : ndarray, shape (n,), dtype=int
        Column permutation indices.
    tau : ndarray, shape (min(m,n),)
        Householder reflection scalars.
    info : int
        0 on success, <0 for invalid arguments.
    """
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] A_arr = \
        np.asfortranarray(A, dtype=np.float64).copy()

    cdef int m = A_arr.shape[0]
    cdef int n = A_arr.shape[1]
    cdef int lda = max(1, m)
    cdef int minmn = min(m, n)

    cdef double rcond_val = float(rcond)
    cdef double svlmax_val = float(svlmax)

    if rcond_val < 0.0 or rcond_val > 1.0:
        raise ValueError(f"rcond must be in [0, 1], got {rcond_val}")
    if svlmax_val < 0.0:
        raise ValueError(f"svlmax must be >= 0, got {svlmax_val}")

    cdef int rank, info
    cdef cnp.ndarray[double, ndim=1, mode='c'] sval = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray[int, ndim=1, mode='c'] jpvt = np.zeros(n, dtype=np.intc)
    cdef cnp.ndarray[double, ndim=1, mode='c'] tau = np.zeros(minmn, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1, mode='c'] dwork = \
        np.zeros(3*n, dtype=np.float64)

    mb03oy(m, n, &A_arr[0, 0], lda,
           rcond_val, svlmax_val,
           &rank, &sval[0], &jpvt[0], &tau[0], &dwork[0], &info)

    return A_arr, int(rank), sval, jpvt, tau, int(info)
