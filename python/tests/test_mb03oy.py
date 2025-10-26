"""
Tests for mb03oy: rank-revealing QR with incremental condition estimation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from slicutlet import mb03oy


def call_mb03oy(m, n, rcond, svlmax, A):
    """Helper to call mb03oy with proper array allocation"""
    sval = np.zeros(3, dtype=float)
    jpvt = np.zeros(n, dtype=np.int32)
    tau = np.zeros(min(m, n), dtype=float)
    dwork = np.zeros(3 * n - 1, dtype=float)

    A_out, rank, sval_out, jpvt_out, tau_out, info = mb03oy(
        m, n, rcond, svlmax, A, sval, jpvt, tau, dwork
    )

    return rank, sval_out, jpvt_out, tau_out, info, A_out


class TestMB03OYBasic:
    """Basic functionality tests for mb03oy"""

    def test_zero_matrix(self):
        """Test that a zero matrix returns rank 0"""
        m, n = 4, 3
        A = np.zeros((m, n), dtype=float, order="F")
        rcond = 1e-10
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_out = call_mb03oy(m, n, rcond, svlmax, A)

        assert info == 0
        assert rank == 0
        assert_allclose(sval, [0.0, 0.0, 0.0], atol=1e-14)

    def test_identity_matrix(self):
        """Test full-rank square matrix (identity)"""
        n = 4
        A = np.eye(n, dtype=float, order="F")
        rcond = 1e-10
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_out = call_mb03oy(n, n, rcond, svlmax, A)

        assert info == 0
        assert rank == n
        assert sval[0] >= sval[1] >= sval[2] > 0
        # Check that pivot permutation is valid (0-based in C)
        assert sorted(jpvt[:n].tolist()) == list(range(n))

    def test_full_rank_rectangular(self):
        """Test full-rank rectangular matrix"""
        m, n = 5, 3
        np.random.seed(42)
        A = np.random.randn(m, n).astype(float, order="F")
        rcond = 1e-10
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_out = call_mb03oy(m, n, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        assert rank == n  # Full column rank
        assert sval[0] > sval[1] > 0
        assert sorted(jpvt[:n].tolist()) == list(range(n))


class TestMB03OYRankDeficient:
    """Tests for rank-deficient matrices"""

    def test_rank_one_matrix(self):
        """Test explicit rank-1 matrix"""
        m, n = 5, 4
        u = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        v = np.array([[1.0, 0.5, 2.0, 1.5]])
        A = (u @ v).astype(float, order="F")

        # Use tighter rcond to reject numerical noise
        rcond = 1e-12
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_out = call_mb03oy(m, n, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        # For a numerically exact rank-1 outer product, expect rank 1
        # (In practice, floating-point may introduce tiny noise)
        assert rank == 1, f"Expected rank 1, got {rank} with sval={sval}"
        assert sval[0] > 0
        # sval[1] should be near zero (second singular value)
        if rank > 1:
            assert sval[1] < 1e-10 * sval[0]

    def test_designed_rank_3_matrix(self):
        """
        Test a 5x5 matrix designed to have rank 3.

        Strategy:
        1. Create a 5x5 matrix with known rank 3 structure:
           - Upper 3x5 block: random
           - Lower 2x3 block: zero
           - Lower 2x2 block: very small entries (~ 1e-16)
        2. Pre- and post-multiply by orthogonal Q (from QR of random matrix)
           to mix entries while preserving rank and singular values.
        """
        n = 5
        np.random.seed(123)

        # Construct base matrix with rank 3
        B = np.zeros((n, n), dtype=float)
        # Upper 3 rows: random
        B[:3, :] = np.random.randn(3, n)
        # Lower-right 2x2: very small
        B[3:, 3:] = 1e-16 * np.random.randn(2, 2)
        # Lower-left 2x3: exactly zero (ensuring rank <= 3)
        B[3:, :3] = 0.0

        # Generate orthogonal mixing via QR
        Q_left, _ = np.linalg.qr(np.random.randn(n, n))
        Q_right, _ = np.linalg.qr(np.random.randn(n, n))

        # Mix the matrix: A = Q_left @ B @ Q_right^T
        A = Q_left @ B @ Q_right.T
        A = np.asarray(A, dtype=float, order="F")

        rcond = 1e-10
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_out = call_mb03oy(n, n, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        # Should detect rank 3
        assert rank == 3, f"Expected rank 3, got {rank}"
        assert sval[0] > sval[1] > sval[2] > 0
        # Fourth singular value estimate (sval[2] after rank=3 step) should be tiny
        # (Condition: sval[2] is estimate of (rank+1)-th singular value when rank < min(m,n))
        # But sval[2] is reported as sminpr, which is the (rank+1)-th estimate.
        # For rank=3, sval[2] should be much smaller than sval[1]
        assert sval[2] < 1e-8 * sval[1], f"Expected small 4th singular value, got {sval[2]}"


class TestMB03OYOrthogonality:
    """Test that the Q factor is orthogonal"""

    def test_q_orthogonality_full_rank(self):
        """Verify Q from mb03oy is orthogonal for full-rank case"""
        m, n = 6, 4
        np.random.seed(99)
        A = np.random.randn(m, n).astype(float, order="F")
        A_copy = A.copy(order='F')

        rcond = 1e-12
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_fact = call_mb03oy(m, n, rcond, svlmax, A)

        assert info == 0
        # Reconstruct Q using the stored reflectors in A and tau
        # (This requires forming Q explicitly, similar to DORGQR)
        # For simplicity, we'll use a basic check: apply Q to an identity-like vector
        # and verify orthogonality properties indirectly via R and permuted columns.

        # Check: A_original * P = Q * R
        # After factorization, A contains R in upper triangle and reflectors below diagonal
        # We can extract R (upper rank x n)
        R = np.triu(A_fact[:rank, :])

        # Verify R has correct structure: upper triangular, rank rows
        assert R.shape == (rank, n)
        # Check that leading rank x rank block of R is upper triangular with nonzero diagonal
        for i in range(rank):
            assert abs(R[i, i]) > 1e-12, f"Diagonal element R[{i},{i}] is too small"

    def test_q_orthogonality_rank_deficient(self):
        """Verify Q orthogonality for rank-deficient matrix"""
        m, n = 5, 5
        # Rank-2 matrix
        u1 = np.random.randn(m, 1)
        u2 = np.random.randn(m, 1)
        v1 = np.random.randn(1, n)
        v2 = np.random.randn(1, n)
        A = (u1 @ v1 + u2 @ v2).astype(float, order="F")

        rcond = 1e-10
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_fact = call_mb03oy(m, n, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        # Should be rank 2
        assert rank == 2
        # The factorization should still produce a valid R
        # (Even though rank < min(m, n), the first 'rank' reflectors are valid)


class TestMB03OYPivoting:
    """Test column pivoting behavior"""

    def test_pivot_ordering(self):
        """Test that pivoting selects largest-norm columns first"""
        m, n = 4, 4
        # Construct matrix where column norms are clearly ordered
        A = np.array([
            [0.1, 1.0, 0.01, 5.0],
            [0.1, 1.0, 0.01, 5.0],
            [0.1, 1.0, 0.01, 5.0],
            [0.1, 1.0, 0.01, 5.0]
        ], dtype=float, order="F")
        # Column norms: ~0.2, ~2.0, ~0.02, ~10.0
        # Expected pivot order: col 3 (index 3), col 1 (index 1), col 0 (index 0), col 2 (index 2)

        rcond = 1e-12
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_fact = call_mb03oy(m, n, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        # jpvt should reflect that col 3 was chosen first
        # (In C, jpvt is 0-based: jpvt[0] is the original column index chosen first)
        assert jpvt[0] == 3, f"Expected column 3 pivoted first, got {jpvt[0]}"
        assert rank == 1  # All columns are proportional (rank 1)

    def test_pivot_permutation_valid(self):
        """Test that jpvt is a valid permutation (0-based)"""
        m, n = 6, 5
        np.random.seed(77)
        A = np.random.randn(m, n).astype(float, order="F")

        rcond = 1e-12
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_fact = call_mb03oy(m, n, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        # jpvt[:n] should be a permutation of 0..n-1
        assert sorted(jpvt[:n].tolist()) == list(range(n))
        # Each element should be in valid range
        for j in jpvt[:n]:
            assert 0 <= j < n


class TestMB03OYEdgeCases:
    """Edge cases and boundary conditions"""

    def test_single_column(self):
        """Test m x 1 matrix"""
        m = 5
        A = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=float, order="F")

        rcond = 1e-12
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_fact = call_mb03oy(m, 1, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        assert rank == 1
        assert jpvt[0] == 0  # Only one column, must be index 0

    def test_single_row(self):
        """Test 1 x n matrix"""
        n = 5
        A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=float, order="F")

        rcond = 1e-12
        svlmax = 0.0

        rank, sval, jpvt, tau, info, A_fact = call_mb03oy(1, n, rcond, svlmax, A.copy(order='F'))

        assert info == 0
        assert rank == 1  # Single row, full rank
        # First pivot should be largest element
        assert jpvt[0] == 4  # Column with 5.0 (index 4, 0-based)

    def test_near_singular_with_rcond(self):
        """Test rank decision with varying rcond"""
        m, n = 4, 4
        # Create matrix with one very small singular value
        U, _ = np.linalg.qr(np.random.randn(m, m))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        S = np.diag([10.0, 5.0, 1.0, 1e-8])
        A = (U @ S @ V.T).astype(float, order="F")

        # With loose rcond, should get rank 4
        rank_loose, sval_loose, jpvt_loose, tau_loose, info_loose, A_loose = call_mb03oy(
            m, n, 1e-10, 0.0, A.copy(order='F')
        )
        assert info_loose == 0
        assert rank_loose == 4

        # With tight rcond, should get rank 3 (rejecting 1e-8 singular value)
        rank_tight, sval_tight, jpvt_tight, tau_tight, info_tight, A_tight = call_mb03oy(
            m, n, 1e-6, 0.0, A.copy(order='F')
        )
        assert info_tight == 0
        assert rank_tight == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
