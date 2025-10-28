"""
Tests for ab07nd - 2x2 partitioned matrix operations
"""

import numpy as np
from numpy.testing import assert_allclose

from slicutlet import ab07nd


class TestAB07NDBasic:
    """Basic functionality tests for ab07nd"""

    def test_quick_return_m_zero(self):
        """Test quick return when m=0"""
        n, m = 2, 0
        A = np.asfortranarray(np.eye(n))
        B = np.asfortranarray(np.zeros((n, 1)))  # dummy
        C = np.asfortranarray(np.zeros((1, n)))  # dummy
        D = np.asfortranarray([[1.0]])  # dummy

        iwork = np.zeros(2 * max(2, m, 1), dtype=np.int32)
        dwork = np.zeros(max(4 * max(m, 1), n * max(m, 1), 256), dtype=np.float64)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D, iwork, dwork)

        assert rcond == 1.0
        assert info == 0

    def test_identity_d_matrix(self):
        """Test with D = identity matrix"""
        n, m = 2, 2
        A = np.asfortranarray([[1.0, 2.0],
                      [3.0, 4.0]])
        B = np.asfortranarray([[1.0, 0.0],
                      [0.0, 1.0]])
        C = np.asfortranarray([[1.0, 1.0],
                      [1.0, 2.0]])
        D = np.asfortranarray(np.eye(m))

        # Save originals for comparison
        A_orig = A.copy(order='F')
        B_orig = B.copy(order='F')
        C_orig = C.copy(order='F')
        D_orig = D.copy(order='F')

        iwork = np.zeros(2 * max(2, m), dtype=np.int32)
        dwork = np.zeros(max(4 * m, n * m, 256), dtype=np.float64)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D, iwork, dwork)

        assert info == 0, f"ab07nd returned info={info}, rcond={rcond}"

        # D^{-1} = I
        assert_allclose(D_out, np.eye(m), rtol=1e-10, atol=1e-12)
        # B_new = -B * I = -B
        assert_allclose(B_out, -B_orig, rtol=1e-10, atol=1e-12)
        # C_new = I * C = C
        assert_allclose(C_out, C_orig, rtol=1e-10, atol=1e-12)
        # A_new = A - B*I*C = A - B*C
        expected_A = A_orig - B_orig @ C_orig
        assert_allclose(A_out, expected_A, rtol=1e-10, atol=1e-12)
        assert info == 0

    def test_simple_invertible_d(self):
        """Test with simple invertible D matrix"""
        n, m = 2, 2
        A = np.asfortranarray([[1.0, 0.0],
                      [0.0, 1.0]])
        B = np.asfortranarray([[1.0, 0.0],
                      [0.0, 1.0]])
        C = np.asfortranarray([[2.0, 0.0],
                      [0.0, 2.0]])
        D = np.asfortranarray([[2.0, 0.0],
                      [0.0, 2.0]])

        iwork = np.zeros(2 * max(2, m), dtype=np.int32)
        dwork = np.zeros(max(4 * m, n * m, 256), dtype=np.float64)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D, iwork, dwork)

        # D^{-1} should be [[0.5, 0], [0, 0.5]]
        expected_Dinv = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert_allclose(D_out, expected_Dinv, rtol=1e-10, atol=1e-12)
        assert info == 0
        assert rcond > 0.4  # well-conditioned

    def test_rectangular_case(self):
        """Test with n != m"""
        n, m = 3, 2
        A = np.asfortranarray(np.eye(n))
        B = np.asfortranarray(np.ones((n, m)))
        C = np.asfortranarray(np.ones((m, n)))
        D = np.asfortranarray(2.0 * np.eye(m))

        iwork = np.zeros(2 * max(2, m), dtype=np.int32)
        dwork = np.zeros(max(4 * m, n * m, 256), dtype=np.float64)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D, iwork, dwork)

        assert A_out.shape == (n, n)
        assert B_out.shape == (n, m)
        assert C_out.shape == (m, n)
        assert D_out.shape == (m, m)
        assert info == 0


class TestAB07NDNumerical:
    """Numerical accuracy and conditioning tests"""

    def test_well_conditioned_system(self):
        """Test that well-conditioned D gives good rcond"""
        n, m = 2, 2
        A = np.asfortranarray(np.eye(n))
        B = np.asfortranarray(np.eye(n, m))
        C = np.asfortranarray(np.eye(m, n))
        D = np.asfortranarray(np.diag([2.0, 3.0]))

        iwork = np.zeros(2 * max(2, m), dtype=np.int32)
        dwork = np.zeros(max(4 * m, n * m, 256), dtype=np.float64)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D, iwork, dwork)

        assert rcond > 0.5  # well-conditioned
        assert info == 0

    def test_ill_conditioned_d(self):
        """Test with ill-conditioned D matrix"""
        n, m = 2, 2
        A = np.asfortranarray(np.eye(n))
        B = np.asfortranarray(np.eye(n, m))
        C = np.asfortranarray(np.eye(m, n))
        # Very ill-conditioned
        D = np.asfortranarray([[1.0, 1.0],
                      [1.0, 1.0 + 1e-10]])

        iwork = np.zeros(2 * max(2, m), dtype=np.int32)
        dwork = np.zeros(max(4 * m, n * m, 256), dtype=np.float64)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D, iwork, dwork)

        # Should warn about numerical singularity
        assert rcond < 1e-8 or info == m + 1

    def test_transformation_properties(self):
        """Verify the mathematical properties of the transformation"""
        rng = np.random.default_rng(42)
        n, m = 2, 2
        A = np.asfortranarray(rng.random((n, n)))
        B = np.asfortranarray(rng.random((n, m)))
        C = np.asfortranarray(rng.random((m, n)))
        D = np.asfortranarray(np.eye(m) + 0.1 * rng.random((m, m)))  # make sure invertible

        # Save originals for comparison
        D_orig = D.copy(order='F')
        B_orig = B.copy(order='F')
        C_orig = C.copy(order='F')

        iwork = np.zeros(2 * max(2, m), dtype=np.int32)
        dwork = np.zeros(max(4 * m, n * m, 256), dtype=np.float64)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D, iwork, dwork)

        # D_out should be D^{-1}
        assert_allclose(D_orig @ D_out, np.eye(m), rtol=1e-10, atol=1e-12)

        # C_out should be D^{-1} * C
        assert_allclose(C_out, D_out @ C_orig, rtol=1e-10, atol=1e-12)

        # B_out should be -B * D^{-1}
        assert_allclose(B_out, -B_orig @ D_out, rtol=1e-10, atol=1e-12)
