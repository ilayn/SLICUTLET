"""
Tests for ab07nd - 2x2 partitioned matrix operations
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from slicutlet import ab07nd


class TestAB07NDBasic:
    """Basic functionality tests for ab07nd"""

    def test_quick_return_m_zero(self):
        """Test quick return when m=0"""
        n = 2
        m = 0
        A = np.eye(n)
        B = np.zeros((n, 1))  # dummy
        C = np.zeros((1, n))  # dummy
        D = np.array([[1.0]])  # dummy

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D)

        assert rcond == 1.0
        assert info == 0

    def test_identity_d_matrix(self):
        """Test with D = identity matrix"""
        n, m = 2, 2
        A = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        B = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        C = np.array([[1.0, 1.0],
                      [1.0, 2.0]])
        D = np.eye(m)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D)

        # D^{-1} = I
        assert_array_almost_equal(D_out, np.eye(m))
        # B_new = -B * I = -B
        assert_array_almost_equal(B_out, -B)
        # C_new = I * C = C
        assert_array_almost_equal(C_out, C)
        # A_new = A - B*I*C = A - B*C
        expected_A = A - B @ C
        assert_array_almost_equal(A_out, expected_A)
        assert info == 0

    def test_simple_invertible_d(self):
        """Test with simple invertible D matrix"""
        n, m = 2, 2
        A = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        B = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        C = np.array([[2.0, 0.0],
                      [0.0, 2.0]])
        D = np.array([[2.0, 0.0],
                      [0.0, 2.0]])

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D)

        # D^{-1} should be [[0.5, 0], [0, 0.5]]
        expected_Dinv = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert_array_almost_equal(D_out, expected_Dinv)
        assert info == 0
        assert rcond > 0.4  # well-conditioned

    def test_rectangular_case(self):
        """Test with n != m"""
        n, m = 3, 2
        A = np.eye(n)
        B = np.ones((n, m))
        C = np.ones((m, n))
        D = 2.0 * np.eye(m)

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D)

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
        A = np.eye(n)
        B = np.eye(n, m)
        C = np.eye(m, n)
        D = np.diag([2.0, 3.0])

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D)

        assert rcond > 0.5  # well-conditioned
        assert info == 0

    def test_ill_conditioned_d(self):
        """Test with ill-conditioned D matrix"""
        n, m = 2, 2
        A = np.eye(n)
        B = np.eye(n, m)
        C = np.eye(m, n)
        # Very ill-conditioned
        D = np.array([[1.0, 1.0],
                      [1.0, 1.0 + 1e-10]])

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A, B, C, D)

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

        A_out, B_out, C_out, D_out, rcond, info = ab07nd(n, m, A.copy(order='F'), B.copy(order='F'),
                                                          C.copy(order='F'), D.copy(order='F'))        # D_out should be D^{-1}
        assert_array_almost_equal(D @ D_out, np.eye(m), decimal=10)

        # C_out should be D^{-1} * C
        assert_array_almost_equal(C_out, D_out @ C, decimal=10)

        # B_out should be -B * D^{-1}
        assert_array_almost_equal(B_out, -B @ D_out, decimal=10)


class TestAB07NDInputValidation:
    """Test input validation"""

    def test_negative_n(self):
        """Test error on negative n"""
        with pytest.raises(ValueError, match="n must be"):
            ab07nd(-1, 2, np.eye(2), np.eye(2, 2), np.eye(2, 2), np.eye(2))

    def test_negative_m(self):
        """Test error on negative m"""
        with pytest.raises(ValueError, match="m must be"):
            ab07nd(2, -1, np.eye(2), np.eye(2, 2), np.eye(2, 2), np.eye(2))

    def test_wrong_a_shape(self):
        """Test error on wrong A shape"""
        n, m = 2, 2
        with pytest.raises(ValueError, match="A must be"):
            ab07nd(n, m, np.eye(3), np.eye(n, m), np.eye(m, n), np.eye(m))

    def test_wrong_b_shape(self):
        """Test error on wrong B shape"""
        n, m = 2, 2
        with pytest.raises(ValueError, match="B must be"):
            ab07nd(n, m, np.eye(n), np.eye(3, 3), np.eye(m, n), np.eye(m))

    def test_wrong_c_shape(self):
        """Test error on wrong C shape"""
        n, m = 2, 2
        with pytest.raises(ValueError, match="C must be"):
            ab07nd(n, m, np.eye(n), np.eye(n, m), np.eye(3, 3), np.eye(m))

    def test_wrong_d_shape(self):
        """Test error on wrong D shape"""
        n, m = 2, 2
        with pytest.raises(ValueError, match="D must be"):
            ab07nd(n, m, np.eye(n), np.eye(n, m), np.eye(m, n), np.eye(3))

    def test_fortran_vs_c_order(self):
        """Test that both C and Fortran ordered arrays work"""
        n, m = 2, 2
        A_c = np.ascontiguousarray(np.eye(n))
        A_f = np.asfortranarray(np.eye(n))
        B = np.eye(n, m)
        C = np.eye(m, n)
        D = np.eye(m)

        result_c = ab07nd(n, m, A_c, B, C, D)
        result_f = ab07nd(n, m, A_f, B, C, D)

        # Results should be identical
        assert_array_almost_equal(result_c[0], result_f[0])
