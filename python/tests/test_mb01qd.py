"""
Tests for mb01qd - matrix multiplication by scalar CTO/CFROM
"""

import numpy as np
from numpy.testing import assert_allclose

from slicutlet import mb01qd


class TestMB01QDFullMatrix:
    """Tests for full matrix scaling (type=0)"""

    def test_scale_full_matrix_simple(self):
        """Test simple scaling of full matrix"""
        m, n = 3, 3
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
        cfrom, cto = 2.0, 6.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Should multiply by 6/2 = 3
        expected = np.array([[3.0, 6.0, 9.0],
                             [12.0, 15.0, 18.0],
                             [21.0, 24.0, 27.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0


    def test_scale_full_matrix_fraction(self):
        """Test scaling with fractional multiplier"""
        m, n = 2, 2
        a = np.array([[10.0, 20.0],
                      [30.0, 40.0]], dtype=np.float64, order='F')
        cfrom, cto = 4.0, 1.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Should multiply by 1/4 = 0.25
        expected = np.array([[2.5, 5.0],
                             [7.5, 10.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0


    def test_rectangular_matrix(self):
        """Test with rectangular matrix"""
        m, n = 3, 2
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float64, order='F')
        cfrom, cto = 1.0, 10.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        expected = np.array([[10.0, 20.0],
                             [30.0, 40.0],
                             [50.0, 60.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0


class TestMB01QDTriangular:
    """Tests for triangular matrices"""

    def test_lower_triangular(self):
        """Test scaling lower triangular matrix (type=1)"""
        m, n = 3, 3
        a = np.array([[1.0, 0.0, 0.0],
                      [2.0, 3.0, 0.0],
                      [4.0, 5.0, 6.0]], dtype=np.float64, order='F')
        cfrom, cto = 1.0, 2.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            1, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Lower triangle should be doubled
        expected = np.array([[2.0, 0.0, 0.0],
                             [4.0, 6.0, 0.0],
                             [8.0, 10.0, 12.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0


    def test_upper_triangular(self):
        """Test scaling upper triangular matrix (type=2)"""
        m, n = 3, 3
        a = np.array([[1.0, 2.0, 3.0],
                      [0.0, 4.0, 5.0],
                      [0.0, 0.0, 6.0]], dtype=np.float64, order='F')
        cfrom, cto = 2.0, 4.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            2, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Upper triangle should be doubled
        expected = np.array([[2.0, 4.0, 6.0],
                             [0.0, 8.0, 10.0],
                             [0.0, 0.0, 12.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0


    def test_upper_hessenberg(self):
        """Test scaling upper Hessenberg matrix (type=3)"""
        m, n = 4, 4
        a = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [0.0, 9.0, 10.0, 11.0],
                      [0.0, 0.0, 12.0, 13.0]], dtype=np.float64, order='F')
        cfrom, cto = 1.0, 3.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            3, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Upper Hessenberg part (diagonal + upper + first subdiagonal) should be tripled
        expected = np.array([[3.0, 6.0, 9.0, 12.0],
                             [15.0, 18.0, 21.0, 24.0],
                             [0.0, 27.0, 30.0, 33.0],
                             [0.0, 0.0, 36.0, 39.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0


class TestMB01QDOverflow:
    """Tests for overflow/underflow handling"""

    def test_avoid_overflow_large_values(self):
        """Test that scaling avoids overflow with very large values"""
        m, n = 2, 2
        a = np.array([[1e100, 2e100],
                      [3e100, 4e100]], dtype=np.float64, order='F')
        cfrom, cto = 1e50, 1e150

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Should scale without overflow
        assert np.all(np.isfinite(a_out))
        assert info_out == 0
        # Check relative magnitudes preserved
        assert_allclose(a_out[0, 0] / a_out[0, 1], 0.5, rtol=1e-10)


    def test_avoid_underflow_small_values(self):
        """Test that scaling avoids underflow with very small values"""
        m, n = 2, 2
        a = np.array([[1e-100, 2e-100],
                      [3e-100, 4e-100]], dtype=np.float64, order='F')
        cfrom, cto = 1e150, 1e50

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Should scale without underflow
        assert np.all(np.isfinite(a_out))
        assert info_out == 0
        # Check relative magnitudes preserved
        assert_allclose(a_out[0, 0] / a_out[0, 1], 0.5, rtol=1e-10)


class TestMB01QDEdgeCases:
    """Tests for edge cases"""

    def test_empty_matrix(self):
        """Test with empty matrix"""
        m, n = 0, 0
        a = np.array([], dtype=np.float64, order='F').reshape(0, 0)
        cfrom, cto = 1.0, 2.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        assert info_out == 0
        assert a_out.shape == (0, 0)


    def test_identity_scaling(self):
        """Test scaling by 1 (cfrom == cto)"""
        m, n = 2, 2
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float64, order='F')
        a_orig = a.copy()
        cfrom, cto = 5.0, 5.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        # Matrix should remain unchanged
        assert_allclose(a_out, a_orig, rtol=1e-15)
        assert info_out == 0


    def test_single_element(self):
        """Test with 1x1 matrix"""
        m, n = 1, 1
        a = np.array([[5.0]], dtype=np.float64, order='F')
        cfrom, cto = 2.0, 10.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        assert_allclose(a_out[0, 0], 25.0, rtol=1e-15)
        assert info_out == 0


    def test_zeros_in_matrix(self):
        """Test matrix with zero elements"""
        m, n = 3, 3
        a = np.array([[1.0, 0.0, 3.0],
                      [0.0, 5.0, 0.0],
                      [7.0, 0.0, 9.0]], dtype=np.float64, order='F')
        cfrom, cto = 1.0, 2.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        expected = np.array([[2.0, 0.0, 6.0],
                             [0.0, 10.0, 0.0],
                             [14.0, 0.0, 18.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0


    def test_negative_values(self):
        """Test with negative values"""
        m, n = 2, 2
        a = np.array([[-1.0, 2.0],
                      [-3.0, -4.0]], dtype=np.float64, order='F')
        cfrom, cto = 1.0, 3.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01qd(
            0, m, n, 0, 0, cfrom, cto, 0, nrows, a
        )

        expected = np.array([[-3.0, 6.0],
                             [-9.0, -12.0]], dtype=np.float64, order='F')
        assert_allclose(a_out, expected, rtol=1e-15)
        assert info_out == 0
