"""
Tests for mb01pd - matrix scaling/unscaling
"""

import numpy as np
from numpy.testing import assert_allclose

from slicutlet import mb01pd


class TestMB01PDScale:
    """Tests for scaling matrices (scun=1)"""

    def test_scale_full_matrix_small_norm(self):
        """Test scaling a full matrix with small norm"""
        m, n = 3, 3
        a = np.array([[1e-300, 2e-300, 3e-300],
                      [4e-300, 5e-300, 6e-300],
                      [7e-300, 8e-300, 9e-300]], dtype=np.float64, order='F')
        anrm = 1e-300
        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 0, m, n, 0, 0, anrm, 0, nrows, a
        )

        # After scaling, values should be larger
        assert np.all(np.abs(a_out) > 1e-300)
        assert info_out == 0
        # Check that scaling preserved relative magnitudes
        assert_allclose(a_out[0, 0] * 2, a_out[0, 1], rtol=1e-10)


    def test_scale_full_matrix_large_norm(self):
        """Test scaling a full matrix with large norm"""
        m, n = 2, 2
        a = np.array([[1e300, 2e300],
                      [3e300, 4e300]], dtype=np.float64, order='F')
        anrm = 1e300

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 0, m, n, 0, 0, anrm, 0, nrows, a
        )

        # After scaling, values should be smaller
        assert np.all(np.abs(a_out) < 1e300)
        assert info_out == 0
        # Check that ratios are preserved
        assert_allclose(a_out[0, 0] / a_out[0, 1], 0.5, rtol=1e-10)


    def test_scale_lower_triangular(self):
        """Test scaling a lower triangular matrix"""
        m, n = 3, 3
        a = np.array([[1e-300, 0.0, 0.0],
                      [2e-300, 3e-300, 0.0],
                      [4e-300, 5e-300, 6e-300]], dtype=np.float64, order='F')
        anrm = 1e-300

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 1, m, n, 0, 0, anrm, 0, nrows, a
        )

        # Lower triangle should be scaled
        assert np.abs(a_out[0, 0]) > 1e-300
        assert np.abs(a_out[1, 0]) > 1e-300
        # Upper triangle (zeros) should remain unchanged
        assert a_out[0, 1] == 0.0
        assert a_out[0, 2] == 0.0
        assert info_out == 0


    def test_scale_upper_triangular(self):
        """Test scaling an upper triangular matrix"""
        m, n = 3, 3
        a = np.array([[1e-300, 2e-300, 3e-300],
                      [0.0, 4e-300, 5e-300],
                      [0.0, 0.0, 6e-300]], dtype=np.float64, order='F')
        anrm = 1e-300

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 2, m, n, 0, 0, anrm, 0, nrows, a
        )

        # Upper triangle should be scaled
        assert np.abs(a_out[0, 0]) > 1e-300
        assert np.abs(a_out[0, 1]) > 1e-300
        # Lower triangle (zeros) should remain unchanged
        assert a_out[1, 0] == 0.0
        assert a_out[2, 0] == 0.0
        assert a_out[2, 1] == 0.0
        assert info_out == 0


    def test_no_scaling_needed(self):
        """Test when norm is in safe range, no scaling needed"""
        m, n = 2, 2
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float64, order='F')
        a_orig = a.copy()
        anrm = 2.5  # In safe range

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 0, m, n, 0, 0, anrm, 0, nrows, a
        )

        # Matrix should remain unchanged
        assert_allclose(a_out, a_orig, rtol=1e-15)
        assert info_out == 0


class TestMB01PDUnscale:
    """Tests for unscaling matrices (scun=0)"""

    def test_unscale_after_scale(self):
        """Test unscaling after scaling"""
        m, n = 2, 2
        a_orig = np.array([[1e-300, 2e-300],
                           [3e-300, 4e-300]], dtype=np.float64, order='F')
        anrm = 1e-300

        nrows = np.array([], dtype=np.int32)

        # First scale
        a_scaled = a_orig.copy()
        a_scaled, _ = mb01pd(1, 0, m, n, 0, 0, anrm, 0, nrows, a_scaled)

        # Then unscale
        a_unscaled, info_out = mb01pd(0, 0, m, n, 0, 0, anrm, 0, nrows, a_scaled)

        # Should get back original values
        assert_allclose(a_unscaled, a_orig, rtol=1e-10)
        assert info_out == 0


    def test_unscale_lower_triangular(self):
        """Test unscaling a lower triangular matrix"""
        m, n = 3, 3
        a_orig = np.array([[1e-300, 0.0, 0.0],
                           [2e-300, 3e-300, 0.0],
                           [4e-300, 5e-300, 6e-300]], dtype=np.float64, order='F')
        anrm = 1e-300

        nrows = np.array([], dtype=np.int32)

        # Scale then unscale
        a_scaled = a_orig.copy()
        a_scaled, _ = mb01pd(1, 1, m, n, 0, 0, anrm, 0, nrows, a_scaled)
        a_unscaled, info_out = mb01pd(0, 1, m, n, 0, 0, anrm, 0, nrows, a_scaled)

        assert_allclose(a_unscaled, a_orig, rtol=1e-10)
        assert info_out == 0


class TestMB01PDEdgeCases:
    """Tests for edge cases"""

    def test_empty_matrix(self):
        """Test with empty matrix"""
        m, n = 0, 0
        a = np.array([], dtype=np.float64, order='F').reshape(0, 0)
        anrm = 1.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 0, m, n, 0, 0, anrm, 0, nrows, a
        )

        assert info_out == 0
        assert a_out.shape == (0, 0)


    def test_zero_norm(self):
        """Test with zero norm (should return immediately)"""
        m, n = 2, 2
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float64, order='F')
        a_orig = a.copy()
        anrm = 0.0

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 0, m, n, 0, 0, anrm, 0, nrows, a
        )

        # Matrix should remain unchanged
        assert_allclose(a_out, a_orig, rtol=1e-15)
        assert info_out == 0


    def test_single_element(self):
        """Test with 1x1 matrix"""
        m, n = 1, 1
        a = np.array([[1e-300]], dtype=np.float64, order='F')
        anrm = 1e-300

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 0, m, n, 0, 0, anrm, 0, nrows, a
        )

        assert np.abs(a_out[0, 0]) > 1e-300
        assert info_out == 0


    def test_rectangular_matrix(self):
        """Test with rectangular matrix"""
        m, n = 3, 2
        a = np.array([[1e-300, 2e-300],
                      [3e-300, 4e-300],
                      [5e-300, 6e-300]], dtype=np.float64, order='F')
        anrm = 1e-300

        nrows = np.array([], dtype=np.int32)

        a_out, info_out = mb01pd(
            1, 0, m, n, 0, 0, anrm, 0, nrows, a
        )

        assert np.all(np.abs(a_out) > 1e-300)
        assert info_out == 0
        # Verify relative magnitudes preserved
        assert_allclose(a_out[0, 0] * 2, a_out[0, 1], rtol=1e-10)
