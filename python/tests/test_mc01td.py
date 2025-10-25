"""
Tests for mc01td - polynomial stability checking
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from slicutlet import mc01td


class TestMC01TDContinuous:
    """Tests for continuous-time polynomial stability (dico=1)"""

    def test_stable_simple(self):
        """Test simple stable polynomial: (s+1)(s+2) = s^2 + 3s + 2"""
        p = np.array([2.0, 3.0, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            1, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is True
        assert nz_out == 0
        assert dp_out == 2
        assert iwarn_out == 0
        assert info_out == 0

    def test_unstable_simple(self):
        """Test simple unstable: (s-1)(s-2) = s^2 - 3s + 2"""
        p = np.array([2.0, -3.0, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            1, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is False
        assert nz_out == 2  # both roots in right half-plane
        assert dp_out == 2
        assert info_out == 0

    def test_mixed_stability(self):
        """Test mixed: (s+1)(s-1) = s^2 - 1"""
        p = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            1, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is False
        assert nz_out == 1  # one root in right half-plane
        assert dp_out == 2
        assert info_out == 0

    def test_trailing_zeros(self):
        """Test polynomial with trailing zeros"""
        # s^2 + 3s + 2, but with trailing zeros
        p = np.array([2.0, 3.0, 1.0, 0.0, 0.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            1, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is True
        assert nz_out == 0
        assert dp_out == 2  # degree reduced to 2
        assert iwarn_out == 2  # two trailing zeros trimmed
        assert info_out == 0

    def test_linear_stable(self):
        """Test stable linear polynomial: s + 2"""
        p = np.array([2.0, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            1, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is True
        assert nz_out == 0
        assert dp_out == 1

    def test_linear_unstable(self):
        """Test unstable linear polynomial: s - 1"""
        p = np.array([-1.0, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            1, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is False
        assert nz_out == 1
        assert dp_out == 1


class TestMC01TDDiscrete:
    """Tests for discrete-time polynomial stability (dico=0)"""

    def test_stable_inside_unit_circle(self):
        """Test stable discrete polynomial with roots inside unit circle"""
        # (z - 0.5)(z - 0.3) = z^2 - 0.8z + 0.15
        p = np.array([0.15, -0.8, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            0, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is True
        assert nz_out == 0
        assert dp_out == 2
        assert info_out == 0

    def test_unstable_outside_unit_circle(self):
        """Test unstable discrete polynomial with roots outside unit circle"""
        # (z - 1.5)(z - 2.0) = z^2 - 3.5z + 3.0
        p = np.array([3.0, -3.5, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            0, dp, p, stable, nz, dwork, iwarn, info
        )

        assert stable_out is False
        assert nz_out == 2  # both roots outside unit circle
        assert dp_out == 2
        assert info_out == 0

    def test_on_unit_circle(self):
        """Test polynomial with root on unit circle (marginally stable)"""
        # (z - 1)(z - 0.5) = z^2 - 1.5z + 0.5
        p = np.array([0.5, -1.5, 1.0], dtype=np.float64)
        dp = np.array([len(p) - 1], dtype=np.int32)
        stable = np.zeros(1, dtype=np.int32)
        nz = np.zeros(1, dtype=np.int32)
        dwork = np.zeros(4 * len(p), dtype=np.float64)
        iwarn = np.zeros(1, dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)

        stable_out, nz_out, dp_out, iwarn_out, info_out = mc01td(
            0, dp, p, stable, nz, dwork, iwarn, info
        )

        # Root at z=1 is on the boundary, so unstable for strict stability
        assert stable_out is False
        assert nz_out >= 1
        assert dp_out == 2
        assert info_out == 0
