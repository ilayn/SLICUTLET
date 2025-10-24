"""
Tests for mc01td - polynomial stability checking
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from slicutlet import mc01td


class TestMC01TDContinuous:
    """Tests for continuous-time polynomial stability (dico='C' or 1)"""

    def test_stable_simple(self):
        """Test simple stable polynomial: (s+1)(s+2) = s^2 + 3s + 2"""
        p = np.array([2.0, 3.0, 1.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is True
        assert nz == 0
        assert dp == 2
        assert iwarn == 0
        assert info == 0

    def test_unstable_simple(self):
        """Test simple unstable: (s-1)(s-2) = s^2 - 3s + 2"""
        p = np.array([2.0, -3.0, 1.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is False
        assert nz == 2  # both roots in right half-plane
        assert dp == 2
        assert info == 0

    def test_mixed_stability(self):
        """Test mixed: (s+1)(s-1) = s^2 - 1"""
        p = np.array([-1.0, 0.0, 1.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is False
        assert nz == 1  # one root in right half-plane
        assert dp == 2
        assert info == 0

    def test_trailing_zeros(self):
        """Test polynomial with trailing zeros"""
        # s^2 + 3s + 2, but with trailing zeros
        p = np.array([2.0, 3.0, 1.0, 0.0, 0.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is True
        assert nz == 0
        assert dp == 2  # degree reduced to 2
        assert iwarn == 2  # two trailing zeros trimmed
        assert info == 0

    def test_zero_polynomial(self):
        """Test all-zero polynomial"""
        p = np.array([0.0, 0.0, 0.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert info == 1  # zero polynomial error
        assert dp == -1

    def test_linear_stable(self):
        """Test stable linear polynomial: s + 2"""
        p = np.array([2.0, 1.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is True
        assert nz == 0
        assert dp == 1

    def test_linear_unstable(self):
        """Test unstable linear polynomial: s - 1"""
        p = np.array([-1.0, 1.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is False
        assert nz == 1
        assert dp == 1

    def test_constant_positive(self):
        """Test positive constant polynomial"""
        p = np.array([5.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is True
        assert nz == 0
        assert dp == 0

    @pytest.mark.parametrize("roots,expected_nz", [
        ([-1, -2, -3], 0),           # all stable
        ([1, 2, 3], 3),              # all unstable
        ([-1, -2, 1], 1),            # one unstable
        ([-1, 1, 2], 2),             # two unstable
        ([-5, -1, 0.5, 1], 2),       # mixed higher order
    ])
    def test_parametric_continuous(self, roots, expected_nz):
        """Parametric test with various root configurations"""
        # np.poly gives descending coefficients, reverse for ascending
        p = np.poly(roots)[::-1]
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert nz == expected_nz
        assert stable == (nz == 0)
        assert info == 0


class TestMC01TDDiscrete:
    """Tests for discrete-time polynomial stability (dico='D' or 0)"""

    def test_stable_simple(self):
        """Test stable discrete: (z-0.5)(z-0.25) = z^2 - 0.75z + 0.125"""
        p = np.array([0.125, -0.75, 1.0])
        stable, nz, dp, iwarn, info = mc01td('D', p)

        assert stable is True
        assert nz == 0
        assert dp == 2
        assert info == 0

    def test_unstable_one_outside(self):
        """Test one root outside unit circle: (z-1.5)(z-0.5)"""
        p = np.array([0.75, -2.0, 1.0])
        stable, nz, dp, iwarn, info = mc01td('D', p)

        assert stable is False
        assert nz == 1
        assert dp == 2
        assert info == 0

    def test_unstable_both_outside(self):
        """Test both roots outside: (z-1.5)(z-2)"""
        p = np.array([3.0, -3.5, 1.0])
        stable, nz, dp, iwarn, info = mc01td('D', p)

        assert stable is False
        assert nz == 2
        assert dp == 2
        assert info == 0

    def test_trailing_zeros_discrete(self):
        """Test discrete polynomial with trailing zeros"""
        p = np.array([0.75, -2.0, 1.0, 0.0])
        stable, nz, dp, iwarn, info = mc01td('D', p)

        assert stable is False
        assert nz == 1
        assert dp == 2
        assert iwarn == 1

    @pytest.mark.parametrize("roots,expected_nz", [
        ([0.5, 0.3], 0),             # all inside
        ([1.5, 2.0], 2),             # all outside
        ([0.5, 1.5], 1),             # one outside
        ([0.9, 0.8, 0.7], 0),        # all stable
        ([0.5, 1.2, 1.8], 2),        # two outside
    ])
    def test_parametric_discrete(self, roots, expected_nz):
        """Parametric test for discrete case"""
        p = np.poly(roots)[::-1]
        stable, nz, dp, iwarn, info = mc01td('D', p)

        assert nz == expected_nz
        assert stable == (nz == 0)
        assert info == 0


class TestMC01TDInputValidation:
    """Test input validation and error handling"""

    def test_invalid_dico_string(self):
        """Test invalid dico string"""
        p = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="dico must be"):
            mc01td('X', p)

    def test_invalid_dico_int(self):
        """Test invalid dico integer"""
        p = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="dico must be"):
            mc01td(5, p)

    def test_empty_polynomial(self):
        """Test empty polynomial array"""
        p = np.array([])
        with pytest.raises(ValueError, match="at least one coefficient"):
            mc01td('C', p)

    def test_dico_numeric_equivalence(self):
        """Test that 'C'/1 and 'D'/0 are equivalent"""
        p = np.array([2.0, 3.0, 1.0])

        # Continuous
        result_C = mc01td('C', p)
        result_1 = mc01td(1, p)
        assert result_C == result_1

        # Discrete
        p2 = np.array([0.125, -0.75, 1.0])
        result_D = mc01td('D', p2)
        result_0 = mc01td(0, p2)
        assert result_D == result_0

    def test_float_coefficients(self):
        """Test that various numeric types work"""
        # Integer coefficients
        p_int = np.array([2, 3, 1])
        stable1, *_ = mc01td('C', p_int)

        # Float coefficients
        p_float = np.array([2.0, 3.0, 1.0])
        stable2, *_ = mc01td('C', p_float)

        assert stable1 == stable2


class TestMC01TDNumericalEdgeCases:
    """Test numerical edge cases and borderline stability"""

    def test_very_small_coefficients(self):
        """Test polynomial with very small coefficients"""
        p = np.array([1e-10, 1e-9, 1e-8])
        stable, nz, dp, iwarn, info = mc01td('C', p)
        # Should handle gracefully
        assert info in (0, 1, 2)

    def test_large_degree(self):
        """Test higher degree polynomial"""
        # All roots at -1: (s+1)^5
        roots = [-1] * 5
        p = np.poly(roots)[::-1]
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is True
        assert nz == 0

    def test_repeated_roots(self):
        """Test polynomial with repeated roots"""
        # (s+2)^2 = s^2 + 4s + 4
        p = np.array([4.0, 4.0, 1.0])
        stable, nz, dp, iwarn, info = mc01td('C', p)

        assert stable is True
        assert nz == 0
