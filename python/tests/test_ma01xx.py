"""
Tests for MA01XX family of functions
"""
import numpy as np
from numpy.testing import assert_allclose
from slicutlet import (
    ma01ad, ma01bd, ma01bz, ma01cd, ma01dd, ma01dz
)


class TestMA01AD:
    """
    Test MA01AD - Complex square root computation in real arithmetic.
    Returns YR + i*YI = sqrt(XR + i*XI) with YR >= 0 and SIGN(YI) = SIGN(XI).
    """

    def test_positive_real_positive_imag(self):
        """Test with positive real and imaginary parts"""
        xr, xi = 3.0, 4.0
        yr, yi = ma01ad(xr, xi)

        # Verify: (yr + i*yi)^2 = xr + i*xi
        computed_real = yr * yr - yi * yi
        computed_imag = 2 * yr * yi

        assert_allclose(computed_real, xr, rtol=1e-14)
        assert_allclose(computed_imag, xi, rtol=1e-14)
        assert yr >= 0.0

    def test_negative_real_positive_imag(self):
        """Test with negative real and positive imaginary parts"""
        xr, xi = -3.0, 4.0
        yr, yi = ma01ad(xr, xi)

        computed_real = yr * yr - yi * yi
        computed_imag = 2 * yr * yi

        assert_allclose(computed_real, xr, rtol=1e-14)
        assert_allclose(computed_imag, xi, rtol=1e-14)
        assert yr >= 0.0
        assert np.sign(yi) == np.sign(xi)

    def test_positive_real_negative_imag(self):
        """Test with positive real and negative imaginary parts"""
        xr, xi = 3.0, -4.0
        yr, yi = ma01ad(xr, xi)

        computed_real = yr * yr - yi * yi
        computed_imag = 2 * yr * yi

        assert_allclose(computed_real, xr, rtol=1e-14)
        assert_allclose(computed_imag, xi, rtol=1e-14)
        assert yr >= 0.0
        assert np.sign(yi) == np.sign(xi)

    def test_negative_real_negative_imag(self):
        """Test with negative real and negative imaginary parts"""
        xr, xi = -3.0, -4.0
        yr, yi = ma01ad(xr, xi)

        computed_real = yr * yr - yi * yi
        computed_imag = 2 * yr * yi

        assert_allclose(computed_real, xr, rtol=1e-14)
        assert_allclose(computed_imag, xi, rtol=1e-14)
        assert yr >= 0.0
        assert np.sign(yi) == np.sign(xi)

    def test_real_only(self):
        """Test with purely real number"""
        xr, xi = 16.0, 0.0
        yr, yi = ma01ad(xr, xi)

        assert_allclose(yr, 4.0, rtol=1e-14)
        assert_allclose(yi, 0.0, atol=1e-14)

    def test_imaginary_only(self):
        """Test with purely imaginary number"""
        xr, xi = 0.0, 16.0
        yr, yi = ma01ad(xr, xi)

        computed_real = yr * yr - yi * yi
        computed_imag = 2 * yr * yi

        assert_allclose(computed_real, xr, atol=1e-14)
        assert_allclose(computed_imag, xi, rtol=1e-14)

    def test_zero(self):
        """Test square root of zero"""
        xr, xi = 0.0, 0.0
        yr, yi = ma01ad(xr, xi)

        assert_allclose(yr, 0.0, atol=1e-14)
        assert_allclose(yi, 0.0, atol=1e-14)


class TestMA01BD:
    """
    Test MA01BD - General product of K real scalars without overflow/underflow.
    Computes ALPHA / BETA * BASE**(SCALE) = product of scalars.
    """

    def test_simple_product(self):
        """Test simple product of positive numbers"""
        base = 2.0
        logbase = np.log(base).item()
        k = 3
        s = np.array([1, 1, 1], dtype=np.int32)
        a = np.array([2.0, 3.0, 4.0])
        inca = 1

        alpha, beta, scale = ma01bd(base, logbase, k, inca, a, s)

        # Expected: 2 * 3 * 4 = 24
        result = alpha / beta * (base ** scale)
        assert_allclose(result, 24.0, rtol=1e-10)

    def test_mixed_signature(self):
        """Test with mixed multiplication and division"""
        base = 2.0
        logbase = float(np.log(base))
        k = 4
        s = np.array([1, -1, 1, -1], dtype=np.int32)
        a = np.array([8.0, 2.0, 6.0, 3.0])
        inca = 1

        alpha, beta, scale = ma01bd(base, logbase, k, inca, a, s)

        # Expected: 8 / 2 * 6 / 3 = 8
        result = alpha / beta * (base ** scale)
        assert_allclose(result, 8.0, rtol=1e-10)

    def test_large_numbers(self):
        """Test with large numbers to check scaling"""
        base = 10.0
        logbase = float(np.log(base))
        k = 3
        s = np.array([1, 1, 1], dtype=np.int32)
        a = np.array([1e100, 1e100, 1e100])
        inca = 1

        alpha, beta, scale = ma01bd(base, logbase, k, inca, a, s)

        # Result should be representable
        assert np.isfinite(alpha)
        assert np.isfinite(beta)
        assert 0.1 <= abs(alpha) <= 10.0  # Should be normalized

    def test_small_numbers(self):
        """Test with small numbers to check scaling"""
        base = 10.0
        logbase = float(np.log(base))
        k = 3
        s = np.array([1, 1, 1], dtype=np.int32)
        a = np.array([1e-100, 1e-100, 1e-100])
        inca = 1

        alpha, beta, scale = ma01bd(base, logbase, k, inca, a, s)

        # Result should be representable
        assert np.isfinite(alpha)
        assert np.isfinite(beta)

    def test_with_increment(self):
        """Test with array increment > 1"""
        base = 2.0
        logbase = float(np.log(base))
        k = 3
        s = np.array([1, 1, 1], dtype=np.int32)
        a = np.array([2.0, 999.0, 3.0, 999.0, 4.0, 999.0])
        inca = 2  # Take every other element

        alpha, beta, scale = ma01bd(base, logbase, k, inca, a, s)

        # Expected: 2 * 3 * 4 = 24
        result = alpha / beta * (base ** scale)
        assert_allclose(result, 24.0, rtol=1e-10)


class TestMA01BZ:
    """
    Test MA01BZ - General product of K complex scalars avoiding overflow.
    Computes ALPHA / BETA * BASE**(SCALE) = product of complex scalars.
    """

    def test_simple_product(self):
        """Test simple product of complex numbers"""
        base = 2.0
        k = 3
        s = np.array([1, 1, 1], dtype=np.int32)
        a = np.array([1+1j, 2+0j, 1-1j], dtype=np.complex128)
        inca = 1

        alpha, beta, scale = ma01bz(base, k, inca, a, s)

        # Expected: (1+1j) * (2+0j) * (1-1j) = (2+2j) * (1-1j) = 4+0j
        result = alpha / beta * (base ** scale)
        expected = (1+1j) * (2+0j) * (1-1j)
        assert_allclose(result, expected, rtol=1e-10)

    def test_with_division(self):
        """Test with division (s = -1)"""
        base = 2.0
        k = 2
        s = np.array([1, -1], dtype=np.int32)
        a = np.array([4+4j, 1+1j], dtype=np.complex128)
        inca = 1

        alpha, beta, scale = ma01bz(base, k, inca, a, s)

        # Expected: (4+4j) / (1+1j) = 4
        result = alpha / beta * (base ** scale)
        expected = (4+4j) / (1+1j)
        assert_allclose(result, expected, rtol=1e-10)

    def test_zero_in_numerator(self):
        """Test with zero in numerator position"""
        base = 2.0
        k = 2
        s = np.array([1, 1], dtype=np.int32)
        a = np.array([0+0j, 2+3j], dtype=np.complex128)
        inca = 1

        alpha, beta, scale = ma01bz(base, k, inca, a, s)

        # Expected: 0 * (2+3j) = 0
        assert_allclose(alpha, 0.0, atol=1e-14)

    def test_zero_in_denominator(self):
        """Test with zero in denominator position"""
        base = 2.0
        k = 2
        s = np.array([1, -1], dtype=np.int32)
        a = np.array([2+3j, 0+0j], dtype=np.complex128)
        inca = 1

        alpha, beta, scale = ma01bz(base, k, inca, a, s)

        # Beta should be zero to indicate division by zero
        assert_allclose(beta, 0.0, atol=1e-14)

    def test_normalization(self):
        """Test that result stays normalized"""
        base = 2.0
        k = 3
        s = np.array([1, 1, 1], dtype=np.int32)
        a = np.array([100+0j, 100+0j, 100+0j], dtype=np.complex128)
        inca = 1

        alpha, beta, scale = ma01bz(base, k, inca, a, s)

        # Alpha should be normalized: 1 <= |alpha| < base
        mag = np.abs(alpha)
        assert 1.0 <= mag < base or np.isclose(mag, 0.0)


class TestMA01CD:
    """
    Test MA01CD - Sign of sum of two real numbers in scaled representation.
    Computes sign of (A * BASE**IA + B * BASE**IB) without overflow.
    Returns 1, 0, or -1.
    """

    def test_both_positive_same_exponent(self):
        """Test with both positive, same exponent"""
        a, ia = 2.0, 5
        b, ib = 3.0, 5
        result = ma01cd(a, ia, b, ib)
        assert result == 1  # 2 + 3 > 0

    def test_both_negative_same_exponent(self):
        """Test with both negative, same exponent"""
        a, ia = -2.0, 5
        b, ib = -3.0, 5
        result = ma01cd(a, ia, b, ib)
        assert result == -1  # -2 + -3 < 0

    def test_cancel_same_exponent(self):
        """Test cancellation with same exponent"""
        a, ia = 5.0, 3
        b, ib = -5.0, 3
        result = ma01cd(a, ia, b, ib)
        assert result == 0  # 5 + (-5) = 0

    def test_different_exponents_positive(self):
        """Test with different exponents, positive result"""
        a, ia = 1.0, 10  # 1 * 10^10
        b, ib = 1.0, 5   # 1 * 10^5
        result = ma01cd(a, ia, b, ib)
        assert result == 1  # Large positive + small positive = positive

    def test_different_exponents_negative(self):
        """Test with different exponents, negative result"""
        a, ia = -1.0, 10
        b, ib = 1.0, 5
        result = ma01cd(a, ia, b, ib)
        assert result == -1  # Large negative dominates

    def test_opposite_signs_larger_positive(self):
        """Test opposite signs with larger positive"""
        a, ia = 2.0, 5
        b, ib = -1.0, 5
        result = ma01cd(a, ia, b, ib)
        assert result == 1

    def test_opposite_signs_larger_negative(self):
        """Test opposite signs with larger negative"""
        a, ia = 1.0, 5
        b, ib = -2.0, 5
        result = ma01cd(a, ia, b, ib)
        assert result == -1

    def test_both_zero(self):
        """Test with both numbers zero"""
        a, ia = 0.0, 0
        b, ib = 0.0, 0
        result = ma01cd(a, ia, b, ib)
        assert result == 0

    def test_one_zero_positive(self):
        """Test with one zero, one positive"""
        a, ia = 0.0, 0
        b, ib = 5.0, 3
        result = ma01cd(a, ia, b, ib)
        assert result == 1

    def test_one_zero_negative(self):
        """Test with one zero, one negative"""
        a, ia = -5.0, 3
        b, ib = 0.0, 0
        result = ma01cd(a, ia, b, ib)
        assert result == -1


class TestMA01DD:
    """
    Test MA01DD - Approximate symmetric chordal metric for complex numbers.
    Computes D = MIN(|A1 - A2|, |1/A1 - 1/A2|) for A1 = AR1+i*AI1,
    A2 = AR2+i*AI2.
    """

    def test_same_numbers(self):
        """Test chordal metric of identical numbers"""
        ar1, ai1 = 3.0, 4.0
        ar2, ai2 = 3.0, 4.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemin)
        assert_allclose(d, 0.0, atol=1e-14)

    def test_real_numbers(self):
        """Test with purely real numbers"""
        ar1, ai1 = 1.0, 0.0
        ar2, ai2 = 3.0, 0.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemin)

        # Direct distance: |1 - 3| = 2
        # Reciprocal distance: |1/1 - 1/3| = 2/3
        expected = min(2.0, 2.0/3.0)
        assert_allclose(d, expected, rtol=1e-10)

    def test_complex_numbers(self):
        """Test with general complex numbers"""
        ar1, ai1 = 1.0, 1.0
        ar2, ai2 = 2.0, 2.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemin)

        # Should return a positive metric
        assert d > 0.0
        assert np.isfinite(d)

    def test_one_zero(self):
        """Test when one number is zero"""
        ar1, ai1 = 0.0, 0.0
        ar2, ai2 = 1.0, 1.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemin)

        # Distance should be |A2| = sqrt(2)
        expected = np.sqrt(2.0)
        assert_allclose(d, expected, rtol=1e-10)

    def test_both_zero(self):
        """Test when both numbers are zero"""
        ar1, ai1 = 0.0, 0.0
        ar2, ai2 = 0.0, 0.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemin)
        assert_allclose(d, 0.0, atol=1e-14)

    def test_symmetry(self):
        """Test that metric is symmetric"""
        ar1, ai1 = 1.0, 2.0
        ar2, ai2 = 3.0, 4.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1 = ma01dd(ar1, ai1, ar2, ai2, eps, safemin)
        d2 = ma01dd(ar2, ai2, ar1, ai1, eps, safemin)

        assert_allclose(d1, d2, rtol=1e-14)


class TestMA01DZ:
    """
    Test MA01DZ - Approximate symmetric chordal metric in rational form.
    Computes D = MIN(|A1 - A2|, |1/A1 - 1/A2|) for A1 = (AR1+i*AI1)/B1,
    A2 = (AR2+i*AI2)/B2.
    Returns D1/D2 where D2 is 0 or 1.
    """

    def test_same_numbers(self):
        """Test chordal metric of identical numbers"""
        ar1, ai1, b1 = 3.0, 4.0, 1.0
        ar2, ai2, b2 = 3.0, 4.0, 1.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 0
        assert_allclose(d1, 0.0, atol=1e-14)

    def test_finite_numbers(self):
        """Test with finite representable numbers"""
        ar1, ai1, b1 = 1.0, 0.0, 1.0
        ar2, ai2, b2 = 2.0, 0.0, 1.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 0
        assert d2 == 1.0
        # Metric should be min(|1-2|, |1-1/2|) = min(1, 0.5) = 0.5
        assert_allclose(d1, 0.5, rtol=1e-10)

    def test_one_infinite_b_zero(self):
        """Test when one number is infinite (B=0, numerator nonzero)"""
        ar1, ai1, b1 = 1.0, 1.0, 0.0  # Infinite
        ar2, ai2, b2 = 1.0, 0.0, 1.0  # Finite
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 0
        # Should return a finite metric
        assert np.isfinite(d1)

    def test_both_infinite(self):
        """Test when both numbers are infinite"""
        ar1, ai1, b1 = 1.0, 0.0, 0.0
        ar2, ai2, b2 = 0.0, 1.0, 0.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 0
        # Metric should be zero (both at infinity)
        msg = "Both infinite should give D1=0, D2=1"
        assert_allclose(d1, 0.0, atol=1e-14, err_msg=msg)
        assert d2 == 1.0

    def test_nan_first(self):
        """Test when first number is NaN (all zeros)"""
        ar1, ai1, b1 = 0.0, 0.0, 0.0  # NaN
        ar2, ai2, b2 = 1.0, 1.0, 1.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 1
        assert_allclose(d1, 0.0, atol=1e-14)
        assert_allclose(d2, 0.0, atol=1e-14)

    def test_nan_second(self):
        """Test when second number is NaN"""
        ar1, ai1, b1 = 1.0, 1.0, 1.0
        ar2, ai2, b2 = 0.0, 0.0, 0.0  # NaN
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 1
        assert_allclose(d1, 0.0, atol=1e-14)
        assert_allclose(d2, 0.0, atol=1e-14)

    def test_both_nan(self):
        """Test when both numbers are NaN"""
        ar1, ai1, b1 = 0.0, 0.0, 0.0
        ar2, ai2, b2 = 0.0, 0.0, 0.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 1
        assert_allclose(d1, 0.0, atol=1e-14)
        assert_allclose(d2, 0.0, atol=1e-14)

    def test_symmetry(self):
        """Test that metric is symmetric"""
        ar1, ai1, b1 = 1.0, 2.0, 2.0
        ar2, ai2, b2 = 3.0, 4.0, 3.0
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1_fwd, d2_fwd, iwarn_fwd = ma01dz(
            ar1, ai1, b1, ar2, ai2, b2, eps, safemin
        )
        d1_rev, d2_rev, iwarn_rev = ma01dz(
            ar2, ai2, b2, ar1, ai1, b1, eps, safemin
        )

        assert iwarn_fwd == 0
        assert iwarn_rev == 0
        assert d2_fwd == d2_rev
        if d2_fwd != 0:
            assert_allclose(d1_fwd / d2_fwd, d1_rev / d2_rev, rtol=1e-14)

    def test_scaled_numbers(self):
        """Test with scaled rational representations"""
        ar1, ai1, b1 = 2.0, 4.0, 2.0  # Represents 1 + 2j
        ar2, ai2, b2 = 6.0, 9.0, 3.0  # Represents 2 + 3j
        eps = np.finfo(float).eps
        safemin = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemin)

        assert iwarn == 0
        assert d2 == 1.0
        # Should match ma01dd for (1+2j) and (2+3j)
        d_direct = ma01dd(1.0, 2.0, 2.0, 3.0, eps, safemin)
        assert_allclose(d1, d_direct, rtol=1e-10)
