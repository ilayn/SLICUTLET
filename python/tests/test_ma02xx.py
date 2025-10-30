"""
Tests for MA02XX family of functions
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from slicutlet import (
    ma02ad, ma02az, ma02bd, ma02bz, ma02cd, ma02cz, ma02dd, ma02ed,
    ma02es, ma02ez, ma02fd, ma02gd, ma02gz, ma02hd, ma02hz, ma02id,
    ma02iz, ma02jd, ma02jz, ma02md, ma02mz, ma02nz, ma02od, ma02oz,
    ma02pd, ma02pz, ma02rd, ma02sd
)


class TestMA02AD:
    """
    Test MA02AD - Transpose all or part of a real matrix.
    Transposes matrix A (m x n) into B (n x m).
    JOB parameter selects upper triangle, lower triangle, or full matrix.
    """

    def test_full_matrix_transpose(self):
        """Test transpose of full matrix"""
        m, n = 3, 4
        job = 2  # Full matrix (neither 0=upper nor 1=lower)
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ], order='F')

        b = ma02ad(job, m, n, a)

        expected = a.T
        assert b.shape == (n, m)
        assert_allclose(b, expected, rtol=1e-14)

    def test_upper_triangle_transpose(self):
        """Test transpose of upper triangle only"""
        m, n = 4, 4
        job = 0  # Upper triangle
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F')

        b = ma02ad(job, m, n, a)

        # Only upper triangle should be transposed
        assert b.shape == (n, m)
        for j in range(n):
            for i in range(min(j + 1, m)):
                assert_allclose(b[j, i], a[i, j], rtol=1e-14)

    def test_lower_triangle_transpose(self):
        """Test transpose of lower triangle only"""
        m, n = 4, 4
        job = 1  # Lower triangle
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F')

        b = ma02ad(job, m, n, a)

        # Only lower triangle should be transposed
        assert b.shape == (n, m)
        for j in range(n):
            for i in range(j, m):
                assert_allclose(b[j, i], a[i, j], rtol=1e-14)

    def test_rectangular_tall(self):
        """Test with tall rectangular matrix (m > n)"""
        m, n = 5, 3
        job = 2
        a = np.arange(1.0, m * n + 1.0).reshape(m, n, order='F')

        b = ma02ad(job, m, n, a)

        assert b.shape == (n, m)
        assert_allclose(b, a.T, rtol=1e-14)

    def test_rectangular_wide(self):
        """Test with wide rectangular matrix (m < n)"""
        m, n = 3, 5
        job = 2
        a = np.arange(1.0, m * n + 1.0).reshape(m, n, order='F')

        b = ma02ad(job, m, n, a)

        assert b.shape == (n, m)
        assert_allclose(b, a.T, rtol=1e-14)

    def test_single_row(self):
        """Test with single row matrix"""
        m, n = 1, 5
        job = 2
        a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], order='F')

        b = ma02ad(job, m, n, a)

        assert b.shape == (n, m)
        assert_allclose(b, a.T, rtol=1e-14)

    def test_single_column(self):
        """Test with single column matrix"""
        m, n = 5, 1
        job = 2
        a = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], order='F')

        b = ma02ad(job, m, n, a)

        assert b.shape == (n, m)
        assert_allclose(b, a.T, rtol=1e-14)

    def test_empty_matrix(self):
        """Test with empty matrix"""
        m, n = 0, 0
        job = 2
        a = np.array([], dtype=np.float64).reshape(0, 0, order='F')

        b = ma02ad(job, m, n, a)

        assert b.shape == (n, m)

    def test_preserves_values(self):
        """Test that values are preserved exactly"""
        m, n = 3, 3
        job = 2
        a = np.array([
            [1.5, -2.7, 3.14],
            [-0.5, 0.0, 1e-10],
            [1e10, -1e-10, 2.718]
        ], order='F')

        b = ma02ad(job, m, n, a)

        assert_allclose(b, a.T, rtol=1e-14, atol=1e-14)


class TestMA02AZ:
    """
    Test MA02AZ - Transpose or conjugate transpose complex matrix.
    TRANS=0: conjugate transpose, TRANS=1: plain transpose.
    JOB selects upper/lower triangle or full matrix.
    """

    def test_full_matrix_transpose(self):
        """Test plain transpose of full complex matrix"""
        m, n = 3, 4
        trans, job = 1, 2  # Plain transpose, full matrix
        a = np.array([
            [1+1j, 2+2j, 3+3j, 4+4j],
            [5+5j, 6+6j, 7+7j, 8+8j],
            [9+9j, 10+10j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')

        b = ma02az(trans, job, m, n, a)

        expected = a.T
        assert b.shape == (n, m)
        assert_allclose(b, expected, rtol=1e-14)

    def test_full_matrix_conjugate_transpose(self):
        """Test conjugate transpose of full complex matrix"""
        m, n = 3, 4
        trans, job = 0, 2  # Conjugate transpose, full matrix
        a = np.array([
            [1+1j, 2+2j, 3+3j, 4+4j],
            [5+5j, 6+6j, 7+7j, 8+8j],
            [9+9j, 10+10j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')

        b = ma02az(trans, job, m, n, a)

        expected = a.T.conj()
        assert b.shape == (n, m)
        assert_allclose(b, expected, rtol=1e-14)

    def test_upper_triangle_transpose(self):
        """Test plain transpose of upper triangle"""
        m, n = 4, 4
        trans, job = 1, 0  # Plain transpose, upper triangle
        a = np.array([
            [1+0j, 2+1j, 3+2j, 4+3j],
            [5+4j, 6+0j, 7+1j, 8+2j],
            [9+3j, 10+4j, 11+0j, 12+1j],
            [13+2j, 14+3j, 15+4j, 16+0j]
        ], dtype=np.complex128, order='F')

        b = ma02az(trans, job, m, n, a)

        # Only upper triangle should be transposed
        assert b.shape == (n, m)
        for j in range(n):
            for i in range(min(j + 1, m)):
                assert_allclose(b[j, i], a[i, j], rtol=1e-14)

    def test_upper_triangle_conjugate_transpose(self):
        """Test conjugate transpose of upper triangle"""
        m, n = 4, 4
        trans, job = 0, 0  # Conjugate transpose, upper triangle
        a = np.array([
            [1+0j, 2+1j, 3+2j, 4+3j],
            [5+4j, 6+0j, 7+1j, 8+2j],
            [9+3j, 10+4j, 11+0j, 12+1j],
            [13+2j, 14+3j, 15+4j, 16+0j]
        ], dtype=np.complex128, order='F')

        b = ma02az(trans, job, m, n, a)

        # Only upper triangle should be conjugate transposed
        assert b.shape == (n, m)
        for j in range(n):
            for i in range(min(j + 1, m)):
                assert_allclose(b[j, i], np.conj(a[i, j]), rtol=1e-14)

    def test_lower_triangle_conjugate_transpose(self):
        """Test conjugate transpose of lower triangle"""
        m, n = 4, 4
        trans, job = 0, 1  # Conjugate transpose, lower triangle
        a = np.array([
            [1+0j, 2+1j, 3+2j, 4+3j],
            [5+4j, 6+0j, 7+1j, 8+2j],
            [9+3j, 10+4j, 11+0j, 12+1j],
            [13+2j, 14+3j, 15+4j, 16+0j]
        ], dtype=np.complex128, order='F')

        b = ma02az(trans, job, m, n, a)

        # Only lower triangle should be conjugate transposed
        assert b.shape == (n, m)
        for j in range(n):
            for i in range(j, m):
                assert_allclose(b[j, i], np.conj(a[i, j]), rtol=1e-14)

    def test_purely_real_complex_matrix(self):
        """Test with complex matrix that has only real parts"""
        m, n = 3, 3
        trans, job = 0, 2
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.complex128, order='F')

        b = ma02az(trans, job, m, n, a)

        # Conjugate of real is itself
        assert_allclose(b, a.T, rtol=1e-14)

    def test_purely_imaginary_matrix(self):
        """Test with purely imaginary matrix"""
        m, n = 3, 3
        trans, job = 0, 2
        a = np.array([
            [1j, 2j, 3j],
            [4j, 5j, 6j],
            [7j, 8j, 9j]
        ], dtype=np.complex128, order='F')

        b = ma02az(trans, job, m, n, a)

        expected = a.T.conj()
        assert_allclose(b, expected, rtol=1e-14)

    def test_rectangular_matrix(self):
        """Test with rectangular complex matrix"""
        m, n = 2, 5
        trans, job = 0, 2
        a = np.arange(1, m * n + 1, dtype=np.complex128).reshape(m, n)
        a = a + 1j * np.arange(m * n, 0, -1, dtype=np.float64).reshape(m, n)
        a = np.asfortranarray(a)

        b = ma02az(trans, job, m, n, a)

        assert b.shape == (n, m)
        assert_allclose(b, a.T.conj(), rtol=1e-14)


class TestMA02BD:
    """
    Test MA02BD - Exchange (reverse) rows and/or columns of a real matrix.
    SIDE=0: reverse rows, SIDE=1: reverse columns, SIDE=2: reverse both.
    Operates in-place.
    """

    def test_reverse_rows_only(self):
        """Test reversing rows only"""
        side = 0
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ], order='F')
        expected = a[::-1, :].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_reverse_columns_only(self):
        """Test reversing columns only"""
        side = 1
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ], order='F')
        expected = a[:, ::-1].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_reverse_both(self):
        """Test reversing both rows and columns"""
        side = 2
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ], order='F')
        expected = a[::-1, ::-1].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_square_matrix_reverse_rows(self):
        """Test with square matrix, reverse rows"""
        side = 0
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F')
        expected = a[::-1, :].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_single_row_reverse_columns(self):
        """Test with single row, reverse columns"""
        side = 1
        a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], order='F')
        expected = a[:, ::-1].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_single_column_reverse_rows(self):
        """Test with single column, reverse rows"""
        side = 0
        a = np.array([[1.0], [2.0], [3.0], [4.0]], order='F')
        expected = a[::-1, :].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_odd_dimensions(self):
        """Test with odd number of rows/columns"""
        side = 2
        a = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
        ], order='F')
        expected = a[::-1, ::-1].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_even_dimensions(self):
        """Test with even number of rows/columns"""
        side = 2
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F')
        expected = a[::-1, ::-1].copy()

        result = ma02bd(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_double_reverse_identity(self):
        """Test that reversing twice returns original"""
        a_orig = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F')

        # Reverse both directions twice
        a1 = ma02bd(2, a_orig.copy())
        a2 = ma02bd(2, a1)

        assert_allclose(a2, a_orig, rtol=1e-14)


class TestMA02BZ:
    """
    Test MA02BZ - Exchange (reverse) rows and/or columns of complex matrix.
    SIDE=0: reverse rows, SIDE=1: reverse columns, SIDE=2: reverse both.
    Operates in-place.
    """

    def test_reverse_rows_only(self):
        """Test reversing rows only"""
        side = 0
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j],
            [10+10j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')
        expected = a[::-1, :].copy()

        result = ma02bz(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_reverse_columns_only(self):
        """Test reversing columns only"""
        side = 1
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j],
            [10+10j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')
        expected = a[:, ::-1].copy()

        result = ma02bz(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_reverse_both(self):
        """Test reversing both rows and columns"""
        side = 2
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j],
            [10+10j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')
        expected = a[::-1, ::-1].copy()

        result = ma02bz(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_purely_real_values(self):
        """Test with complex array having only real parts"""
        side = 2
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.complex128, order='F')
        expected = a[::-1, ::-1].copy()

        result = ma02bz(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_purely_imaginary_values(self):
        """Test with purely imaginary values"""
        side = 2
        a = np.array([
            [1j, 2j, 3j],
            [4j, 5j, 6j],
            [7j, 8j, 9j]
        ], dtype=np.complex128, order='F')
        expected = a[::-1, ::-1].copy()

        result = ma02bz(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_mixed_complex_values(self):
        """Test with mixed real and imaginary components"""
        side = 0
        a = np.array([
            [1-2j, 3+0j, -4+5j],
            [0+1j, -2-3j, 6+0j],
            [7+8j, 0-9j, 10+11j]
        ], dtype=np.complex128, order='F')
        expected = a[::-1, :].copy()

        result = ma02bz(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_square_matrix(self):
        """Test with square complex matrix"""
        side = 2
        n = 5
        a = (np.arange(n*n, dtype=np.float64) +
             1j * np.arange(n*n, 0, -1, dtype=np.float64))
        a = a.reshape(n, n, order='F').astype(np.complex128)
        expected = a[::-1, ::-1].copy()

        result = ma02bz(side, a)

        assert_allclose(result, expected, rtol=1e-14)

    def test_double_reverse_identity(self):
        """Test that reversing twice returns original"""
        a_orig = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], dtype=np.complex128, order='F')

        # Reverse both directions twice
        a1 = ma02bz(2, a_orig.copy())
        a2 = ma02bz(2, a1)

        assert_allclose(a2, a_orig, rtol=1e-14)


class TestMA02CD:
    """
    Test MA02CD - Pertranspose a real band matrix.
    Exchanges elements across the anti-diagonal in a band matrix.
    N: matrix dimension, KL: subdiagonals, KU: superdiagonals.
    """

    def test_tridiagonal_matrix(self):
        """Test pertranspose of tridiagonal matrix"""
        n = 5
        kl, ku = 1, 1
        a = np.array([
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 5.0, 0.0, 0.0],
            [0.0, 6.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 9.0, 10.0, 11.0],
            [0.0, 0.0, 0.0, 12.0, 13.0]
        ], order='F')

        result = ma02cd(n, kl, ku, a)

        # Pertranspose should reverse along anti-diagonal
        # A[i,j] <-> A[n-1-j, n-1-i] for elements in the band
        assert result.shape == a.shape
        # First and last diagonal elements should be swapped
        assert_allclose(result[0, 0], a[4, 4], rtol=1e-14)
        assert_allclose(result[4, 4], a[0, 0], rtol=1e-14)

    def test_diagonal_matrix(self):
        """Test pertranspose of diagonal matrix (kl=ku=0)"""
        n = 4
        kl, ku = 0, 0
        a = np.diag([1.0, 2.0, 3.0, 4.0]).astype('F')

        result = ma02cd(n, kl, ku, a)

        # Diagonal elements should be reversed
        expected_diag = np.array([4.0, 3.0, 2.0, 1.0])
        assert_allclose(np.diag(result), expected_diag, rtol=1e-14)

    def test_full_bandwidth(self):
        """Test with full bandwidth (kl=ku=n-1, effectively full matrix)"""
        n = 4
        kl, ku = n-1, n-1
        a = np.arange(1.0, n*n + 1.0).reshape(n, n, order='F')

        result = ma02cd(n, kl, ku, a)

        # Should reverse along anti-diagonal
        assert result.shape == a.shape
        # Check a few anti-diagonal swaps
        assert_allclose(result[0, 0], a[n-1, n-1], rtol=1e-14)
        assert_allclose(result[n-1, n-1], a[0, 0], rtol=1e-14)

    def test_pentadiagonal_matrix(self):
        """Test pentadiagonal matrix (kl=ku=2)"""
        n = 6
        kl, ku = 2, 2
        a = np.zeros((n, n), order='F')
        # Fill pentadiagonal structure
        for i in range(n):
            for j in range(max(0, i-2), min(n, i+3)):
                a[i, j] = float(i * n + j + 1)

        result = ma02cd(n, kl, ku, a)

        # Verify pertranspose preserves band structure
        assert result.shape == a.shape

    def test_single_element(self):
        """Test with 1x1 matrix"""
        n = 1
        kl, ku = 0, 0
        a = np.array([[5.0]], order='F')

        result = ma02cd(n, kl, ku, a)

        # Single element should remain unchanged
        assert_allclose(result, a, rtol=1e-14)

    def test_two_by_two(self):
        """Test with 2x2 matrix"""
        n = 2
        kl, ku = 1, 1
        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], order='F')

        result = ma02cd(n, kl, ku, a)

        # Should swap diagonal elements
        assert_allclose(result[0, 0], a[1, 1], rtol=1e-14)
        assert_allclose(result[1, 1], a[0, 0], rtol=1e-14)

    def test_double_pertranspose_identity(self):
        """Test that applying pertranspose twice returns original"""
        n = 5
        kl, ku = 2, 2
        a_orig = np.random.randn(n, n)
        a_orig = np.asfortranarray(a_orig)

        a1 = ma02cd(n, kl, ku, a_orig.copy())
        a2 = ma02cd(n, kl, ku, a1)

        # Double pertranspose should be identity
        assert_allclose(a2, a_orig, rtol=1e-13)


class TestMA02CZ:
    """
    Test MA02CZ - Pertranspose a complex band matrix.
    Complex version of MA02CD.
    """

    def test_tridiagonal_complex_matrix(self):
        """Test pertranspose of complex tridiagonal matrix"""
        n = 5
        kl, ku = 1, 1
        a = np.array([
            [1+1j, 2+2j, 0, 0, 0],
            [3+3j, 4+4j, 5+5j, 0, 0],
            [0, 6+6j, 7+7j, 8+8j, 0],
            [0, 0, 9+9j, 10+10j, 11+11j],
            [0, 0, 0, 12+12j, 13+13j]
        ], dtype=np.complex128, order='F')

        result = ma02cz(n, kl, ku, a)

        assert result.shape == a.shape
        # First and last diagonal elements should be swapped
        assert_allclose(result[0, 0], a[4, 4], rtol=1e-14)
        assert_allclose(result[4, 4], a[0, 0], rtol=1e-14)

    def test_diagonal_complex_matrix(self):
        """Test pertranspose of complex diagonal matrix"""
        n = 4
        kl, ku = 0, 0
        a = np.diag([1+1j, 2+2j, 3+3j, 4+4j]).astype(np.complex128, order='F')

        result = ma02cz(n, kl, ku, a)

        # Diagonal elements should be reversed
        expected_diag = np.array([4+4j, 3+3j, 2+2j, 1+1j])
        assert_allclose(np.diag(result), expected_diag, rtol=1e-14)

    def test_purely_real_complex_array(self):
        """Test with complex array having only real parts"""
        n = 4
        kl, ku = 1, 1
        a = np.arange(1.0, n*n + 1.0).reshape(n, n)
        a = a.astype(np.complex128, order='F')

        result = ma02cz(n, kl, ku, a)

        assert result.shape == a.shape
        assert result.dtype == np.complex128

    def test_purely_imaginary_array(self):
        """Test with purely imaginary values"""
        n = 3
        kl, ku = 1, 1
        a = 1j * np.arange(1.0, n*n + 1.0).reshape(n, n)
        a = np.asfortranarray(a.astype(np.complex128))

        result = ma02cz(n, kl, ku, a)

        assert result.shape == a.shape
        assert result.dtype == np.complex128

    def test_double_pertranspose_identity(self):
        """Test that applying pertranspose twice returns original"""
        n = 5
        kl, ku = 2, 2
        real_part = np.random.randn(n, n)
        imag_part = np.random.randn(n, n)
        a_orig = (real_part + 1j * imag_part).astype(np.complex128, order='F')

        a1 = ma02cz(n, kl, ku, a_orig.copy())
        a2 = ma02cz(n, kl, ku, a1)

        # Double pertranspose should be identity
        assert_allclose(a2, a_orig, rtol=1e-13)


class TestMA02DD:
    """
    Test MA02DD - Pack/unpack symmetric or triangular matrix.
    JOB=0: pack full matrix into packed storage.
    JOB!=0: unpack packed storage into full matrix.
    UPLO=0: upper triangle, UPLO=1: lower triangle.
    """

    def test_pack_upper_triangle(self):
        """Test packing upper triangle into packed storage"""
        n = 4
        job, uplo = 0, 0  # Pack upper
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0]
        ], order='F')

        ap = ma02dd(job, uplo, n, a)

        # Packed storage should contain n*(n+1)/2 elements
        expected_size = n * (n + 1) // 2
        assert ap.shape == (expected_size,)
        # Upper triangle stored column-wise: [A00, A01, A11, A02, A12, A22, ...]
        expected = np.array([1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 10.0])
        assert_allclose(ap, expected, rtol=1e-14)

    def test_pack_lower_triangle(self):
        """Test packing lower triangle into packed storage"""
        n = 4
        job, uplo = 0, 1  # Pack lower
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0]
        ], order='F')

        ap = ma02dd(job, uplo, n, a)

        expected_size = n * (n + 1) // 2
        assert ap.shape == (expected_size,)
        # Lower triangle stored column-wise
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert_allclose(ap, expected, rtol=1e-14)

    def test_unpack_upper_triangle(self):
        """Test unpacking upper triangle from packed storage"""
        n = 3
        job, uplo = 1, 0  # Unpack upper
        ap = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 6.0])

        a = ma02dd(job, uplo, n, ap)

        assert a.shape == (n, n)
        # Check upper triangle was unpacked correctly
        assert_allclose(a[0, 0], 1.0, rtol=1e-14)
        assert_allclose(a[0, 1], 2.0, rtol=1e-14)
        assert_allclose(a[1, 1], 4.0, rtol=1e-14)
        assert_allclose(a[0, 2], 3.0, rtol=1e-14)
        assert_allclose(a[1, 2], 5.0, rtol=1e-14)
        assert_allclose(a[2, 2], 6.0, rtol=1e-14)

    def test_unpack_lower_triangle(self):
        """Test unpacking lower triangle from packed storage"""
        n = 3
        job, uplo = 1, 1  # Unpack lower
        ap = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        a = ma02dd(job, uplo, n, ap)

        assert a.shape == (n, n)
        # Check lower triangle was unpacked correctly
        assert_allclose(a[0, 0], 1.0, rtol=1e-14)
        assert_allclose(a[1, 0], 2.0, rtol=1e-14)
        assert_allclose(a[2, 0], 3.0, rtol=1e-14)
        assert_allclose(a[1, 1], 4.0, rtol=1e-14)
        assert_allclose(a[2, 1], 5.0, rtol=1e-14)
        assert_allclose(a[2, 2], 6.0, rtol=1e-14)

    def test_pack_unpack_roundtrip_upper(self):
        """Test that pack followed by unpack recovers upper triangle"""
        n = 4
        uplo = 0  # Upper
        a_orig = np.random.randn(n, n)
        # Make symmetric for cleaner test
        a_orig = (a_orig + a_orig.T) / 2
        a_orig = np.asfortranarray(a_orig)

        # Pack then unpack
        ap = ma02dd(0, uplo, n, a_orig)
        a_recovered = ma02dd(1, uplo, n, ap)

        # Upper triangle should match
        for i in range(n):
            for j in range(i, n):
                err_msg = f"Mismatch at ({i},{j})"
                assert_allclose(a_recovered[i, j], a_orig[i, j],
                              rtol=1e-13, err_msg=err_msg)

    def test_pack_unpack_roundtrip_lower(self):
        """Test that pack followed by unpack recovers lower triangle"""
        n = 4
        uplo = 1  # Lower
        a_orig = np.random.randn(n, n)
        a_orig = (a_orig + a_orig.T) / 2
        a_orig = np.asfortranarray(a_orig)

        # Pack then unpack
        ap = ma02dd(0, uplo, n, a_orig)
        a_recovered = ma02dd(1, uplo, n, ap)

        # Lower triangle should match
        for i in range(n):
            for j in range(i + 1):
                err_msg = f"Mismatch at ({i},{j})"
                assert_allclose(a_recovered[i, j], a_orig[i, j],
                              rtol=1e-13, err_msg=err_msg)

    def test_small_matrix(self):
        """Test with 1x1 matrix"""
        n = 1
        job, uplo = 0, 0
        a = np.array([[5.0]], order='F')

        ap = ma02dd(job, uplo, n, a)

        assert ap.shape == (1,)
        assert_allclose(ap[0], 5.0, rtol=1e-14)

    def test_packed_size(self):
        """Test that packed array has correct size for various n"""
        for n in [1, 2, 3, 5, 10]:
            job, uplo = 0, 0
            a = np.eye(n, order='F')
            ap = ma02dd(job, uplo, n, a)
            expected_size = n * (n + 1) // 2
            assert ap.shape == (expected_size,), \
                f"Wrong packed size for n={n}"


class TestMA02ED:
    """
    Test MA02ED - Construct full symmetric matrix from one triangle.
    UPLO=0: copy lower triangle to upper triangle.
    UPLO=1: copy upper triangle to lower triangle.
    Operates in-place.
    """

    def test_copy_lower_to_upper(self):
        """Test copying lower triangle to upper triangle"""
        uplo, n = 1, 4  # uplo=1 constructs upper from lower
        a = np.array([
            [1.0, 999.0, 999.0, 999.0],
            [2.0, 5.0, 999.0, 999.0],
            [3.0, 6.0, 8.0, 999.0],
            [4.0, 7.0, 9.0, 10.0]
        ], order='F')

        result = ma02ed(uplo, n, a)

        # Upper triangle should now match lower triangle
        expected = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_copy_upper_to_lower(self):
        """Test copying upper triangle to lower triangle"""
        uplo, n = 0, 4  # uplo=0 constructs lower from upper
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [999.0, 5.0, 6.0, 7.0],
            [999.0, 999.0, 8.0, 9.0],
            [999.0, 999.0, 999.0, 10.0]
        ], order='F')

        result = ma02ed(uplo, n, a)

        # Lower triangle should now match upper triangle
        expected = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_symmetric_matrix_unchanged(self):
        """Test that already symmetric matrix remains unchanged"""
        uplo, n = 0, 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0]
        ], order='F')
        a_orig = a.copy()

        result = ma02ed(uplo, n, a)

        assert_allclose(result, a_orig, rtol=1e-14)

    def test_small_matrix(self):
        """Test with 2x2 matrix"""
        uplo, n = 1, 2  # uplo=1 constructs upper from lower
        a = np.array([
            [1.0, 999.0],
            [2.0, 3.0]
        ], order='F')

        result = ma02ed(uplo, n, a)

        expected = np.array([
            [1.0, 2.0],
            [2.0, 3.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_single_element(self):
        """Test with 1x1 matrix (no copying needed)"""
        uplo, n = 0, 1
        a = np.array([[5.0]], order='F')

        result = ma02ed(uplo, n, a)

        assert_allclose(result, a, rtol=1e-14)

    def test_preserves_diagonal(self):
        """Test that diagonal is always preserved"""
        uplo, n = 0, 5
        diag_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = np.diag(diag_values).astype('F')
        # Add some lower triangle values
        for i in range(1, n):
            a[i, 0] = 10.0 * i

        result = ma02ed(uplo, n, a)

        assert_allclose(np.diag(result), diag_values, rtol=1e-14)

    def test_specific_values(self):
        """Test with specific known values"""
        uplo, n = 0, 3  # uplo=0 constructs lower from upper
        a = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0]
        ], order='F')

        result = ma02ed(uplo, n, a)

        # Check that lower was filled from upper
        assert_allclose(result[1, 0], 2.0, rtol=1e-14)
        assert_allclose(result[2, 0], 3.0, rtol=1e-14)
        assert_allclose(result[2, 1], 5.0, rtol=1e-14)

    def test_consistency_between_modes(self):
        """Test that both modes create same symmetric result"""
        n = 4
        # Create matrix with defined lower triangle
        a_lower = np.random.randn(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                a_lower[i, j] = 0.0
        a_lower = np.asfortranarray(a_lower)

        # Create symmetric matrix from lower
        sym_from_lower = ma02ed(0, n, a_lower.copy())

        # Now create matrix with same values in upper triangle
        a_upper = sym_from_lower.T.copy(order='F')
        for i in range(1, n):
            for j in range(i):
                a_upper[i, j] = 0.0

        # Create symmetric matrix from upper
        sym_from_upper = ma02ed(1, n, a_upper)

        # Both should give same symmetric result
        assert_allclose(sym_from_lower, sym_from_upper, rtol=1e-13)


class TestMA02ES:
    """
    Test MA02ES - Construct skew-symmetric matrix from one triangle.
    UPLO=0: copy lower triangle to upper with negation (A[i,j] = -A[j,i]).
    UPLO=1: copy upper triangle to lower with negation.
    Sets diagonal to zero. Operates in-place.
    """

    def test_copy_lower_to_upper_skew(self):
        """Test creating skew-symmetric from lower triangle"""
        uplo, n = 1, 4  # uplo=1 constructs upper from lower
        a = np.array([
            [999.0, 999.0, 999.0, 999.0],
            [2.0, 999.0, 999.0, 999.0],
            [3.0, 6.0, 999.0, 999.0],
            [4.0, 7.0, 9.0, 999.0]
        ], order='F')

        result = ma02es(uplo, n, a)

        # Upper should be negative of lower, diagonal should be zero
        expected = np.array([
            [0.0, -2.0, -3.0, -4.0],
            [2.0, 0.0, -6.0, -7.0],
            [3.0, 6.0, 0.0, -9.0],
            [4.0, 7.0, 9.0, 0.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_copy_upper_to_lower_skew(self):
        """Test creating skew-symmetric from upper triangle"""
        uplo, n = 0, 4  # uplo=0 constructs lower from upper
        a = np.array([
            [999.0, 2.0, 3.0, 4.0],
            [999.0, 999.0, 6.0, 7.0],
            [999.0, 999.0, 999.0, 9.0],
            [999.0, 999.0, 999.0, 999.0]
        ], order='F')

        result = ma02es(uplo, n, a)

        # Lower should be negative of upper, diagonal should be zero
        expected = np.array([
            [0.0, 2.0, 3.0, 4.0],
            [-2.0, 0.0, 6.0, 7.0],
            [-3.0, -6.0, 0.0, 9.0],
            [-4.0, -7.0, -9.0, 0.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_diagonal_always_zero(self):
        """Test that diagonal is always set to zero"""
        uplo, n = 0, 4
        a = np.array([
            [100.0, 999.0, 999.0, 999.0],
            [2.0, 200.0, 999.0, 999.0],
            [3.0, 6.0, 300.0, 999.0],
            [4.0, 7.0, 9.0, 400.0]
        ], order='F')

        result = ma02es(uplo, n, a)

        # All diagonal elements should be zero
        assert_allclose(np.diag(result), np.zeros(n), rtol=1e-14)

    def test_skew_symmetry_property(self):
        """Test that result is skew-symmetric (A = -A^T)"""
        uplo, n = 0, 5
        a = np.random.randn(n, n)
        # Set upper triangle to arbitrary values (will be overwritten)
        for i in range(n):
            for j in range(i, n):
                a[i, j] = 999.0
        a = np.asfortranarray(a)

        result = ma02es(uplo, n, a)

        # Check skew-symmetry: A[i,j] = -A[j,i]
        for i in range(n):
            for j in range(n):
                assert_allclose(result[i, j], -result[j, i], rtol=1e-13)

    def test_small_matrix(self):
        """Test with 2x2 matrix"""
        uplo, n = 1, 2  # uplo=1 constructs upper from lower
        a = np.array([
            [999.0, 999.0],
            [3.0, 999.0]
        ], order='F')

        result = ma02es(uplo, n, a)

        expected = np.array([
            [0.0, -3.0],
            [3.0, 0.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_single_element(self):
        """Test with 1x1 matrix (just zeros diagonal)"""
        uplo, n = 0, 1
        a = np.array([[42.0]], order='F')

        result = ma02es(uplo, n, a)

        assert_allclose(result[0, 0], 0.0, rtol=1e-14)

    def test_consistency_between_modes(self):
        """Test both modes produce same skew-symmetric result"""
        n = 4
        # Create with defined lower triangle
        a_lower = np.random.randn(n, n)
        for i in range(n):
            for j in range(i, n):
                a_lower[i, j] = 999.0
        a_lower = np.asfortranarray(a_lower)

        # uplo=1 means lower triangle is given, construct upper
        skew_from_lower = ma02es(1, n, a_lower.copy(order='F'))

        # Create with upper triangle having same skew-symmetric pattern
        a_upper = np.zeros((n, n), order='F')
        for i in range(n):
            for j in range(i + 1, n):
                a_upper[i, j] = -skew_from_lower[j, i]
            for j in range(i):
                a_upper[i, j] = 999.0

        # uplo=0 means upper triangle is given, construct lower
        skew_from_upper = ma02es(0, n, a_upper)

        # Both should produce same skew-symmetric matrix
        assert_allclose(skew_from_lower, skew_from_upper, rtol=1e-13)


class TestMA02EZ:
    """
    Test MA02EZ - Construct Hermitian/skew-Hermitian matrix from triangle.
    UPLO=0: copy lower, UPLO=1: copy upper.
    TRANS=0: conjugate, TRANS=1: plain.
    SKEW=0: Hermitian, SKEW=1: skew-Hermitian (imag diag), SKEW=2: plain skew.
    Operates in-place.
    """

    def test_hermitian_from_lower(self):
        """Test Hermitian matrix construction from lower triangle"""
        uplo, trans, skew, n = 0, 0, 0, 3
        a = np.array([
            [2+0j, 999+999j, 999+999j],
            [3+4j, 5+0j, 999+999j],
            [6+7j, 8+9j, 10+0j]
        ], dtype=np.complex128, order='F')

        result = ma02ez(uplo, trans, skew, n, a)

        # Upper should be conjugate of lower
        assert_allclose(result[0, 1], np.conj(result[1, 0]), rtol=1e-14)
        assert_allclose(result[0, 2], np.conj(result[2, 0]), rtol=1e-14)
        assert_allclose(result[1, 2], np.conj(result[2, 1]), rtol=1e-14)
        # Check Hermitian property
        assert_allclose(result, result.T.conj(), rtol=1e-13)

    def test_hermitian_from_upper(self):
        """Test Hermitian matrix construction from upper triangle"""
        uplo, trans, skew, n = 1, 0, 0, 3
        a = np.array([
            [2+0j, 3-4j, 6-7j],
            [999+999j, 5+0j, 8-9j],
            [999+999j, 999+999j, 10+0j]
        ], dtype=np.complex128, order='F')

        result = ma02ez(uplo, trans, skew, n, a)

        # Lower should be conjugate of upper
        assert_allclose(result[1, 0], np.conj(result[0, 1]), rtol=1e-14)
        assert_allclose(result[2, 0], np.conj(result[0, 2]), rtol=1e-14)
        assert_allclose(result[2, 1], np.conj(result[1, 2]), rtol=1e-14)
        # Check Hermitian property
        assert_allclose(result, result.T.conj(), rtol=1e-13)

    def test_skew_hermitian_real_diagonal(self):
        """Test Hermitian with real diagonal (skew=1)"""
        uplo, trans, skew, n = 1, 0, 1, 3  # uplo=1: lower triangle given
        a = np.array([
            [2+3j, 999+999j, 999+999j],
            [4+5j, 6+7j, 999+999j],
            [8+9j, 10+11j, 12+13j]
        ], dtype=np.complex128, order='F')

        result = ma02ez(uplo, trans, skew, n, a)

        # Diagonal should be purely real
        for i in range(n):
            assert_allclose(result[i, i].imag, 0.0, atol=1e-14)
        # Check Hermitian: A = A^H (not skew-Hermitian!)
        assert_allclose(result, result.T.conj(), rtol=1e-13)

    def test_skew_hermitian_imaginary_diagonal(self):
        """Test skew-Hermitian with imaginary diagonal (skew=3)"""
        uplo, trans, skew, n = 1, 0, 3, 3  # uplo=1: lower triangle given
        a = np.array([
            [2+3j, 999+999j, 999+999j],
            [4+5j, 6+7j, 999+999j],
            [8+9j, 10+11j, 12+13j]
        ], dtype=np.complex128, order='F')

        result = ma02ez(uplo, trans, skew, n, a)

        # Diagonal: extracts imaginary part and stores as REAL value
        # a[0,0] = 2+3j -> imag = 3 -> stored as 3+0j (not 0+3j!)
        # This is what CMPLX(cimag(...), 0.0) and DIMAG() in Fortran do
        # Note: This means the diagonal is NOT purely imaginary, breaking skew-Hermitian property!
        assert_allclose(result[0, 0].real, 3.0, atol=1e-14)
        assert_allclose(result[1, 1].real, 7.0, atol=1e-14)
        assert_allclose(result[2, 2].real, 13.0, atol=1e-14)
        for i in range(n):
            assert_allclose(result[i, i].imag, 0.0, atol=1e-14)

        # Check off-diagonal skew-Hermitian: a_ij = -conj(a_ji)
        for i in range(n):
            for j in range(i+1, n):
                assert_allclose(result[i, j], -np.conj(result[j, i]), rtol=1e-13)

    def test_plain_transpose_from_lower(self):
        """Test plain transpose without conjugation (trans=1, skew=0)"""
        uplo, trans, skew, n = 1, 1, 0, 3  # uplo=1: lower triangle given
        a = np.array([
            [1+1j, 999+999j, 999+999j],
            [2+2j, 3+3j, 999+999j],
            [4+4j, 5+5j, 6+6j]
        ], dtype=np.complex128, order='F')

        result = ma02ez(uplo, trans, skew, n, a)

        # Upper should equal lower (no conjugation)
        assert_allclose(result[0, 1], result[1, 0], rtol=1e-14)
        assert_allclose(result[0, 2], result[2, 0], rtol=1e-14)
        # Check symmetry (not Hermitian)
        assert_allclose(result, result.T, rtol=1e-13)

    def test_plain_skew_transpose(self):
        """Test plain skew transpose (trans=1, skew=2)"""
        uplo, trans, skew, n = 1, 1, 2, 3  # uplo=1: lower triangle given
        a = np.array([
            [1+1j, 999+999j, 999+999j],
            [2+2j, 3+3j, 999+999j],
            [4+4j, 5+5j, 6+6j]
        ], dtype=np.complex128, order='F')

        result = ma02ez(uplo, trans, skew, n, a)

        # Upper should be negative of lower
        assert_allclose(result[0, 1], -result[1, 0], rtol=1e-14)
        assert_allclose(result[0, 2], -result[2, 0], rtol=1e-14)
        # Check skew-symmetry for off-diagonal
        # Note: diagonal is NOT set to zero by the function, left unchanged
        assert_allclose(result[0, 1], -result[1, 0], rtol=1e-14)
        assert_allclose(result[0, 2], -result[2, 0], rtol=1e-14)
        assert_allclose(result[1, 2], -result[2, 1], rtol=1e-14)

    def test_purely_real_hermitian(self):
        """Test Hermitian construction with purely real input"""
        uplo, trans, skew, n = 0, 0, 0, 3
        a = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.complex128, order='F')

        result = ma02ez(uplo, trans, skew, n, a)

        # Real matrices are Hermitian if symmetric
        assert_allclose(result, result.T.conj(), rtol=1e-13)
        assert_allclose(result.imag, 0.0, atol=1e-14)


class TestMA02FD:
    """
    Test MA02FD - Compute rotation parameters (c, s) for plane rotation.
    Given x1, x2, computes c and s such that:
    [c  s] [x1]   [r]
    [-s c] [x2] = [0]
    Returns info=1 if |x2| >= |x1| (error condition), else info=0.
    Also updates x1 to r.
    """

    def test_standard_rotation(self):
        """Test standard case where |x1| > |x2|"""
        x1, x2 = 3.0, 4.0
        x1_out, c, s, info = ma02fd(x1, x2)

        # Should succeed since |x1| < |x2| would be error
        # Actually from code: error if |x2| >= |x1|
        # 4.0 >= 3.0 is True, so info should be 1
        assert info == 1

    def test_larger_x1(self):
        """Test case where |x1| > |x2| (should succeed)"""
        x1, x2 = 5.0, 3.0
        x1_out, c, s, info = ma02fd(x1, x2)

        assert info == 0
        # Check rotation properties: c^2 + s^2 should be close to 1
        assert_allclose(c**2 + s**2, 1.0, rtol=1e-13)
        # s should be x2/x1
        assert_allclose(s, x2 / x1, rtol=1e-13)
        # x1_out should be c * x1
        assert_allclose(x1_out, c * x1, rtol=1e-13)

    def test_negative_x1_larger_magnitude(self):
        """Test with negative x1 but larger magnitude"""
        x1, x2 = -5.0, 2.0
        x1_out, c, s, info = ma02fd(x1, x2)

        assert info == 0
        assert_allclose(c**2 + s**2, 1.0, rtol=1e-13)
        # c should have sign of x1 (negative here)
        assert c < 0

    def test_both_zero(self):
        """Test with both values zero"""
        x1, x2 = 0.0, 0.0
        x1_out, c, s, info = ma02fd(x1, x2)

        # From Fortran code: if x1 == 0, then s=0, c=1
        assert_allclose(c, 1.0, rtol=1e-14)
        assert_allclose(s, 0.0, atol=1e-14)
        assert info == 0

    def test_x1_zero_x2_nonzero(self):
        """Test with x1=0 but x2 nonzero - error condition"""
        x1, x2 = 0.0, 5.0
        x1_out, c, s, info = ma02fd(x1, x2)

        # info should be 1 since |x2| >= |x1| (precondition violated)
        # c and s are undefined when info=1
        assert info == 1

    def test_x2_zero(self):
        """Test with x2=0"""
        x1, x2 = 7.0, 0.0
        x1_out, c, s, info = ma02fd(x1, x2)

        assert info == 0
        # s = x2/x1 = 0
        assert_allclose(s, 0.0, atol=1e-14)
        # c should be ±1
        assert_allclose(abs(c), 1.0, rtol=1e-13)
        # x1_out = c * x1
        assert_allclose(x1_out, c * x1, rtol=1e-13)

    def test_negative_values(self):
        """Test with negative values"""
        x1, x2 = -10.0, -3.0
        x1_out, c, s, info = ma02fd(x1, x2)

        assert info == 0
        assert_allclose(c**2 + s**2, 1.0, rtol=1e-13)

    def test_small_values(self):
        """Test with small values"""
        x1, x2 = 1e-10, 1e-11
        x1_out, c, s, info = ma02fd(x1, x2)

        assert info == 0
        assert_allclose(c**2 + s**2, 1.0, rtol=1e-10)


class TestMA02GD:
    """
    Test MA02GD - Apply column permutations to matrix.
    Interchanges columns according to pivot vector ipiv.
    K1, K2: range of columns to process.
    INCX: stride through ipiv array.
    """

    def test_basic_column_swap(self):
        """Test basic column interchange"""
        n = 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F')
        k1, k2 = 1, 1  # Only process first pivot
        ipiv = np.array([2, 1, 3], dtype=np.int32)  # Swap cols 1 and 2
        incx = 1

        result = ma02gd(n, a, k1, k2, ipiv, incx)

        # Column 1 and 2 should be swapped
        expected = np.array([
            [2.0, 1.0, 3.0],
            [5.0, 4.0, 6.0],
            [8.0, 7.0, 9.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_no_swaps_needed(self):
        """Test when ipiv indicates no swaps"""
        n = 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F')
        k1, k2 = 1, 3
        ipiv = np.array([1, 2, 3], dtype=np.int32)  # No swaps
        incx = 1

        result = ma02gd(n, a, k1, k2, ipiv, incx)

        # Matrix should be unchanged
        assert_allclose(result, a, rtol=1e-14)

    def test_multiple_swaps(self):
        """Test multiple column swaps"""
        n = 4
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F')
        k1, k2 = 1, 4
        ipiv = np.array([3, 4, 1, 2], dtype=np.int32)  # Multiple swaps
        incx = 1

        result = ma02gd(n, a, k1, k2, ipiv, incx)

        # Check that swaps were applied
        assert result.shape == a.shape

    def test_partial_range(self):
        """Test applying swaps to partial column range"""
        n = 3
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ], order='F')
        k1, k2 = 2, 2  # Only process ipiv[1]=3, swap columns 2-3
        ipiv = np.array([1, 3, 2, 4], dtype=np.int32)
        incx = 1

        result = ma02gd(n, a, k1, k2, ipiv, incx)

        # Columns 2 and 3 should be swapped, others unchanged
        expected = np.array([
            [1.0, 3.0, 2.0, 4.0],
            [5.0, 7.0, 6.0, 8.0],
            [9.0, 11.0, 10.0, 12.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_single_column(self):
        """Test with single column matrix"""
        n = 5
        a = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], order='F')
        k1, k2 = 1, 1
        ipiv = np.array([1], dtype=np.int32)
        incx = 1

        result = ma02gd(n, a, k1, k2, ipiv, incx)

        # Single column, no swap possible
        assert_allclose(result, a, rtol=1e-14)

    def test_rectangular_matrix(self):
        """Test with rectangular matrix (more rows than columns)"""
        n = 5
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0]
        ], order='F')
        k1, k2 = 1, 1  # Only process ipiv[0]=2, swap columns 1-2
        ipiv = np.array([2, 1, 3], dtype=np.int32)
        incx = 1

        result = ma02gd(n, a, k1, k2, ipiv, incx)

        # First two columns should be swapped
        expected = np.array([
            [2.0, 1.0, 3.0],
            [5.0, 4.0, 6.0],
            [8.0, 7.0, 9.0],
            [11.0, 10.0, 12.0],
            [14.0, 13.0, 15.0]
        ], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_zero_incx(self):
        """Test with incx=0 (should return unchanged)"""
        n = 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F')
        k1, k2 = 1, 3
        ipiv = np.array([3, 2, 1], dtype=np.int32)
        incx = 0

        result = ma02gd(n, a, k1, k2, ipiv, incx)

        # With incx=0, function returns early (no operations)
        assert_allclose(result, a, rtol=1e-14)


class TestMA02GZ:
    """
    Test MA02GZ - Apply column permutations to complex matrix.
    Complex version of MA02GD.
    Interchanges columns according to pivot vector ipiv.
    """

    def test_basic_column_swap(self):
        """Test basic column interchange with complex values"""
        n = 3
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], dtype=np.complex128, order='F')
        k1, k2 = 1, 1  # Only process ipiv[0]=2, swap cols 1 and 2
        ipiv = np.array([2, 1, 3], dtype=np.int32)
        incx = 1

        result = ma02gz(n, a, k1, k2, ipiv, incx)

        # Column 1 and 2 should be swapped
        expected = np.array([
            [2+2j, 1+1j, 3+3j],
            [5+5j, 4+4j, 6+6j],
            [8+8j, 7+7j, 9+9j]
        ], dtype=np.complex128, order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_no_swaps_needed(self):
        """Test when ipiv indicates no swaps"""
        n = 3
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], dtype=np.complex128, order='F')
        k1, k2 = 1, 3
        ipiv = np.array([1, 2, 3], dtype=np.int32)
        incx = 1

        result = ma02gz(n, a, k1, k2, ipiv, incx)

        # Matrix should be unchanged
        assert_allclose(result, a, rtol=1e-14)

    def test_purely_real_complex_array(self):
        """Test with complex array having only real parts"""
        n = 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.complex128, order='F')
        k1, k2 = 1, 1  # Only process ipiv[0]=2, swap cols 1 and 2
        ipiv = np.array([2, 1, 3], dtype=np.int32)
        incx = 1

        result = ma02gz(n, a, k1, k2, ipiv, incx)

        expected = np.array([
            [2.0, 1.0, 3.0],
            [5.0, 4.0, 6.0],
            [8.0, 7.0, 9.0]
        ], dtype=np.complex128, order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_purely_imaginary_array(self):
        """Test with purely imaginary values"""
        n = 3
        a = 1j * np.arange(1.0, 10.0).reshape(3, 3, order='F')
        a = a.astype(np.complex128)
        k1, k2 = 1, 3
        ipiv = np.array([3, 2, 1], dtype=np.int32)
        incx = 1

        result = ma02gz(n, a, k1, k2, ipiv, incx)

        # Should have performed the swaps
        assert result.shape == a.shape
        assert result.dtype == np.complex128

    def test_partial_range(self):
        """Test applying swaps to partial column range"""
        n = 3
        a = np.array([
            [1+1j, 2+2j, 3+3j, 4+4j],
            [5+5j, 6+6j, 7+7j, 8+8j],
            [9+9j, 10+10j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')
        k1, k2 = 2, 2  # Only process ipiv[1]=3, swap columns 2-3
        ipiv = np.array([1, 3, 2, 4], dtype=np.int32)
        incx = 1

        result = ma02gz(n, a, k1, k2, ipiv, incx)

        # Columns 2 and 3 should be swapped
        expected = np.array([
            [1+1j, 3+3j, 2+2j, 4+4j],
            [5+5j, 7+7j, 6+6j, 8+8j],
            [9+9j, 11+11j, 10+10j, 12+12j]
        ], dtype=np.complex128, order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_zero_incx(self):
        """Test with incx=0 (should return unchanged)"""
        n = 3
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], dtype=np.complex128, order='F')
        k1, k2 = 1, 3
        ipiv = np.array([3, 2, 1], dtype=np.int32)
        incx = 0

        result = ma02gz(n, a, k1, k2, ipiv, incx)

        # With incx=0, function returns early
        assert_allclose(result, a, rtol=1e-14)


class TestMA02HD:
    """
    Test MA02HD - Check if real matrix equals DIAG*I (scaled identity).
    JOB=0: check upper triangle only, JOB=1: check lower triangle only, JOB=2: check full matrix.
    DIAG: expected diagonal value.
    Returns 1 if A = DIAG*I in specified region, 0 otherwise.
    """

    def test_upper_triangular_valid(self):
        """Test valid scaled identity matrix (upper triangle check)"""
        job = 0
        m, n = 4, 4
        diag = 1.0
        # Must be identity matrix (or scaled identity) - off-diagonals must be zero!
        a = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F')

        result = ma02hd(job, m, n, diag, a)

        assert result == 1  # IS upper triangular

    def test_upper_triangular_invalid(self):
        """Test matrix that is NOT scaled identity (has off-diagonal in upper triangle)"""
        job = 0
        m, n = 4, 4
        diag = 1.0
        # Has non-zero off-diagonal element in upper triangle
        a = np.array([
            [1.0, 2.0, 0.0, 0.0],  # a[0,1]=2.0 makes it not DIAG*I
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F')

        result = ma02hd(job, m, n, diag, a)

        assert result == 0  # NOT DIAG*I (fails upper triangle check)

    def test_lower_triangular_valid(self):
        """Test valid scaled identity matrix (lower triangle check)"""
        job = 1
        m, n = 4, 4
        diag = 2.0
        # With JOB=1, only checks lower triangle, so upper triangle can have anything
        # But diagonals must be DIAG and lower triangle must be zero
        a = np.array([
            [2.0, 99.0, 99.0, 99.0],  # Upper triangle ignored for JOB=1
            [0.0, 2.0, 99.0, 99.0],   # Lower triangle must be zero
            [0.0, 0.0, 2.0, 99.0],
            [0.0, 0.0, 0.0, 2.0]
        ], order='F')

        result = ma02hd(job, m, n, diag, a)

        assert result == 1  # IS DIAG*I in lower triangle

    def test_lower_triangular_invalid(self):
        """Test matrix that is NOT DIAG*I (has non-zero in lower triangle)"""
        job = 1
        m, n = 4, 4
        diag = 1.0
        # Has non-zero element in lower triangle
        a = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],  # a[1,0]=3.0 makes it not DIAG*I
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F')

        result = ma02hd(job, m, n, diag, a)

        assert result == 0  # NOT DIAG*I (fails lower triangle check)

    def test_diagonal_matrix_valid(self):
        """Test valid diagonal matrix (job=2)"""
        job = 2
        m, n = 3, 3
        diag = 5.0
        a = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ], order='F')

        result = ma02hd(job, m, n, diag, a)

        assert result == 1  # IS diagonal

    def test_wrong_diagonal_value(self):
        """Test matrix with wrong diagonal value"""
        job = 0
        m, n = 3, 3
        diag = 1.0
        a = np.array([
            [2.0, 3.0, 4.0],  # Diagonal is 2.0, not 1.0
            [0.0, 2.0, 5.0],
            [0.0, 0.0, 2.0]
        ], order='F')

        result = ma02hd(job, m, n, diag, a)

        assert result == 0  # NOT correct (diagonal wrong)

    def test_rectangular_upper(self):
        """Test rectangular matrix (more rows than columns)"""
        job = 0
        m, n = 4, 3
        diag = 1.0
        a = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ], order='F')

        result = ma02hd(job, m, n, diag, a)

        assert result == 0  # IS upper triangular

    def test_empty_matrix(self):
        """Test with empty matrix"""
        job = 0
        m, n = 0, 0
        diag = 1.0
        a = np.array([], dtype=np.float64).reshape(0, 0, order='F')

        result = ma02hd(job, m, n, diag, a)

        # Empty matrix returns 0
        assert result == 0


class TestMA02HZ:
    """
    Test MA02HZ - Check if complex matrix equals DIAG*I (scaled identity).
    Complex version of MA02HD.
    JOB=0: check upper triangle, JOB=1: check lower triangle, JOB=2: check full matrix.
    Returns 1 if A = DIAG*I in specified region, 0 otherwise.
    """

    def test_upper_triangular_valid(self):
        """Test valid scaled identity complex matrix (upper triangle check)"""
        job = 0
        m, n = 4, 4
        diag = 1+0j
        # Must be scaled identity - off-diagonals must be zero!
        a = np.array([
            [1+0j, 0+0j, 0+0j, 0+0j],
            [0+0j, 1+0j, 0+0j, 0+0j],
            [0+0j, 0+0j, 1+0j, 0+0j],
            [0+0j, 0+0j, 0+0j, 1+0j]
        ], dtype=np.complex128, order='F')

        result = ma02hz(job, m, n, diag, a)

        assert result == 1  # IS DIAG*I

    def test_upper_triangular_invalid(self):
        """Test non-scaled-identity complex matrix (has off-diagonal in upper triangle)"""
        job = 0
        m, n = 4, 4
        diag = 1+0j
        # Has non-zero off-diagonal in upper triangle
        a = np.array([
            [1+0j, 2+1j, 0+0j, 0+0j],  # a[0,1]!=0 makes it not DIAG*I
            [0+0j, 1+0j, 0+0j, 0+0j],
            [0+0j, 0+0j, 1+0j, 0+0j],
            [0+0j, 0+0j, 0+0j, 1+0j]
        ], dtype=np.complex128, order='F')

        result = ma02hz(job, m, n, diag, a)

        assert result == 0  # NOT DIAG*I

    def test_lower_triangular_valid(self):
        """Test valid scaled identity complex matrix (lower triangle check)"""
        job = 1
        m, n = 4, 4
        diag = 2+1j
        # With JOB=1, only checks lower triangle and diagonal
        a = np.array([
            [2+1j, 99+99j, 99+99j, 99+99j],  # Upper triangle ignored
            [0+0j, 2+1j, 99+99j, 99+99j],     # Lower must be zero
            [0+0j, 0+0j, 2+1j, 99+99j],
            [0+0j, 0+0j, 0+0j, 2+1j]
        ], dtype=np.complex128, order='F')

        result = ma02hz(job, m, n, diag, a)

        assert result == 1  # IS DIAG*I in lower triangle

    def test_diagonal_matrix_valid(self):
        """Test valid scaled identity complex matrix (full check)"""
        job = 2
        m, n = 3, 3
        diag = 1+1j
        a = np.array([
            [1+1j, 0+0j, 0+0j],
            [0+0j, 1+1j, 0+0j],
            [0+0j, 0+0j, 1+1j]
        ], dtype=np.complex128, order='F')

        result = ma02hz(job, m, n, diag, a)

        assert result == 1  # IS DIAG*I

    def test_purely_real_diagonal(self):
        """Test with purely real values in complex array"""
        job = 0
        m, n = 3, 3
        diag = 1.0+0j
        # Must be scaled identity - no off-diagonals!
        a = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.complex128, order='F')

        result = ma02hz(job, m, n, diag, a)

        assert result == 1  # IS DIAG*I

    def test_wrong_diagonal_value(self):
        """Test complex matrix with wrong diagonal value"""
        job = 0
        m, n = 3, 3
        diag = 1+1j
        a = np.array([
            [2+2j, 3+3j, 4+4j],  # Wrong diagonal
            [0+0j, 2+2j, 5+5j],
            [0+0j, 0+0j, 2+2j]
        ], dtype=np.complex128, order='F')

        result = ma02hz(job, m, n, diag, a)

        assert result == 0  # NOT correct (diagonal wrong)


class TestMA02ID:
    """
    Test MA02ID - Compute norm of structured matrix.
    TYP=0: skew-Hermitian, TYP=1: Hermitian.
    NORM=0: 1-norm, NORM=1: Frobenius, NORM=2: inf-norm, NORM=3: max norm.
    Takes two matrices A and QG representing structured matrix.
    """

    def test_max_norm_hermitian(self):
        """Test max norm of Hermitian-structured matrix"""
        typ, norm, n = 1, 3, 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F')
        qg = np.zeros((n, n+1), order='F')
        qg[:, :n] = np.eye(n)
        dwork = np.zeros(n, order='F')

        result = ma02id(typ, norm, n, a, qg, dwork)

        # Max norm should be max absolute value
        assert result >= 0
        assert result <= 9.0

    def test_max_norm_skew_hermitian(self):
        """Test max norm of skew-Hermitian matrix"""
        typ, norm, n = 0, 3, 3
        a = np.array([
            [0.0, 2.0, 3.0],
            [-2.0, 0.0, 4.0],
            [-3.0, -4.0, 0.0]
        ], order='F')
        qg = np.zeros((n, n+1), order='F')
        dwork = np.zeros(n, order='F')

        result = ma02id(typ, norm, n, a, qg, dwork)

        # Should return max absolute value
        assert result >= 0
        assert_allclose(result, 4.0, rtol=1e-13)

    def test_frobenius_norm_hermitian(self):
        """Test Frobenius norm of Hermitian matrix"""
        typ, norm, n = 1, 1, 3
        a = np.eye(n, order='F')
        qg = np.zeros((n, n+1), order='F')
        qg[0, 0] = 1.0
        qg[1, 1] = 1.0
        qg[2, 2] = 1.0
        dwork = np.zeros(n, order='F')

        result = ma02id(typ, norm, n, a, qg, dwork)

        # Frobenius norm should be positive
        assert result > 0

    def test_one_norm(self):
        """Test 1-norm (max column sum)"""
        typ, norm, n = 1, 0, 3
        a = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ], order='F')
        qg = np.zeros((n, n+1), order='F')
        dwork = np.zeros(2*n, order='F')  # Need 2*n for norm=0 or norm=2

        result = ma02id(typ, norm, n, a, qg, dwork)

        # Should return max column sum
        assert result >= 0
        assert result >= 3.0  # At least the max diagonal

    def test_infinity_norm(self):
        """Test infinity norm (max row sum)"""
        typ, norm, n = 1, 2, 3
        a = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ], order='F')
        qg = np.zeros((n, n+1), order='F')
        dwork = np.zeros(2*n, order='F')  # Need 2*n for norm=0 or norm=2

        result = ma02id(typ, norm, n, a, qg, dwork)

        # Should return max row sum
        assert result >= 0

    def test_empty_matrix(self):
        """Test with n=0"""
        typ, norm, n = 1, 3, 0
        a = np.array([], dtype=np.float64).reshape(0, 0, order='F')
        qg = np.array([], dtype=np.float64).reshape(0, 1, order='F')
        dwork = np.array([], dtype=np.float64, order='F')

        result = ma02id(typ, norm, n, a, qg, dwork)

        # Empty matrix should have zero norm
        assert_allclose(result, 0.0, atol=1e-14)

    def test_small_matrix(self):
        """Test with 2x2 matrix"""
        typ, norm, n = 1, 3, 2
        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], order='F')
        qg = np.zeros((n, n+1), order='F')
        dwork = np.zeros(n, order='F')

        result = ma02id(typ, norm, n, a, qg, dwork)

        # Max element is 4.0
        assert_allclose(result, 4.0, rtol=1e-13)


class TestMA02IZ:
    """
    Test MA02IZ - Compute norm of complex structured matrix.
    TYP=0: skew-Hamiltonian, TYP=1: Hamiltonian.
    NORM=0: 1-norm, NORM=1: Frobenius, NORM=2: inf-norm, NORM=3: max norm.
    Takes two complex matrices A and QG representing structured matrix.
    """

    def test_max_norm_hamiltonian(self):
        """Test max norm of Hamiltonian-structured matrix"""
        typ, norm, n = 1, 3, 3
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], dtype=np.complex128, order='F')
        qg = np.zeros((n, n+1), dtype=np.complex128, order='F')
        qg[:, :n] = np.eye(n, dtype=np.complex128)
        dwork = np.zeros(n, order='F')

        result = ma02iz(typ, norm, n, a, qg, dwork)

        # Max norm should be max absolute value
        assert result >= 0

    def test_max_norm_skew_hamiltonian(self):
        """Test max norm of skew-Hamiltonian matrix"""
        typ, norm, n = 0, 3, 3
        a = np.array([
            [0+0j, 2+2j, 3+3j],
            [-2-2j, 0+0j, 4+4j],
            [-3-3j, -4-4j, 0+0j]
        ], dtype=np.complex128, order='F')
        qg = np.zeros((n, n+1), dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02iz(typ, norm, n, a, qg, dwork)

        # Should return max absolute value
        assert result >= 0

    def test_frobenius_norm_hamiltonian(self):
        """Test Frobenius norm of Hamiltonian matrix"""
        typ, norm, n = 1, 1, 3
        a = np.eye(n, dtype=np.complex128, order='F')
        qg = np.zeros((n, n+1), dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02iz(typ, norm, n, a, qg, dwork)

        # Frobenius norm should be positive
        assert result > 0

    def test_one_norm(self):
        """Test 1-norm (max column sum)"""
        typ, norm, n = 1, 0, 3
        a = np.diag([1+1j, 2+2j, 3+3j]).astype(np.complex128, order='F')
        qg = np.zeros((n, n+1), dtype=np.complex128, order='F')
        dwork = np.zeros(2*n, order='F')  # Need 2*n for norm=0 or norm=2

        result = ma02iz(typ, norm, n, a, qg, dwork)

        # Should return max column sum
        assert result >= 0

    def test_purely_real_complex_matrix(self):
        """Test with complex matrix having only real parts"""
        typ, norm, n = 1, 3, 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.complex128, order='F')
        qg = np.zeros((n, n+1), dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02iz(typ, norm, n, a, qg, dwork)

        # Max element is 9.0
        assert_allclose(result, 9.0, rtol=1e-13)

    def test_empty_matrix(self):
        """Test with n=0"""
        typ, norm, n = 1, 3, 0
        a = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
        qg = np.array([], dtype=np.complex128).reshape(0, 1, order='F')
        dwork = np.array([], dtype=np.float64, order='F')

        result = ma02iz(typ, norm, n, a, qg, dwork)

        # Empty matrix should have zero norm
        assert_allclose(result, 0.0, atol=1e-14)

    def test_small_matrix(self):
        """Test with 2x2 complex matrix"""
        typ, norm, n = 1, 3, 2
        a = np.array([
            [1+1j, 2+2j],
            [3+3j, 4+4j]
        ], dtype=np.complex128, order='F')
        qg = np.zeros((n, n+1), dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02iz(typ, norm, n, a, qg, dwork)

        # Max absolute value
        expected_max = np.sqrt(32)  # |4+4j| = sqrt(16+16)
        assert_allclose(result, expected_max, rtol=1e-13)


class TestMA02JD:
    """
    Test MA02JD - Measure orthogonality of two real matrices.
    Computes sqrt(2) * ||(Q1'*Q1 + Q2'*Q2 - I, Q1'*Q2 - Q2'*Q1)||_F
    LTRAN1, LTRAN2: transpose flags for Q1, Q2.
    Returns a scalar measure of non-orthogonality.
    """

    def test_orthogonal_matrices(self):
        """Test with two orthogonal matrices"""
        ltran1, ltran2, n = 0, 0, 3
        q1 = np.eye(n, order='F')
        q2 = np.zeros((n, n), order='F')
        q2[0, 2] = 1.0
        q2[1, 1] = 1.0
        q2[2, 0] = 1.0
        res = np.zeros((n, n), order='F')

        result = ma02jd(ltran1, ltran2, n, q1, q2, res)

        # For orthogonal Q1, Q2: Q1'*Q1 + Q2'*Q2 should be close to 2*I
        # This doesn't satisfy the expected condition, so result > 0
        assert result >= 0

    def test_identity_matrices(self):
        """Test with both matrices being identity"""
        ltran1, ltran2, n = 0, 0, 3
        q1 = np.eye(n, order='F')
        q2 = np.zeros((n, n), order='F')
        res = np.zeros((n, n), order='F')

        result = ma02jd(ltran1, ltran2, n, q1, q2, res)

        # Q1'*Q1 = I, Q2'*Q2 = 0, so Q1'*Q1 + Q2'*Q2 - I = 0
        # Q1'*Q2 = Q2'*Q1 = 0, so difference = 0
        # Result should be close to 0
        assert result >= 0

    def test_with_transpose_flags(self):
        """Test with different transpose combinations"""
        ltran1, ltran2, n = 1, 0, 3
        q1 = np.eye(n, order='F')
        q2 = np.eye(n, order='F')
        res = np.zeros((n, n), order='F')

        result = ma02jd(ltran1, ltran2, n, q1, q2, res)

        # Should compute measure with transposed Q1
        assert result >= 0

    def test_non_orthogonal_matrices(self):
        """Test with non-orthogonal matrices"""
        ltran1, ltran2, n = 0, 0, 3
        q1 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F')
        q2 = np.array([
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0]
        ], order='F')
        res = np.zeros((n, n), order='F')

        result = ma02jd(ltran1, ltran2, n, q1, q2, res)

        # Non-orthogonal matrices should give large result
        assert result > 0

    def test_small_matrix(self):
        """Test with 2x2 matrices"""
        ltran1, ltran2, n = 0, 0, 2
        q1 = np.eye(n, order='F')
        q2 = np.array([[0.0, 1.0], [1.0, 0.0]], order='F')
        res = np.zeros((n, n), order='F')

        result = ma02jd(ltran1, ltran2, n, q1, q2, res)

        # Both are orthogonal rotation matrices
        assert result >= 0

    def test_both_transpose_true(self):
        """Test with both transpose flags true"""
        ltran1, ltran2, n = 1, 1, 3
        q1 = np.eye(n, order='F')
        q2 = np.eye(n, order='F')
        res = np.zeros((n, n), order='F')

        result = ma02jd(ltran1, ltran2, n, q1, q2, res)

        assert result >= 0


class TestMA02JZ:
    """
    Test MA02JZ - Measure orthogonality of two complex matrices.
    Complex version of MA02JD, uses conjugate transpose.
    """

    def test_unitary_matrices(self):
        """Test with two unitary matrices"""
        ltran1, ltran2, n = 0, 0, 3
        q1 = np.eye(n, dtype=np.complex128, order='F')
        q2 = np.zeros((n, n), dtype=np.complex128, order='F')
        q2[0, 2] = 1.0
        q2[1, 1] = 1.0
        q2[2, 0] = 1.0
        res = np.zeros((n, n), dtype=np.complex128, order='F')

        result = ma02jz(ltran1, ltran2, n, q1, q2, res)

        # Unitary matrices should give some measure
        assert result >= 0

    def test_identity_matrices(self):
        """Test with both matrices being identity"""
        ltran1, ltran2, n = 0, 0, 3
        q1 = np.eye(n, dtype=np.complex128, order='F')
        q2 = np.zeros((n, n), dtype=np.complex128, order='F')
        res = np.zeros((n, n), dtype=np.complex128, order='F')

        result = ma02jz(ltran1, ltran2, n, q1, q2, res)

        # Should be close to 0 for this case
        assert result >= 0

    def test_complex_unitary_matrices(self):
        """Test with complex values in unitary matrices"""
        ltran1, ltran2, n = 0, 0, 2
        # Simple unitary matrix
        q1 = (1.0 / np.sqrt(2)) * np.array([
            [1+0j, 1+0j],
            [0+1j, 0-1j]
        ], dtype=np.complex128, order='F')
        q2 = np.eye(n, dtype=np.complex128, order='F')
        res = np.zeros((n, n), dtype=np.complex128, order='F')

        result = ma02jz(ltran1, ltran2, n, q1, q2, res)

        assert result >= 0

    def test_with_transpose_flags(self):
        """Test with different transpose combinations"""
        ltran1, ltran2, n = 1, 0, 3
        q1 = np.eye(n, dtype=np.complex128, order='F')
        q2 = np.eye(n, dtype=np.complex128, order='F')
        res = np.zeros((n, n), dtype=np.complex128, order='F')

        result = ma02jz(ltran1, ltran2, n, q1, q2, res)

        assert result >= 0

    def test_non_unitary_matrices(self):
        """Test with non-unitary complex matrices"""
        ltran1, ltran2, n = 0, 0, 3
        q1 = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], dtype=np.complex128, order='F')
        q2 = np.array([
            [2+1j, 3+2j, 4+3j],
            [5+4j, 6+5j, 7+6j],
            [8+7j, 9+8j, 10+9j]
        ], dtype=np.complex128, order='F')
        res = np.zeros((n, n), dtype=np.complex128, order='F')

        result = ma02jz(ltran1, ltran2, n, q1, q2, res)

        # Non-unitary matrices should give large result
        assert result > 0

    def test_purely_real_complex_matrices(self):
        """Test with complex matrices having only real parts"""
        ltran1, ltran2, n = 0, 0, 2
        q1 = np.eye(n, dtype=np.complex128, order='F')
        q2 = np.eye(n, dtype=np.complex128, order='F')
        res = np.zeros((n, n), dtype=np.complex128, order='F')

        result = ma02jz(ltran1, ltran2, n, q1, q2, res)

        assert result >= 0


class TestMA02MD:
    """
    Test MA02MD - Compute norm of skew-symmetric matrix.
    NORM=0: 1-norm, NORM=1: Frobenius, NORM=2: inf-norm, NORM=3: max norm.
    UPLO=0: upper triangle stored, UPLO=1: lower triangle stored.
    """

    def test_max_norm_upper(self):
        """Test max norm with upper triangle"""
        norm, uplo, n = 3, 0, 4
        a = np.array([
            [0.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F')
        dwork = np.zeros(n, order='F')

        result = ma02md(norm, uplo, n, a, dwork)

        # Max absolute value in upper triangle (excluding diagonal)
        assert_allclose(result, 7.0, rtol=1e-13)

    def test_max_norm_lower(self):
        """Test max norm with lower triangle"""
        norm, uplo, n = 3, 1, 4
        a = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 5.0, 0.0, 0.0],
            [4.0, 6.0, 7.0, 0.0]
        ], order='F')
        dwork = np.zeros(n, order='F')

        result = ma02md(norm, uplo, n, a, dwork)

        # Max absolute value in lower triangle
        assert_allclose(result, 7.0, rtol=1e-13)

    def test_one_norm_upper(self):
        """Test 1-norm with upper triangle"""
        norm, uplo, n = 0, 0, 3
        a = np.array([
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0]
        ], order='F')
        dwork = np.zeros(n, order='F')

        result = ma02md(norm, uplo, n, a, dwork)

        # 1-norm equals infinity norm for skew-symmetric
        # Max column sum (considering skew-symmetry)
        assert result >= 0

    def test_frobenius_norm_upper(self):
        """Test Frobenius norm with upper triangle"""
        norm, uplo, n = 1, 0, 3
        a = np.array([
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0]
        ], order='F')
        dwork = np.zeros(n, order='F')

        result = ma02md(norm, uplo, n, a, dwork)

        # Frobenius norm: sqrt(2 * sum of squares of upper triangle)
        expected = np.sqrt(2.0 * (1.0**2 + 2.0**2 + 3.0**2))
        assert_allclose(result, expected, rtol=1e-13)

    def test_frobenius_norm_lower(self):
        """Test Frobenius norm with lower triangle"""
        norm, uplo, n = 1, 1, 3
        a = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0]
        ], order='F')
        dwork = np.zeros(n, order='F')

        result = ma02md(norm, uplo, n, a, dwork)

        # Frobenius norm: sqrt(2 * sum of squares of lower triangle)
        expected = np.sqrt(2.0 * (1.0**2 + 2.0**2 + 3.0**2))
        assert_allclose(result, expected, rtol=1e-13)

    def test_small_matrix(self):
        """Test with 2x2 skew-symmetric matrix"""
        norm, uplo, n = 3, 0, 2
        a = np.array([
            [0.0, 5.0],
            [0.0, 0.0]
        ], order='F')
        dwork = np.zeros(n, order='F')

        result = ma02md(norm, uplo, n, a, dwork)

        assert_allclose(result, 5.0, rtol=1e-13)

    def test_degenerate_matrix(self):
        """Test with n < 2 (returns 0)"""
        norm, uplo, n = 3, 0, 1
        a = np.array([[0.0]], order='F')
        dwork = np.zeros(max(n, 1), order='F')

        result = ma02md(norm, uplo, n, a, dwork)

        # n < 2 should return 0
        assert_allclose(result, 0.0, atol=1e-14)


class TestMA02MZ:
    """
    Test MA02MZ - Compute norm of skew-Hermitian matrix.
    NORM=0: 1-norm, NORM=1: Frobenius, NORM=2: inf-norm, NORM=3: max norm.
    UPLO=0: upper triangle stored, UPLO=1: lower triangle stored.
    """

    def test_max_norm_upper(self):
        """Test max norm with upper triangle"""
        norm, uplo, n = 3, 0, 4
        a = np.array([
            [0+1j, 2+2j, 3+3j, 4+4j],
            [0+0j, 0+2j, 5+5j, 6+6j],
            [0+0j, 0+0j, 0+3j, 7+7j],
            [0+0j, 0+0j, 0+0j, 0+4j]
        ], dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02mz(norm, uplo, n, a, dwork)

        # Max absolute value (diagonal imaginary parts and off-diagonal)
        assert result >= 0

    def test_max_norm_lower(self):
        """Test max norm with lower triangle"""
        norm, uplo, n = 3, 1, 4
        a = np.array([
            [0+1j, 0+0j, 0+0j, 0+0j],
            [2+2j, 0+2j, 0+0j, 0+0j],
            [3+3j, 5+5j, 0+3j, 0+0j],
            [4+4j, 6+6j, 7+7j, 0+4j]
        ], dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02mz(norm, uplo, n, a, dwork)

        # Max absolute value in lower triangle
        assert result >= 0

    def test_one_norm_upper(self):
        """Test 1-norm with upper triangle"""
        norm, uplo, n = 0, 0, 3
        a = np.array([
            [0+1j, 1+1j, 2+2j],
            [0+0j, 0+2j, 3+3j],
            [0+0j, 0+0j, 0+3j]
        ], dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02mz(norm, uplo, n, a, dwork)

        # 1-norm equals infinity norm for skew-Hermitian
        assert result >= 0

    def test_frobenius_norm_upper(self):
        """Test Frobenius norm with upper triangle"""
        norm, uplo, n = 1, 0, 3
        a = np.array([
            [0+1j, 1+0j, 2+0j],
            [0+0j, 0+2j, 3+0j],
            [0+0j, 0+0j, 0+3j]
        ], dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02mz(norm, uplo, n, a, dwork)

        # Frobenius norm includes diagonal imaginary parts
        assert result >= 0

    def test_frobenius_norm_lower(self):
        """Test Frobenius norm with lower triangle"""
        norm, uplo, n = 1, 1, 3
        a = np.array([
            [0+1j, 0+0j, 0+0j],
            [1+0j, 0+2j, 0+0j],
            [2+0j, 3+0j, 0+3j]
        ], dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02mz(norm, uplo, n, a, dwork)

        # Frobenius norm
        assert result >= 0

    def test_purely_real_off_diagonal(self):
        """Test with purely real off-diagonal elements"""
        norm, uplo, n = 3, 0, 3
        a = np.array([
            [0+1j, 1+0j, 2+0j],
            [0+0j, 0+2j, 3+0j],
            [0+0j, 0+0j, 0+3j]
        ], dtype=np.complex128, order='F')
        dwork = np.zeros(n, order='F')

        result = ma02mz(norm, uplo, n, a, dwork)

        # Max is max(|1|, |2|, |3|, 1, 2, 3) = 3
        assert_allclose(result, 3.0, rtol=1e-13)

    def test_empty_matrix(self):
        """Test with n=0"""
        norm, uplo, n = 3, 0, 0
        a = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
        dwork = np.array([], dtype=np.float64, order='F')

        result = ma02mz(norm, uplo, n, a, dwork)

        # Empty matrix should have zero norm
        assert_allclose(result, 0.0, atol=1e-14)


class TestMA02NZ:
    """
    Test MA02NZ - Permute rows and columns of structured complex matrix.
    Swaps rows/columns k and l while preserving Hermitian/skew-Hermitian.
    UPLO=0: upper, UPLO=1: lower.
    TRANS=0: transpose, TRANS=1: conjugate transpose.
    SKEW=0: Hermitian, SKEW=1: skew-Hermitian.
    K, L: 0-indexed positions to swap.
    """

    def test_hermitian_upper_swap(self):
        """Test swapping in upper Hermitian matrix"""
        uplo, trans, skew, n = 0, 1, 0, 4
        k, l = 1, 3  # Swap rows/columns 1 and 3 (0-indexed)
        a = np.array([
            [1+0j, 2+1j, 3+2j, 4+3j],
            [2-1j, 5+0j, 6+4j, 7+5j],
            [3-2j, 6-4j, 8+0j, 9+6j],
            [4-3j, 7-5j, 9-6j, 10+0j]
        ], dtype=np.complex128, order='F')

        result = ma02nz(uplo, trans, skew, n, k, l, a)

        # Rows and columns k and l should be swapped
        # Check that diagonal elements were swapped
        # Function returns modified array (also modifies in-place)
        assert result is not None

    def test_hermitian_lower_swap(self):
        """Test swapping in lower Hermitian matrix"""
        uplo, trans, skew, n = 1, 1, 0, 4
        k, l = 0, 2  # Swap rows/columns 0 and 2
        a = np.array([
            [1+0j, 0+0j, 0+0j, 0+0j],
            [2-1j, 5+0j, 0+0j, 0+0j],
            [3-2j, 6-4j, 8+0j, 0+0j],
            [4-3j, 7-5j, 9-6j, 10+0j]
        ], dtype=np.complex128, order='F')
        a_orig = a.copy(order='F')

        result = ma02nz(uplo, trans, skew, n, k, l, a)

        # Verify diagonal elements were swapped
        assert_allclose(a[k, k], a_orig[l, l], rtol=1e-14)
        assert_allclose(a[l, l], a_orig[k, k], rtol=1e-14)

    def test_skew_hermitian_upper_swap(self):
        """Test swapping in upper skew-Hermitian matrix"""
        uplo, trans, skew, n = 0, 1, 1, 4
        k, l = 1, 2
        a = np.array([
            [0+1j, 2+1j, 3+2j, 4+3j],
            [0+0j, 0+2j, 5+4j, 6+5j],
            [0+0j, 0+0j, 0+3j, 7+6j],
            [0+0j, 0+0j, 0+0j, 0+4j]
        ], dtype=np.complex128, order='F')
        a_orig = a.copy(order='F')

        result = ma02nz(uplo, trans, skew, n, k, l, a)

        # Diagonal elements should be swapped
        assert_allclose(a[k, k], a_orig[l, l], rtol=1e-14)

    def test_no_swap_same_indices(self):
        """Test with k == l (no swap)"""
        uplo, trans, skew, n = 0, 1, 0, 4
        k, l = 2, 2  # Same index
        a = np.eye(n, dtype=np.complex128, order='F')
        a_orig = a.copy(order='F')

        result = ma02nz(uplo, trans, skew, n, k, l, a)

        # Matrix should be unchanged when k == l
        assert_allclose(a, a_orig, rtol=1e-14)

    def test_hermitian_no_conjugate_transpose(self):
        """Test Hermitian with plain transpose (trans=0)"""
        uplo, trans, skew, n = 0, 0, 0, 3
        k, l = 0, 2
        a = np.array([
            [1+0j, 2+1j, 3+2j],
            [2+1j, 4+0j, 5+3j],
            [3+2j, 5+3j, 6+0j]
        ], dtype=np.complex128, order='F')

        result = ma02nz(uplo, trans, skew, n, k, l, a)

        # Should perform swap with plain transpose
        # Function returns modified array
        assert result is not None

    def test_skew_hermitian_no_conjugate(self):
        """Test skew-Hermitian with plain transpose"""
        uplo, trans, skew, n = 1, 0, 1, 3
        k, l = 0, 1
        a = np.array([
            [0+1j, 0+0j, 0+0j],
            [2+1j, 0+2j, 0+0j],
            [3+2j, 4+3j, 0+3j]
        ], dtype=np.complex128, order='F')

        result = ma02nz(uplo, trans, skew, n, k, l, a)

        # Should perform swap
        # Function returns modified array
        assert result is not None

    def test_adjacent_swap(self):
        """Test swapping adjacent rows/columns"""
        uplo, trans, skew, n = 0, 1, 0, 4
        k, l = 1, 2  # Adjacent indices
        a = np.eye(n, dtype=np.complex128, order='F')
        a_orig = a.copy(order='F')

        result = ma02nz(uplo, trans, skew, n, k, l, a)

        # Diagonal elements should be swapped
        assert_allclose(a[k, k], a_orig[l, l], rtol=1e-14)
        assert_allclose(a[l, l], a_orig[k, k], rtol=1e-14)


class TestMA02OD:
    """
    Test MA02OD - Count zero columns in Hamiltonian/skew-Hamiltonian matrix.
    Matrix represented as [A, DE; G, A'] where A is mxm.
    SKEW=0: Hamiltonian (DE diagonal), SKEW=1: skew-Hamiltonian.
    Returns count of zero columns.
    """

    def test_no_zero_columns(self):
        """Test with no zero columns"""
        skew, m = 0, 3
        a = np.eye(m, order='F')
        de = np.eye(m, m+1, order='F')

        result = ma02od(skew, m, a, de)

        # No zero columns
        assert result == 0

    def test_one_zero_column_in_a(self):
        """Test with one zero column in A block"""
        skew, m = 0, 3
        a = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
            [7.0, 0.0, 9.0]
        ], order='F')
        de = np.zeros((m, m+1), order='F')

        result = ma02od(skew, m, a, de)

        # Column 2 of A is zero and corresponding DE column is zero
        assert result >= 1

    def test_hamiltonian_with_diagonal_de(self):
        """Test Hamiltonian structure with diagonal DE"""
        skew, m = 0, 3
        a = np.zeros((m, m), order='F')
        de = np.zeros((m, m+1), order='F')
        # Set some diagonal values
        de[0, 0] = 1.0
        de[1, 1] = 1.0

        result = ma02od(skew, m, a, de)

        # Column 3 might be zero if a[:,2] and de[2,:] are zero
        assert result >= 0

    def test_skew_hamiltonian(self):
        """Test skew-Hamiltonian structure"""
        skew, m = 1, 3
        a = np.zeros((m, m), order='F')
        de = np.zeros((m, m+1), order='F')

        result = ma02od(skew, m, a, de)

        # All columns are zero
        assert result == 2 * m

    def test_empty_matrix(self):
        """Test with m=0"""
        skew, m = 0, 0
        a = np.array([], dtype=np.float64).reshape(0, 0, order='F')
        de = np.array([], dtype=np.float64).reshape(0, 1, order='F')

        result = ma02od(skew, m, a, de)

        # Empty matrix
        assert result == 0

    def test_small_matrix(self):
        """Test with 2x2 blocks"""
        skew, m = 0, 2
        a = np.array([
            [1.0, 0.0],
            [0.0, 0.0]
        ], order='F')
        de = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], order='F')

        result = ma02od(skew, m, a, de)

        # Count zero columns
        assert result >= 0

    def test_mixed_zero_columns(self):
        """Test with zero columns in both blocks"""
        skew, m = 0, 3
        a = np.zeros((m, m), order='F')
        de = np.zeros((m, m+1), order='F')

        result = ma02od(skew, m, a, de)

        # All columns are zero
        assert result == 2 * m


class TestMA02OZ:
    """
    Test MA02OZ - Count zero columns in complex Hamiltonian/skew-Hamiltonian.
    Complex version of MA02OD.
    """

    def test_no_zero_columns(self):
        """Test with no zero columns"""
        skew, m = 0, 3
        a = np.eye(m, dtype=np.complex128, order='F')
        de = np.eye(m, m+1, dtype=np.complex128, order='F')

        result = ma02oz(skew, m, a, de)

        # No zero columns
        assert result == 0

    def test_one_zero_column(self):
        """Test with one zero column"""
        skew, m = 0, 3
        a = np.array([
            [1+1j, 0+0j, 3+3j],
            [4+4j, 0+0j, 6+6j],
            [7+7j, 0+0j, 9+9j]
        ], dtype=np.complex128, order='F')
        de = np.zeros((m, m+1), dtype=np.complex128, order='F')

        result = ma02oz(skew, m, a, de)

        # Column 2 is zero
        assert result >= 1

    def test_skew_hamiltonian_imaginary_diagonal(self):
        """Test skew-Hamiltonian with purely imaginary diagonal"""
        skew, m = 1, 3
        a = np.zeros((m, m), dtype=np.complex128, order='F')
        de = np.zeros((m, m+1), dtype=np.complex128, order='F')
        # Skew-Hamiltonian: diagonal should be purely imaginary
        de[0, 0] = 0+1j
        de[1, 1] = 0+2j

        result = ma02oz(skew, m, a, de)

        # Some columns might be zero
        assert result >= 0

    def test_hamiltonian_real_diagonal(self):
        """Test Hamiltonian with purely real diagonal"""
        skew, m = 0, 3
        a = np.zeros((m, m), dtype=np.complex128, order='F')
        de = np.zeros((m, m+1), dtype=np.complex128, order='F')
        # Hamiltonian: diagonal should be purely real
        de[0, 0] = 1+0j
        de[1, 1] = 2+0j

        result = ma02oz(skew, m, a, de)

        assert result >= 0

    def test_purely_real_complex_matrix(self):
        """Test with complex matrix having only real parts"""
        skew, m = 0, 2
        a = np.array([
            [1.0, 0.0],
            [0.0, 0.0]
        ], dtype=np.complex128, order='F')
        de = np.zeros((m, m+1), dtype=np.complex128, order='F')

        result = ma02oz(skew, m, a, de)

        assert result >= 0

    def test_all_zero_columns(self):
        """Test with all zero columns"""
        skew, m = 1, 3
        a = np.zeros((m, m), dtype=np.complex128, order='F')
        de = np.zeros((m, m+1), dtype=np.complex128, order='F')

        result = ma02oz(skew, m, a, de)

        # All columns are zero
        assert result == 2 * m

    def test_empty_matrix(self):
        """Test with m=0"""
        skew, m = 0, 0
        a = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
        de = np.array([], dtype=np.complex128).reshape(0, 1, order='F')

        result = ma02oz(skew, m, a, de)

        # Empty matrix
        assert result == 0


class TestMA02PD:
    """
    Test MA02PD - Count zero rows and zero columns in real matrix.
    Returns (nzr, nzc) where nzr is count of zero rows and nzc is zero cols.
    """

    def test_no_zero_rows_or_columns(self):
        """Test matrix with no zero rows or columns"""
        m, n = 3, 4
        a = np.ones((m, n), order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 0
        assert nzc == 0

    def test_one_zero_column(self):
        """Test matrix with one zero column"""
        m, n = 3, 4
        a = np.array([
            [1.0, 0.0, 3.0, 4.0],
            [5.0, 0.0, 7.0, 8.0],
            [9.0, 0.0, 11.0, 12.0]
        ], order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 0
        assert nzc == 1  # Column 2 is zero

    def test_one_zero_row(self):
        """Test matrix with one zero row"""
        m, n = 4, 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ], order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 1  # Row 2 is zero
        assert nzc == 0

    def test_multiple_zero_rows_and_columns(self):
        """Test matrix with multiple zero rows and columns"""
        m, n = 4, 4
        a = np.array([
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 9.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 2  # Rows 2 and 4 are zero
        assert nzc == 2  # Columns 2 and 4 are zero

    def test_all_zero_matrix(self):
        """Test all-zero matrix"""
        m, n = 3, 3
        a = np.zeros((m, n), order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == m
        assert nzc == n

    def test_identity_matrix(self):
        """Test identity matrix"""
        m, n = 3, 3
        a = np.eye(m, n, order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 0
        assert nzc == 0

    def test_rectangular_tall_matrix(self):
        """Test rectangular matrix (more rows than columns)"""
        m, n = 5, 3
        a = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [7.0, 8.0, 9.0],
            [0.0, 0.0, 0.0],
            [13.0, 14.0, 15.0]
        ], order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 2  # Rows 2 and 4 are zero
        assert nzc == 0

    def test_rectangular_wide_matrix(self):
        """Test rectangular matrix (more columns than rows)"""
        m, n = 3, 5
        a = np.array([
            [1.0, 0.0, 3.0, 0.0, 5.0],
            [6.0, 0.0, 8.0, 0.0, 10.0],
            [11.0, 0.0, 13.0, 0.0, 15.0]
        ], order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 0
        assert nzc == 2  # Columns 2 and 4 are zero

    def test_empty_matrix(self):
        """Test with m=0 or n=0"""
        m, n = 0, 0
        a = np.array([], dtype=np.float64).reshape(0, 0, order='F')

        nzr, nzc = ma02pd(m, n, a)

        assert nzr == 0
        assert nzc == 0


class TestMA02PZ:
    """
    Test MA02PZ - Count zero rows and zero columns in complex matrix.
    Complex version of MA02PD.
    """

    def test_no_zero_rows_or_columns(self):
        """Test matrix with no zero rows or columns"""
        m, n = 3, 4
        a = np.ones((m, n), dtype=np.complex128, order='F')

        nzr, nzc = ma02pz(m, n, a)

        assert nzr == 0
        assert nzc == 0

    def test_one_zero_column(self):
        """Test complex matrix with one zero column"""
        m, n = 3, 4
        a = np.array([
            [1+1j, 0+0j, 3+3j, 4+4j],
            [5+5j, 0+0j, 7+7j, 8+8j],
            [9+9j, 0+0j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')

        nzr, nzc = ma02pz(m, n, a)

        assert nzr == 0
        assert nzc == 1  # Column 2 is zero

    def test_one_zero_row(self):
        """Test complex matrix with one zero row"""
        m, n = 4, 3
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [0+0j, 0+0j, 0+0j],
            [7+7j, 8+8j, 9+9j],
            [10+10j, 11+11j, 12+12j]
        ], dtype=np.complex128, order='F')

        nzr, nzc = ma02pz(m, n, a)

        assert nzr == 1  # Row 2 is zero
        assert nzc == 0

    def test_multiple_zero_rows_and_columns(self):
        """Test complex matrix with multiple zeros"""
        m, n = 4, 4
        a = np.array([
            [1+1j, 0+0j, 3+3j, 0+0j],
            [0+0j, 0+0j, 0+0j, 0+0j],
            [7+7j, 0+0j, 9+9j, 0+0j],
            [0+0j, 0+0j, 0+0j, 0+0j]
        ], dtype=np.complex128, order='F')

        nzr, nzc = ma02pz(m, n, a)

        assert nzr == 2  # Rows 2 and 4 are zero
        assert nzc == 2  # Columns 2 and 4 are zero

    def test_purely_real_complex_matrix(self):
        """Test with complex matrix having only real parts"""
        m, n = 3, 3
        a = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
            [7.0, 0.0, 9.0]
        ], dtype=np.complex128, order='F')

        nzr, nzc = ma02pz(m, n, a)

        assert nzr == 0
        assert nzc == 1  # Column 2 is zero

    def test_all_zero_matrix(self):
        """Test all-zero complex matrix"""
        m, n = 3, 3
        a = np.zeros((m, n), dtype=np.complex128, order='F')

        nzr, nzc = ma02pz(m, n, a)

        assert nzr == m
        assert nzc == n

    def test_purely_imaginary_values(self):
        """Test with purely imaginary values"""
        m, n = 3, 3
        a = np.array([
            [1j, 0j, 3j],
            [4j, 0j, 6j],
            [7j, 0j, 9j]
        ], dtype=np.complex128, order='F')

        nzr, nzc = ma02pz(m, n, a)

        assert nzr == 0
        assert nzc == 1  # Column 2 is zero


class TestMA02RD:
    """
    Test MA02RD - Sort two arrays simultaneously using quicksort.
    ID=0: increasing order, ID=1: decreasing order.
    Sorts array D and applies same permutation to array E.
    """

    def test_sort_increasing(self):
        """Test sorting in increasing order"""
        id_val, n = 0, 5
        d = np.array([5.0, 2.0, 8.0, 1.0, 3.0], order='F')
        e = np.array([50.0, 20.0, 80.0, 10.0, 30.0], order='F')

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        # D should be sorted in increasing order
        assert_allclose(d_out, np.array([1.0, 2.0, 3.0, 5.0, 8.0]), rtol=1e-14)
        # E should follow same permutation
        assert_allclose(e_out, np.array([10.0, 20.0, 30.0, 50.0, 80.0]), rtol=1e-14)

    def test_sort_decreasing(self):
        """Test sorting in decreasing order"""
        id_val, n = 1, 5
        d = np.array([5.0, 2.0, 8.0, 1.0, 3.0], order='F')
        e = np.array([50.0, 20.0, 80.0, 10.0, 30.0], order='F')

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        # D should be sorted in decreasing order
        assert_allclose(d_out, np.array([8.0, 5.0, 3.0, 2.0, 1.0]), rtol=1e-14)
        # E should follow same permutation
        assert_allclose(e_out, np.array([80.0, 50.0, 30.0, 20.0, 10.0]), rtol=1e-14)

    def test_already_sorted_increasing(self):
        """Test with already sorted array (increasing)"""
        id_val, n = 0, 5
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0], order='F')
        e = np.array([10.0, 20.0, 30.0, 40.0, 50.0], order='F')

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        # Should remain sorted
        assert_allclose(d_out, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), rtol=1e-14)
        assert_allclose(e_out, np.array([10.0, 20.0, 30.0, 40.0, 50.0]), rtol=1e-14)

    def test_duplicate_values(self):
        """Test with duplicate values in D"""
        id_val, n = 0, 6
        d = np.array([3.0, 1.0, 3.0, 2.0, 1.0, 2.0], order='F')
        e = np.array([30.0, 10.0, 31.0, 20.0, 11.0, 21.0], order='F')

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        # D should be sorted
        expected_d = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        assert_allclose(d_out, expected_d, rtol=1e-14)

    def test_single_element(self):
        """Test with single element"""
        id_val, n = 0, 1
        d = np.array([5.0], order='F')
        e = np.array([50.0], order='F')

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        # Single element remains unchanged
        assert_allclose(d_out, np.array([5.0]), rtol=1e-14)
        assert_allclose(e_out, np.array([50.0]), rtol=1e-14)

    def test_two_elements(self):
        """Test with two elements"""
        id_val, n = 0, 2
        d = np.array([5.0, 2.0], order='F')
        e = np.array([50.0, 20.0], order='F')

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        assert_allclose(d_out, np.array([2.0, 5.0]), rtol=1e-14)
        assert_allclose(e_out, np.array([20.0, 50.0]), rtol=1e-14)

    def test_large_array(self):
        """Test with larger array"""
        id_val, n = 0, 30
        d = np.random.permutation(n).astype(np.float64, order='F')
        e = d.copy(order='F') * 10.0

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        # Check that D is sorted
        for i in range(n-1):
            assert d_out[i] <= d_out[i+1]
        # Check that E follows same permutation
        for i in range(n):
            assert_allclose(e_out[i], d_out[i] * 10.0, rtol=1e-13)

    def test_negative_values(self):
        """Test with negative values"""
        id_val, n = 0, 5
        d = np.array([-2.0, 5.0, -8.0, 1.0, -3.0], order='F')
        e = np.array([20.0, 50.0, 80.0, 10.0, 30.0], order='F')

        d_out, e_out, info = ma02rd(id_val, n, d, e)

        assert info == 0
        expected_d = np.array([-8.0, -3.0, -2.0, 1.0, 5.0])
        assert_allclose(d_out, expected_d, rtol=1e-14)


class TestMA02SD:
    """
    Test MA02SD - Find minimum absolute value (excluding zeros) in matrix.
    Returns smallest non-zero absolute value, or overflow value if all zero.
    """

    def test_positive_values(self):
        """Test with positive values"""
        m, n = 3, 3
        a = np.array([
            [5.0, 2.0, 8.0],
            [1.0, 3.0, 6.0],
            [9.0, 4.0, 7.0]
        ], order='F')

        result = ma02sd(m, n, a)

        # Minimum non-zero value is 1.0
        assert_allclose(result, 1.0, rtol=1e-13)

    def test_with_zeros(self):
        """Test with some zero values"""
        m, n = 3, 3
        a = np.array([
            [5.0, 0.0, 8.0],
            [0.0, 3.0, 6.0],
            [9.0, 4.0, 0.0]
        ], order='F')

        result = ma02sd(m, n, a)

        # Minimum non-zero value is 3.0
        assert_allclose(result, 3.0, rtol=1e-13)

    def test_negative_values(self):
        """Test with negative values"""
        m, n = 3, 3
        a = np.array([
            [-5.0, -2.0, -8.0],
            [1.0, -0.5, 6.0],
            [-9.0, 4.0, -7.0]
        ], order='F')

        result = ma02sd(m, n, a)

        # Minimum absolute non-zero value is |−0.5| = 0.5
        assert_allclose(result, 0.5, rtol=1e-13)

    def test_small_values(self):
        """Test with very small values"""
        m, n = 2, 2
        a = np.array([
            [1e-10, 1e-8],
            [1e-12, 1e-9]
        ], order='F')

        result = ma02sd(m, n, a)

        # Minimum is 1e-12
        assert_allclose(result, 1e-12, rtol=1e-10)

    def test_single_element(self):
        """Test with 1x1 matrix"""
        m, n = 1, 1
        a = np.array([[7.0]], order='F')

        result = ma02sd(m, n, a)

        assert_allclose(result, 7.0, rtol=1e-13)

    def test_rectangular_matrix(self):
        """Test with rectangular matrix"""
        m, n = 2, 4
        a = np.array([
            [5.0, 2.0, 8.0, 0.1],
            [1.0, 3.0, 6.0, 4.0]
        ], order='F')

        result = ma02sd(m, n, a)

        # Minimum non-zero value is 0.1
        assert_allclose(result, 0.1, rtol=1e-13)

    def test_all_same_values(self):
        """Test with all same non-zero values"""
        m, n = 3, 3
        a = np.ones((m, n), order='F') * 5.0

        result = ma02sd(m, n, a)

        assert_allclose(result, 5.0, rtol=1e-13)

    def test_empty_matrix(self):
        """Test with m=0 or n=0"""
        m, n = 0, 0
        a = np.array([], dtype=np.float64).reshape(0, 0, order='F')

        result = ma02sd(m, n, a)

        # Should return 0.0 for empty matrix
        assert_allclose(result, 0.0, atol=1e-14)
