"""
Tests for MB01XX family - matrix operations
"""

import numpy as np
from numpy.testing import assert_allclose

from slicutlet import (
    mb01ld, mb01oc, mb01od, mb01oe, mb01oh, mb01oo, mb01os, mb01ot,
    mb01rb, mb01rd, mb01rh, mb01rt, mb01ru, mb01rw, mb01rx, mb01ry,
    mb01sd, mb01ss, mb01td, mb01ud, mb01uw, mb01ux, mb01uy, mb01uz,
    mb01vd, mb01wd, mb01xd, mb01xy, mb01yd, mb01zd
)


class TestMB01LD:
    """Tests for mb01ld - symmetric rank-k operation A := alpha*B*B' + beta*A"""

    def test_upper_no_transpose(self):
        """Test upper triangular update without transpose"""
        rng = np.random.default_rng(1234567890)
        n, k = 4, 3
        alpha, beta = 2.5, 0.5

        # Create symmetric matrix A (upper)
        A = rng.uniform(-5, 5, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-5, 5, (n, k)))

        # Expected: A := alpha*B*B' + beta*A
        expected = alpha * B @ B.T + beta * A

        A_out, info = mb01ld(0, 0, n, k, alpha, beta, B, A)

        assert info == 0
        # Only upper triangle is updated
        assert_allclose(np.triu(A_out), np.triu(expected), rtol=1e-12)

    def test_lower_transpose(self):
        """Test lower triangular update with transpose"""
        rng = np.random.default_rng(2345678901)
        n, k = 5, 4
        alpha, beta = 1.0, 2.0

        # Create symmetric matrix A (lower)
        A = rng.uniform(-3, 3, (n, n))
        A = np.tril(A) + np.tril(A, -1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-3, 3, (k, n)))

        # Expected: A := alpha*B'*B + beta*A
        expected = alpha * B.T @ B + beta * A

        A_out, info = mb01ld(1, 1, n, k, alpha, beta, B, A)

        assert info == 0
        assert_allclose(np.tril(A_out), np.tril(expected), rtol=1e-12)


class TestMB01OC:
    """Tests for mb01oc - Hessenberg matrix operations"""

    def test_basic_operation(self):
        """Test basic Hessenberg matrix operation"""
        rng = np.random.default_rng(3456789012)
        n = 6
        alpha, beta = 1.5, -0.5

        H = np.asfortranarray(rng.uniform(-4, 4, (n, n)))
        A = np.asfortranarray(rng.uniform(-4, 4, (n, n)))

        H_out, A_out, info = mb01oc(n, alpha, beta, H, A)

        assert info == 0
        assert H_out.shape == (n, n)
        assert A_out.shape == (n, n)


class TestMB01OD:
    """Tests for mb01od - Combined Hessenberg and triangular operations"""

    def test_combined_operation(self):
        """Test combined operation"""
        rng = np.random.default_rng(4567890123)
        n = 5
        alpha, beta = 2.0, 1.0

        H = np.asfortranarray(rng.uniform(-3, 3, (n, n)))
        A = np.asfortranarray(rng.uniform(-3, 3, (n, n)))

        H_out, A_out, info = mb01od(0, n, alpha, beta, H, A)

        assert info == 0
        assert H_out.shape == (n, n)
        assert A_out.shape == (n, n)


class TestMB01OE:
    """Tests for mb01oe - Triangular matrix helper"""

    def test_upper_triangular(self):
        """Test upper triangular operation"""
        rng = np.random.default_rng(5678901234)
        n = 6
        alpha = 2.5

        T = np.triu(rng.uniform(-5, 5, (n, n)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-5, 5, (n, n)))

        T_out, A_out, info = mb01oe(0, 0, n, alpha, T, A)

        assert info == 0


class TestMB01OH:
    """Tests for mb01oh - Hessenberg helper"""

    def test_hessenberg_helper(self):
        """Test Hessenberg helper operation"""
        rng = np.random.default_rng(6789012345)
        n = 5
        alpha = 1.5

        H = np.asfortranarray(rng.uniform(-4, 4, (n, n)))
        A = np.asfortranarray(rng.uniform(-4, 4, (n, n)))

        H_out, A_out, info = mb01oh(n, alpha, H, A)

        assert info == 0


class TestMB01OO:
    """Tests for mb01oo - Pure computation without alpha/beta"""

    def test_pure_computation(self):
        """Test pure matrix computation"""
        rng = np.random.default_rng(7890123456)
        n = 4

        H = np.asfortranarray(rng.uniform(-3, 3, (n, n)))
        A = np.asfortranarray(rng.uniform(-3, 3, (n, n)))

        H_out, A_out, info = mb01oo(n, H, A)

        assert info == 0


class TestMB01OS:
    """Tests for mb01os - Similar to mb01oo"""

    def test_computation(self):
        """Test matrix computation"""
        rng = np.random.default_rng(8901234567)
        n = 5

        H = np.asfortranarray(rng.uniform(-4, 4, (n, n)))
        A = np.asfortranarray(rng.uniform(-4, 4, (n, n)))

        H_out, A_out, info = mb01os(n, H, A)

        assert info == 0


class TestMB01OT:
    """Tests for mb01ot - Triangular helper"""

    def test_triangular_helper(self):
        """Test triangular helper operation"""
        rng = np.random.default_rng(9012345678)
        n = 6
        alpha = 0.75

        T = np.triu(rng.uniform(-5, 5, (n, n)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-5, 5, (n, n)))

        T_out, A_out, info = mb01ot(0, n, alpha, T, A)

        assert info == 0


class TestMB01RB:
    """Tests for mb01rb - Rank-2k operation"""

    def test_rank_2k_upper(self):
        """Test rank-2k update on upper triangle"""
        rng = np.random.default_rng(1234509876)
        n, k = 4, 3
        alpha, beta = 1.5, 0.5

        # Create symmetric matrix C (upper)
        C = rng.uniform(-3, 3, (n, n))
        C = np.triu(C) + np.triu(C, 1).T
        C = np.asfortranarray(C)

        A = np.asfortranarray(rng.uniform(-3, 3, (n, k)))
        B = np.asfortranarray(rng.uniform(-3, 3, (n, k)))

        # Expected: C := alpha*(A*B' + B*A') + beta*C
        expected = alpha * (A @ B.T + B @ A.T) + beta * C

        C_out, info = mb01rb(0, 0, n, k, alpha, beta, A, B, C)

        assert info == 0
        assert_allclose(np.triu(C_out), np.triu(expected), rtol=1e-12)

    def test_rank_2k_lower_transpose(self):
        """Test rank-2k update on lower triangle with transpose"""
        rng = np.random.default_rng(2345609871)
        n, k = 5, 3
        alpha, beta = 2.0, 1.0

        C = rng.uniform(-2, 2, (n, n))
        C = np.tril(C) + np.tril(C, -1).T
        C = np.asfortranarray(C)

        A = np.asfortranarray(rng.uniform(-2, 2, (k, n)))
        B = np.asfortranarray(rng.uniform(-2, 2, (k, n)))

        # Expected: C := alpha*(A'*B + B'*A) + beta*C
        expected = alpha * (A.T @ B + B.T @ A) + beta * C

        C_out, info = mb01rb(1, 1, n, k, alpha, beta, A, B, C)

        assert info == 0
        assert_allclose(np.tril(C_out), np.tril(expected), rtol=1e-12)


class TestMB01RD:
    """Tests for mb01rd - Symmetric rank-k with specific pattern"""

    def test_upper_pattern(self):
        """Test upper triangular rank-k operation"""
        rng = np.random.default_rng(3456709812)
        n, k = 4, 3
        alpha, beta = 1.0, 0.5

        A = rng.uniform(-4, 4, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-4, 4, (n, k)))

        expected = alpha * B @ B.T + beta * A

        A_out, info = mb01rd(0, 0, n, k, alpha, beta, B, A)

        assert info == 0
        assert_allclose(np.triu(A_out), np.triu(expected), rtol=1e-12)


class TestMB01RH:
    """Tests for mb01rh - Variant of rank-k operation"""

    def test_basic_operation(self):
        """Test basic rank-k variant"""
        rng = np.random.default_rng(4567809123)
        n, k = 5, 4
        alpha, beta = 2.5, 1.5

        A = rng.uniform(-3, 3, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-3, 3, (n, k)))

        A_out, info = mb01rh(0, 0, n, k, alpha, beta, B, A)

        assert info == 0


class TestMB01RT:
    """Tests for mb01rt - Another rank-k variant"""

    def test_rank_k_variant(self):
        """Test rank-k variant operation"""
        rng = np.random.default_rng(5678909234)
        n, k = 4, 3
        alpha, beta = 1.2, 0.8

        A = rng.uniform(-5, 5, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-5, 5, (n, k)))

        A_out, info = mb01rt(0, 0, n, k, alpha, beta, B, A)

        assert info == 0


class TestMB01RU:
    """Tests for mb01ru - Rank-k update variant"""

    def test_update_variant(self):
        """Test rank-k update variant"""
        rng = np.random.default_rng(6789009345)
        n, k = 6, 4
        alpha, beta = 0.5, 1.5

        A = rng.uniform(-2, 2, (n, n))
        A = np.tril(A) + np.tril(A, -1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-2, 2, (n, k)))

        A_out, info = mb01ru(1, 0, n, k, alpha, beta, B, A)

        assert info == 0


class TestMB01RW:
    """Tests for mb01rw - Rank-k operation variant"""

    def test_rw_variant(self):
        """Test mb01rw rank-k operation"""
        rng = np.random.default_rng(7890109456)
        n, k = 5, 3
        alpha, beta = 3.0, 0.25

        A = rng.uniform(-4, 4, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-4, 4, (n, k)))

        A_out, info = mb01rw(0, 0, n, k, alpha, beta, B, A)

        assert info == 0


class TestMB01RX:
    """Tests for mb01rx - Rank-2k helper"""

    def test_rank_2k_helper(self):
        """Test rank-2k helper operation"""
        rng = np.random.default_rng(8901209567)
        n, k = 4, 3
        alpha, beta = 1.0, 1.0

        C = rng.uniform(-3, 3, (n, n))
        C = np.triu(C) + np.triu(C, 1).T
        C = np.asfortranarray(C)

        A = np.asfortranarray(rng.uniform(-3, 3, (n, k)))
        B = np.asfortranarray(rng.uniform(-3, 3, (n, k)))

        C_out, info = mb01rx(0, 0, n, k, alpha, beta, A, B, C)

        assert info == 0


class TestMB01RY:
    """Tests for mb01ry - Hessenberg rank-k variant"""

    def test_hessenberg_rank_k(self):
        """Test Hessenberg rank-k operation"""
        rng = np.random.default_rng(9012309678)
        n = 5
        alpha, beta = 1.5, 0.5

        H = np.asfortranarray(rng.uniform(-4, 4, (n, n)))
        A = np.asfortranarray(rng.uniform(-4, 4, (n, n)))

        H_out, A_out, info = mb01ry(n, alpha, beta, H, A)

        assert info == 0


class TestMB01SD:
    """Tests for mb01sd - Row/column scaling"""

    def test_row_scaling(self):
        """Test row scaling operation"""
        rng = np.random.default_rng(1237890456)
        m, n = 5, 4

        A = np.asfortranarray(rng.uniform(-5, 5, (m, n)))
        d = np.asfortranarray(rng.uniform(0.5, 2.0, m))

        # jobs = 0: multiply rows by d
        A_out, info = mb01sd(0, m, n, A, d)

        assert info == 0
        expected = d[:, None] * A
        assert_allclose(A_out, expected, rtol=1e-14)

    def test_column_scaling(self):
        """Test column scaling operation"""
        rng = np.random.default_rng(2348901567)
        m, n = 6, 5

        A = np.asfortranarray(rng.uniform(-4, 4, (m, n)))
        d = np.asfortranarray(rng.uniform(0.5, 2.5, n))

        # jobs = 1: multiply columns by d
        A_out, info = mb01sd(1, m, n, A, d)

        assert info == 0
        expected = A * d[None, :]
        assert_allclose(A_out, expected, rtol=1e-14)

    def test_row_inverse_scaling(self):
        """Test row scaling with inverse"""
        rng = np.random.default_rng(3459012678)
        m, n = 4, 6

        A = np.asfortranarray(rng.uniform(-3, 3, (m, n)))
        d = np.asfortranarray(rng.uniform(0.5, 2.0, m))

        # jobs = 2: divide rows by d
        A_out, info = mb01sd(2, m, n, A, d)

        assert info == 0
        expected = A / d[:, None]
        assert_allclose(A_out, expected, rtol=1e-14)


class TestMB01SS:
    """Tests for mb01ss - Symmetric matrix scaling"""

    def test_symmetric_scaling_upper(self):
        """Test symmetric scaling of upper triangular part"""
        rng = np.random.default_rng(4560123789)
        n = 5

        A = rng.uniform(-4, 4, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        d = np.asfortranarray(rng.uniform(0.5, 2.0, n))

        # jobs = 0: A := D*A*D (upper)
        A_out, info = mb01ss(0, 0, n, A, d)

        assert info == 0
        expected = np.diag(d) @ A @ np.diag(d)
        assert_allclose(np.triu(A_out), np.triu(expected), rtol=1e-13)

    def test_symmetric_scaling_lower(self):
        """Test symmetric scaling of lower triangular part"""
        rng = np.random.default_rng(5671234890)
        n = 4

        A = rng.uniform(-3, 3, (n, n))
        A = np.tril(A) + np.tril(A, -1).T
        A = np.asfortranarray(A)

        d = np.asfortranarray(rng.uniform(0.8, 1.5, n))

        # jobs = 0: A := D*A*D (lower)
        A_out, info = mb01ss(0, 1, n, A, d)

        assert info == 0
        expected = np.diag(d) @ A @ np.diag(d)
        assert_allclose(np.tril(A_out), np.tril(expected), rtol=1e-13)


class TestMB01TD:
    """Tests for mb01td - Matrix squaring A := A*A"""

    def test_matrix_squaring(self):
        """Test matrix squaring operation"""
        rng = np.random.default_rng(6782345901)
        n = 5

        A = np.asfortranarray(rng.uniform(-2, 2, (n, n)))
        A_copy = A.copy()

        A_out, info = mb01td(n, A)

        assert info == 0
        expected = A_copy @ A_copy
        assert_allclose(A_out, expected, rtol=1e-12)


class TestMB01UD:
    """Tests for mb01ud - Hessenberg update"""

    def test_hessenberg_update(self):
        """Test Hessenberg matrix update"""
        rng = np.random.default_rng(7893456012)
        n = 6
        alpha, beta = 2.0, 0.5

        H = np.asfortranarray(rng.uniform(-4, 4, (n, n)))
        A = np.asfortranarray(rng.uniform(-4, 4, (n, n)))

        H_out, A_out, info = mb01ud(n, alpha, beta, H, A)

        assert info == 0


class TestMB01UW:
    """Tests for mb01uw - Hessenberg variant"""

    def test_hessenberg_variant(self):
        """Test Hessenberg variant operation"""
        rng = np.random.default_rng(8904567123)
        n = 5
        alpha, beta = 1.5, 1.0

        H = np.asfortranarray(rng.uniform(-3, 3, (n, n)))
        A = np.asfortranarray(rng.uniform(-3, 3, (n, n)))

        H_out, A_out, info = mb01uw(n, alpha, beta, H, A)

        assert info == 0


class TestMB01UX:
    """Tests for mb01ux - Triangular operation"""

    def test_triangular_operation_left_upper(self):
        """Test triangular operation from left, upper"""
        rng = np.random.default_rng(9015678234)
        m, n = 5, 4
        alpha = 2.0

        T = np.triu(rng.uniform(-3, 3, (m, m)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-3, 3, (m, n)))
        dwork = np.asfortranarray(np.zeros(max(m, n)))

        T_out, A_out, info = mb01ux(0, 0, 0, m, n, alpha, T, A, dwork)

        assert info == 0

    def test_triangular_operation_right_lower(self):
        """Test triangular operation from right, lower"""
        rng = np.random.default_rng(1230156789)
        m, n = 4, 6
        alpha = 1.5

        T = np.tril(rng.uniform(-2, 2, (n, n)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-2, 2, (m, n)))
        dwork = np.asfortranarray(np.zeros(max(m, n)))

        T_out, A_out, info = mb01ux(1, 1, 0, m, n, alpha, T, A, dwork)

        assert info == 0


class TestMB01UY:
    """Tests for mb01uy - Real triangular multiplication"""

    def test_left_upper_no_transpose(self):
        """Test left multiplication with upper triangular"""
        rng = np.random.default_rng(2341267890)
        m, n = 5, 4
        alpha = 1.5

        T = np.triu(rng.uniform(-3, 3, (m, m)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-3, 3, (m, n)))
        dwork = np.asfortranarray(np.zeros(max(m, n)))

        T_out, A_out, info = mb01uy(0, 0, 0, m, n, alpha, T, A, dwork)

        assert info == 0

    def test_right_lower_transpose(self):
        """Test right multiplication with lower triangular transpose"""
        rng = np.random.default_rng(3452378901)
        m, n = 6, 5
        alpha = 2.0

        T = np.tril(rng.uniform(-4, 4, (n, n)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-4, 4, (m, n)))
        dwork = np.asfortranarray(np.zeros(max(m, n)))

        T_out, A_out, info = mb01uy(1, 1, 1, m, n, alpha, T, A, dwork)

        assert info == 0


class TestMB01UZ:
    """Tests for mb01uz - Complex triangular multiplication"""

    def test_complex_left_upper(self):
        """Test complex left multiplication with upper triangular"""
        rng = np.random.default_rng(4563489012)
        m, n = 4, 3
        alpha = 1.5 + 0.5j

        T = np.triu(rng.uniform(-3, 3, (m, m)) + 1j * rng.uniform(-3, 3, (m, m)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-3, 3, (m, n)) + 1j * rng.uniform(-3, 3, (m, n)))
        zwork = np.asfortranarray(np.zeros(max(m, n), dtype=np.complex128))

        T_out, A_out, info = mb01uz(0, 0, 0, m, n, alpha, T, A, zwork)

        assert info == 0
        assert T_out.dtype == np.complex128
        assert A_out.dtype == np.complex128

    def test_complex_right_lower(self):
        """Test complex right multiplication with lower triangular"""
        rng = np.random.default_rng(5674590123)
        m, n = 5, 4
        alpha = 2.0 - 1.0j

        T = np.tril(rng.uniform(-2, 2, (n, n)) + 1j * rng.uniform(-2, 2, (n, n)))
        T = np.asfortranarray(T)
        A = np.asfortranarray(rng.uniform(-2, 2, (m, n)) + 1j * rng.uniform(-2, 2, (m, n)))
        zwork = np.asfortranarray(np.zeros(max(m, n), dtype=np.complex128))

        T_out, A_out, info = mb01uz(1, 1, 0, m, n, alpha, T, A, zwork)

        assert info == 0
        assert T_out.dtype == np.complex128
        assert A_out.dtype == np.complex128


class TestMB01VD:
    """Tests for mb01vd - Kronecker product operation"""

    def test_kronecker_product(self):
        """Test Kronecker product computation"""
        rng = np.random.default_rng(6785601234)
        m, n, p = 3, 4, 2

        A = np.asfortranarray(rng.uniform(-3, 3, (m, n)))
        B = np.asfortranarray(rng.uniform(-3, 3, (p, p)))
        C = np.asfortranarray(np.zeros((m * p, n * p)))

        C_out, info = mb01vd(0, m, n, p, A, B, C)

        assert info == 0
        assert C_out.shape == (m * p, n * p)


class TestMB01WD:
    """Tests for mb01wd - Discrete/continuous conversion"""

    def test_discrete_to_continuous(self):
        """Test discrete to continuous conversion"""
        rng = np.random.default_rng(7896712345)
        n = 4

        A = np.asfortranarray(rng.uniform(-2, 2, (n, n)))
        dwork = np.asfortranarray(np.zeros(n))

        A_out, info = mb01wd(0, n, A, dwork)

        assert info == 0

    def test_continuous_to_discrete(self):
        """Test continuous to discrete conversion"""
        rng = np.random.default_rng(8907823456)
        n = 5

        A = np.asfortranarray(rng.uniform(-1, 1, (n, n)))
        dwork = np.asfortranarray(np.zeros(n))

        A_out, info = mb01wd(1, n, A, dwork)

        assert info == 0


class TestMB01XD:
    """Tests for mb01xd - Matrix transpose (out-of-place)"""

    def test_transpose_rectangular(self):
        """Test out-of-place transpose of rectangular matrix"""
        rng = np.random.default_rng(9018934567)
        m, n = 5, 3

        A = np.asfortranarray(rng.uniform(-5, 5, (m, n)))
        B = np.asfortranarray(np.zeros((n, m)))

        B_out, info = mb01xd(m, n, A, B)

        assert info == 0
        expected = A.T
        assert_allclose(B_out, expected, rtol=1e-15)

    def test_transpose_square(self):
        """Test out-of-place transpose of square matrix"""
        rng = np.random.default_rng(1234045678)
        n = 6

        A = np.asfortranarray(rng.uniform(-4, 4, (n, n)))
        B = np.asfortranarray(np.zeros((n, n)))

        B_out, info = mb01xd(n, n, A, B)

        assert info == 0
        expected = A.T
        assert_allclose(B_out, expected, rtol=1e-15)


class TestMB01XY:
    """Tests for mb01xy - Symmetric transpose (in-place)"""

    def test_inplace_transpose_upper(self):
        """Test in-place transpose of upper symmetric matrix"""
        rng = np.random.default_rng(2345156789)
        n = 5

        # Create symmetric matrix (upper)
        A = rng.uniform(-4, 4, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)
        A_copy = A.copy()

        A_out, info = mb01xy(0, n, A)

        assert info == 0
        # For symmetric matrix, transpose should be identity
        assert_allclose(A_out, A_copy, rtol=1e-15)

    def test_inplace_transpose_lower(self):
        """Test in-place transpose of lower symmetric matrix"""
        rng = np.random.default_rng(3456267890)
        n = 4

        # Create symmetric matrix (lower)
        A = rng.uniform(-3, 3, (n, n))
        A = np.tril(A) + np.tril(A, -1).T
        A = np.asfortranarray(A)
        A_copy = A.copy()

        A_out, info = mb01xy(1, n, A)

        assert info == 0
        # For symmetric matrix, transpose should be identity
        assert_allclose(A_out, A_copy, rtol=1e-15)


class TestMB01YD:
    """Tests for mb01yd - Symmetric rank-k update variant"""

    def test_upper_rank_k(self):
        """Test upper symmetric rank-k update"""
        rng = np.random.default_rng(4567378901)
        n, k = 5, 3
        alpha, beta = 1.5, 0.5

        A = rng.uniform(-4, 4, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-4, 4, (n, k)))

        expected = alpha * B @ B.T + beta * A

        A_out, info = mb01yd(0, 0, n, k, alpha, beta, B, A)

        assert info == 0
        assert_allclose(np.triu(A_out), np.triu(expected), rtol=1e-12)

    def test_lower_rank_k_transpose(self):
        """Test lower symmetric rank-k update with transpose"""
        rng = np.random.default_rng(5678489012)
        n, k = 4, 3
        alpha, beta = 2.0, 1.0

        A = rng.uniform(-3, 3, (n, n))
        A = np.tril(A) + np.tril(A, -1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-3, 3, (k, n)))

        expected = alpha * B.T @ B + beta * A

        A_out, info = mb01yd(1, 1, n, k, alpha, beta, B, A)

        assert info == 0
        assert_allclose(np.tril(A_out), np.tril(expected), rtol=1e-12)


class TestMB01ZD:
    """Tests for mb01zd - Final rank-k variant"""

    def test_final_rank_k_upper(self):
        """Test final rank-k variant on upper triangle"""
        rng = np.random.default_rng(6789590123)
        n, k = 6, 4
        alpha, beta = 0.5, 1.5

        A = rng.uniform(-2, 2, (n, n))
        A = np.triu(A) + np.triu(A, 1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-2, 2, (n, k)))

        A_out, info = mb01zd(0, 0, n, k, alpha, beta, B, A)

        assert info == 0

    def test_final_rank_k_lower(self):
        """Test final rank-k variant on lower triangle"""
        rng = np.random.default_rng(7890601234)
        n, k = 5, 3
        alpha, beta = 3.0, 0.25

        A = rng.uniform(-4, 4, (n, n))
        A = np.tril(A) + np.tril(A, -1).T
        A = np.asfortranarray(A)

        B = np.asfortranarray(rng.uniform(-4, 4, (n, k)))

        A_out, info = mb01zd(1, 0, n, k, alpha, beta, B, A)

        assert info == 0
