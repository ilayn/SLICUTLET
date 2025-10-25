import numpy as np
from numpy.testing import assert_allclose
from slicutlet import ab04md


class TestAB04MD:
    """Tests for ab04md: bilinear transformation of state-space systems."""

    def test_discrete_to_continuous_simple(self):
        """Test discrete-time to continuous-time conversion with simple system."""
        n, m, p = 2, 1, 1
        A = np.asfortranarray([[0.5, 0.0], [0.0, 0.8]])
        B = np.asfortranarray([[1.0], [0.5]])
        C = np.asfortranarray([[1.0, 0.0]])
        D = np.asfortranarray([[0.0]])

        alpha = 1.0
        beta = 1.0

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        A_c, B_c, C_c, D_c, info = ab04md(0, n, m, p, alpha, beta, A, B, C, D, iwork, dwork)

        assert info == 0, f"Expected info=0, got {info}"
        assert A_c.shape == (2, 2)
        assert B_c.shape == (2, 1)
        assert C_c.shape == (1, 2)
        assert D_c.shape == (1, 1)

    def test_continuous_to_discrete_simple(self):
        """Test continuous-time to discrete-time conversion with simple system."""
        n, m, p = 2, 1, 1
        A = np.asfortranarray([[-1.0, 0.0], [0.0, -2.0]])
        B = np.asfortranarray([[1.0], [0.5]])
        C = np.asfortranarray([[1.0, 1.0]])
        D = np.asfortranarray([[0.0]])

        alpha = 1.0
        beta = 1.0

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        A_d, B_d, C_d, D_d, info = ab04md(1, n, m, p, alpha, beta, A, B, C, D, iwork, dwork)

        assert info == 0, f"Expected info=0, got {info}"
        assert A_d.shape == (2, 2)
        assert B_d.shape == (2, 1)
        assert C_d.shape == (1, 2)
        assert D_d.shape == (1, 1)

    def test_roundtrip_conversion(self):
        """Test that discrete->continuous->discrete roundtrip recovers original system."""
        n, m, p = 2, 1, 1
        A_orig = np.asfortranarray([[0.6, 0.1], [-0.1, 0.7]])
        B_orig = np.asfortranarray([[1.0], [0.8]])
        C_orig = np.asfortranarray([[0.5, 1.0]])
        D_orig = np.asfortranarray([[0.2]])

        alpha = 1.0
        beta = 1.0

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        # Convert discrete to continuous
        A_c, B_c, C_c, D_c, info = ab04md(0, n, m, p, alpha, beta,
                                              A_orig.copy(), B_orig.copy(),
                                              C_orig.copy(), D_orig.copy(), iwork, dwork)
        assert info == 0

        # Convert back to discrete
        A_d, B_d, C_d, D_d, info = ab04md(1, n, m, p, alpha, beta,
                                              A_c, B_c, C_c, D_c, iwork, dwork)
        assert info == 0

        # Should recover original system (within numerical tolerance)
        assert_allclose(A_d, A_orig, rtol=1e-10, atol=1e-12)
        assert_allclose(B_d, B_orig, rtol=1e-10, atol=1e-12)
        assert_allclose(C_d, C_orig, rtol=1e-10, atol=1e-12)
        assert_allclose(D_d, D_orig, rtol=1e-10, atol=1e-12)

    def test_zero_state_system(self):
        """Test with zero-state system (n=0)."""
        n, m, p = 0, 2, 2
        A = np.asfortranarray(np.zeros((0, 0)))
        B = np.asfortranarray(np.zeros((0, 2)))
        C = np.asfortranarray([[1.0, 0.5], [0.0, 1.0]])
        D = np.asfortranarray([[0.1, 0.2], [0.3, 0.4]])

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        A_new, B_new, C_new, D_new, info = ab04md(0, n, m, p, 1.0, 1.0, A, B, C, D, iwork, dwork)

        assert info == 0
        assert A_new.shape == (0, 0)
        assert B_new.shape == (0, 2)

    def test_zero_input_system(self):
        """Test with zero-input system (m=0)."""
        n, m, p = 2, 0, 1
        A = np.asfortranarray([[0.5, 0.0], [0.0, 0.8]])
        B = np.asfortranarray(np.zeros((2, 0)))
        C = np.asfortranarray(np.zeros((1, 2)))
        D = np.asfortranarray(np.zeros((1, 0)))

        iwork = np.zeros(4 * max(m, p, 1), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        A_new, B_new, C_new, D_new, info = ab04md(0, n, m, p, 1.0, 1.0, A, B, C, D, iwork, dwork)

        assert info == 0
        assert B_new.shape == (2, 0)
        assert D_new.shape == (1, 0)

    def test_zero_output_system(self):
        """Test with zero-output system (p=0)."""
        n, m, p = 2, 1, 0
        A = np.asfortranarray([[0.5, 0.0], [0.0, 0.8]])
        B = np.asfortranarray([[1.0], [0.5]])
        C = np.asfortranarray(np.zeros((0, 2)))
        D = np.asfortranarray(np.zeros((0, 1)))

        iwork = np.zeros(4 * max(m, p, 1), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        A_new, B_new, C_new, D_new, info = ab04md(0, n, m, p, 1.0, 1.0, A, B, C, D, iwork, dwork)

        assert info == 0
        assert C_new.shape == (0, 2)
        assert D_new.shape == (0, 1)

    def test_singular_matrix_discrete(self):
        """Test with singular (alpha*I + A) matrix in discrete-time."""
        n, m, p = 2, 1, 1
        alpha = 1.0
        A = np.asfortranarray([[-1.0, 0.0], [0.0, -1.0]])
        B = np.asfortranarray([[1.0], [0.5]])
        C = np.asfortranarray([[1.0, 0.0]])
        D = np.asfortranarray([[0.0]])

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        _, _, _, _, info = ab04md(0, n, m, p, alpha, 1.0, A, B, C, D, iwork, dwork)

        assert info == 1, f"Expected info=1 (singular matrix), got {info}"

    def test_singular_matrix_continuous(self):
        """Test with singular (beta*I - A) matrix in continuous-time."""
        n, m, p = 2, 1, 1
        beta = 1.0
        A = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])
        B = np.asfortranarray([[1.0], [0.5]])
        C = np.asfortranarray([[1.0, 0.0]])
        D = np.asfortranarray([[0.0]])

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        _, _, _, _, info = ab04md(1, n, m, p, 1.0, beta, A, B, C, D, iwork, dwork)

        assert info == 2, f"Expected info=2 (singular matrix), got {info}"

    def test_different_alpha_beta(self):
        """Test with non-standard alpha and beta values."""
        n, m, p = 2, 1, 1
        A = np.asfortranarray([[0.5, 0.0], [0.0, 0.8]])
        B = np.asfortranarray([[1.0], [0.5]])
        C = np.asfortranarray([[1.0, 0.0]])
        D = np.asfortranarray([[0.1]])

        alpha = 2.0
        beta = 0.5

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        A_c, B_c, C_c, D_c, info = ab04md(0, n, m, p, alpha, beta, A, B, C, D, iwork, dwork)

        assert info == 0
        assert A_c.shape == (2, 2)

    def test_mimo_system(self):
        """Test with multi-input multi-output system."""
        n, m, p = 3, 2, 2
        A = np.asfortranarray([[0.5, 0.1, 0.0],
                      [0.0, 0.7, 0.2],
                      [0.0, 0.0, 0.6]])
        B = np.asfortranarray([[1.0, 0.5],
                      [0.8, 0.3],
                      [0.2, 0.9]])
        C = np.asfortranarray([[1.0, 0.0, 0.5],
                      [0.0, 1.0, 0.3]])
        D = np.asfortranarray([[0.1, 0.0],
                      [0.0, 0.2]])

        iwork = np.zeros(4 * max(m, p), dtype=np.int32)
        dwork = np.zeros(max(1, n), dtype=np.float64)

        A_c, B_c, C_c, D_c, info = ab04md(0, n, m, p, 1.0, 1.0, A, B, C, D, iwork, dwork)

        assert info == 0
        assert A_c.shape == (3, 3)
        assert B_c.shape == (3, 2)
        assert C_c.shape == (2, 3)
        assert D_c.shape == (2, 2)
