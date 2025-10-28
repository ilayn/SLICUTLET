"""Tests for ab01nd: Controllability Staircase Form"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from slicutlet import ab01nd


class TestAB01NDFullyControllable:
    """Test cases where system is fully controllable"""

    def test_simple_controllable_system(self):
        """Test a simple 2x2 fully controllable system"""
        # Simple companion form system - fully controllable
        n, m = 2, 1
        A = np.array([[0.0, 1.0],
                      [-2.0, -3.0]], order='F')
        B = np.array([[0.0],
                      [1.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2  # 'I' - compute Z
        tol = 0.0  # Use default tolerance

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == n  # Fully controllable
        assert indcon >= 1

        # Z should be orthogonal
        ZtZ = Z_out.T @ Z_out
        assert_allclose(ZtZ, np.eye(n), atol=1e-10)

        # Verify that Z'*B has correct structure (first block nonzero)
        # B_out is already Z'*B from the algorithm
        assert np.linalg.norm(B_out[:nblk_out[0], :]) > 1e-10


    def test_multi_input_controllable(self):
        """Test a multi-input fully controllable system"""
        n, m = 3, 2
        A = np.array([[1.0, 2.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 2.0]], order='F')
        B = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2  # 'I'
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == n  # Fully controllable

        # Z should be orthogonal
        ZtZ = Z_out.T @ Z_out
        assert_allclose(ZtZ, np.eye(n), atol=1e-10)


class TestAB01NDPartiallyControllable:
    """Test cases where system has uncontrollable modes"""

    def test_partially_controllable_system(self):
        """Test a system with both controllable and uncontrollable modes"""
        n, m = 3, 1
        # System with one uncontrollable mode
        A = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 1.0],
                      [0.0, 0.0, 2.0]], order='F')
        B = np.array([[0.0],
                      [1.0],
                      [0.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2  # 'I'
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont < n  # Not fully controllable
        assert ncont >= 1  # At least one controllable state

        # Z should be orthogonal
        ZtZ = Z_out.T @ Z_out
        assert_allclose(ZtZ, np.eye(n), atol=1e-10)


    def test_single_controllable_state(self):
        """Test system with only one controllable state"""
        n, m = 3, 1
        # Most states uncontrollable
        A = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]], order='F')
        B = np.array([[1.0],
                      [0.0],
                      [0.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2  # 'I'
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == 1  # Only one controllable state

        # Z should be orthogonal
        ZtZ = Z_out.T @ Z_out
        assert_allclose(ZtZ, np.eye(n), atol=1e-10)


class TestAB01NDJobzOptions:
    """Test different JOBZ options"""

    def test_jobz_n_no_accumulation(self):
        """Test JOBZ='N' - no transformation accumulation"""
        n, m = 2, 1
        A = np.array([[0.0, 1.0],
                      [-2.0, -3.0]], order='F')
        B = np.array([[0.0],
                      [1.0]], order='F')

        Z = np.zeros((1, 1), order='F')  # Can be minimal size when JOBZ='N'
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 0  # 'N' - don't accumulate
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == n  # Still fully controllable


    def test_jobz_f_factored_form(self):
        """Test JOBZ='F' - store transformations in factored form"""
        n, m = 2, 1
        A = np.array([[0.0, 1.0],
                      [-2.0, -3.0]], order='F')
        B = np.array([[0.0],
                      [1.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 1  # 'F' - factored form
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == n
        # In factored form, tau contains the scalar factors


class TestAB01NDEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_system(self):
        """Test with n=0 (empty system)"""
        n, m = 0, 1
        A = np.zeros((1, 1), order='F')
        B = np.zeros((1, 1), order='F')
        Z = np.zeros((1, 1), order='F')
        tau = np.zeros(1)
        nblk = np.zeros(1, dtype=np.int32)
        iwork = np.zeros(max(1, m), dtype=np.int32)
        dwork = np.zeros(max(1, 3*m))

        jobz = 2
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == 0
        assert indcon == 0


    def test_zero_input_matrix(self):
        """Test with B=0 (no inputs can control the system)"""
        n, m = 3, 1
        A = np.array([[1.0, 2.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 2.0]], order='F')
        B = np.zeros((n, m), order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == 0  # No controllable states
        assert indcon == 0

        # Z should be orthogonal
        ZtZ = Z_out.T @ Z_out
        assert_allclose(ZtZ, np.eye(n), atol=1e-10)


    def test_single_state_single_input(self):
        """Test minimal meaningful system (1x1)"""
        n, m = 1, 1
        A = np.array([[2.0]], order='F')
        B = np.array([[1.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == 1  # Fully controllable
        assert indcon == 1


    def test_custom_tolerance(self):
        """Test with custom tolerance setting"""
        n, m = 2, 1
        A = np.array([[0.0, 1.0],
                      [-2.0, -3.0]], order='F')
        B = np.array([[0.0],
                      [1.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2
        tol = 1e-10  # Custom tolerance

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == n


class TestAB01NDTransformationProperties:
    """Test mathematical properties of the transformation"""

    def test_transformation_preserves_eigenvalues(self):
        """Test that eigenvalues of A are preserved"""
        n, m = 3, 1
        A = np.array([[1.0, 2.0, 0.0],
                      [0.0, 3.0, 1.0],
                      [0.0, 0.0, 2.0]], order='F')
        B = np.array([[1.0],
                      [1.0],
                      [0.0]], order='F')

        # Store original eigenvalues
        eig_orig = np.sort(np.linalg.eigvals(A))

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0

        # Eigenvalues should be preserved (similarity transformation)
        eig_transformed = np.sort(np.linalg.eigvals(A_out[:n, :n]))
        assert_allclose(eig_orig, eig_transformed, rtol=1e-10)


    def test_block_structure_properties(self):
        """Test that the block structure is correctly formed"""
        n, m = 4, 1
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [-1.0, -2.0, -3.0, -4.0]], order='F')
        B = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [1.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == n  # Should be fully controllable

        # Verify block sizes sum to ncont
        block_sum = sum(nblk_out[i] for i in range(indcon))
        assert block_sum == ncont

    def test_block_structure_properties_MIMO(self):
        """Test that the block structure is correctly formed"""
        n, m = 4, 2
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [-1.0, -2.0, -3.0, -4.0]], order='F')
        B = np.array([[0.0, 1.0],
                      [0.0, 0.0],
                      [0.0, 0.0],
                      [1.0, 0.0]], order='F')

        Z = np.zeros((n, n), order='F')
        tau = np.zeros(n)
        nblk = np.zeros(n, dtype=np.int32)
        iwork = np.zeros(m, dtype=np.int32)
        dwork = np.zeros(max(1, n, 3*m))

        jobz = 2
        tol = 0.0

        A_out, B_out, Z_out, tau_out, nblk_out, ncont, indcon, info = ab01nd(
            jobz, n, m, tol, A, B, Z, tau, nblk, iwork, dwork
        )

        assert info == 0
        assert ncont == n  # Should be fully controllable
        assert_array_equal(nblk_out, [2, 1, 1, 0])
        assert indcon == 3

        # Verify block sizes sum to ncont
        block_sum = sum(nblk_out[i] for i in range(indcon))
        assert block_sum == ncont
