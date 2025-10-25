import numpy as np
import slicutlet


def test_ab05md_basic_cascade_lower():
    """Test basic cascade with lower block diagonal structure."""
    n1, m1, p1 = 2, 1, 1
    n2, p2 = 2, 1

    # System 1: simple integrator-like system
    A1 = np.asfortranarray([[-1.0, 1.0], [3.0, 4.0]])
    B1 = np.asfortranarray([[-7.0], [-1.0]])
    C1 = np.asfortranarray([[0.125, 0.0]])
    D1 = np.asfortranarray([[0.0]])

    # System 2: another simple system
    A2 = np.asfortranarray([[0.90, 0.10], [3.123, -5.0]])
    B2 = np.asfortranarray([[0.0], [1.0]])
    C2 = np.asfortranarray([[1.0, 0.0]])
    D2 = np.asfortranarray([[0.0]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 0  # Lower block diagonal
    over = 0  # No overlap

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == 0, f"ab05md failed with info={info}"
    assert n_out == 4

    # Check A matrix structure: lower block diagonal with A1, A2, and coupling
    # A1 should be in top-left
    assert np.allclose(A_out[:n1, :n1], A1)
    # A2 should be in bottom-right
    assert np.allclose(A_out[n1:, n1:], A2)
    # Top-right should be zero
    assert np.allclose(A_out[:n1, n1:], 0.0)
    # Bottom-left should have coupling: B2 * C1
    expected_coupling = B2 @ C1  # (2,1) @ (1,2) = (2,2)
    assert np.allclose(A_out[n1:, :n1], expected_coupling)

    # Check B matrix: B1 on top, B2*D1 on bottom
    assert np.allclose(B_out[:n1, :], B1)
    expected_b2 = B2 @ D1  # (2,1) @ (1,1) = (2,1)
    assert np.allclose(B_out[n1:, :], expected_b2)

    # Check C matrix: D2*C1 on left, C2 on right
    expected_c1 = D2 @ C1  # (1,1) @ (1,2) = (1,2)
    assert np.allclose(C_out[:, :n1], expected_c1)
    assert np.allclose(C_out[:, n1:], C2)

    # Check D matrix: D2*D1
    expected_d = D2 @ D1  # (1,1) @ (1,1) = (1,1)
    assert np.allclose(D_out, expected_d)


def test_ab05md_slicot_example():
    """Test with modified SLICOT example data."""
    n1, m1, p1 = 2, 2, 2
    n2, p2 = 2, 2

    # System 1
    A1 = np.asfortranarray([[1.0, 0.0], [0.0, -1.0]])
    B1 = np.asfortranarray([[1.0, 1.0], [2.0, 0.0]])
    C1 = np.asfortranarray([[3.0, -2.0], [0.0, 1.0]])
    D1 = np.asfortranarray([[1.0, 0.0], [1.0, 0.0]])

    # System 2
    A2 = np.asfortranarray([[-3.0, 0.0], [1.0, 0.0]])
    B2 = np.asfortranarray([[0.0, -1.0], [1.0, 0.0]])
    C2 = np.asfortranarray([[1.0, 1.0], [1.0, 1.0]])
    D2 = np.asfortranarray([[1.0, 1.0], [0.0, 1.0]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 0  # Lower
    over = 0  # No overlap

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == 0
    assert n_out == 4

    # Verify A matrix structure
    assert np.allclose(A_out[:n1, :n1], A1)
    assert np.allclose(A_out[n1:, n1:], A2)
    assert np.allclose(A_out[:n1, n1:], 0.0)
    expected_A_coupling = B2 @ C1
    assert np.allclose(A_out[n1:, :n1], expected_A_coupling)

    # Verify B matrix
    assert np.allclose(B_out[:n1, :], B1)
    expected_B2 = B2 @ D1
    assert np.allclose(B_out[n1:, :], expected_B2)

    # Verify C matrix
    expected_C1 = D2 @ C1
    assert np.allclose(C_out[:, :n1], expected_C1)
    assert np.allclose(C_out[:, n1:], C2)

    # Verify D matrix
    expected_D = D2 @ D1
    assert np.allclose(D_out, expected_D)


def test_ab05md_upper_block():
    """Test upper block diagonal structure."""
    n1, m1, p1 = 2, 3, 1
    n2, p2 = 2, 2

    A1 = np.asfortranarray(np.eye(2) * 0.5)
    B1 = np.asfortranarray(np.ones((2, 3)))
    C1 = np.asfortranarray(np.ones((1, 2)) * 2.0)
    D1 = np.asfortranarray(np.zeros((1, 3)))

    A2 = np.asfortranarray(np.eye(2) * 0.3)
    B2 = np.asfortranarray(np.ones((2, 1)) * 3.0)
    C2 = np.asfortranarray(np.ones((2, 2)))
    D2 = np.asfortranarray(np.array([[1.0], [1.0]]))

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 1  # Upper block diagonal
    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == 0
    assert n_out == 4

    # For upper block: A2 is top-left, A1 is bottom-right
    assert np.allclose(A_out[:n2, :n2], A2)
    assert np.allclose(A_out[n2:, n2:], A1)
    # Coupling is in top-right: B2*C1
    expected_coupling = B2 @ C1
    assert np.allclose(A_out[:n2, n2:], expected_coupling)
    # Bottom-left should be zero
    assert np.allclose(A_out[n2:, :n2], 0.0)


def test_ab05md_with_overlap():
    """Test with overlap mode enabled."""
    n1, m1, p1 = 2, 2, 2
    n2, p2 = 2, 2

    A1 = np.asfortranarray(np.eye(2) * 0.5)
    B1 = np.asfortranarray(np.ones((2, 2)))
    C1 = np.asfortranarray(np.ones((2, 2)))
    D1 = np.asfortranarray(np.zeros((2, 2)))

    A2 = np.asfortranarray(np.eye(2) * 0.3)
    B2 = np.asfortranarray(np.ones((2, 2)))
    C2 = np.asfortranarray(np.ones((2, 2)))
    D2 = np.asfortranarray(np.eye(2))

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 0
    over = 1  # Overlap mode

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == 0
    assert n_out == 4

    # Results should be the same as no-overlap, just uses workspace differently
    assert np.allclose(A_out[:n1, :n1], A1)
    assert np.allclose(A_out[n1:, n1:], A2)
    expected_coupling = B2 @ C1
    assert np.allclose(A_out[n1:, :n1], expected_coupling)


def test_ab05md_identity_systems():
    """Test cascade of two identity-like systems."""
    n1, m1, p1 = 2, 2, 2
    n2, p2 = 2, 2

    # System 1: pure delay (identity matrices)
    A1 = np.asfortranarray(np.zeros((2, 2)))
    B1 = np.asfortranarray(np.eye(2))
    C1 = np.asfortranarray(np.eye(2))
    D1 = np.asfortranarray(np.zeros((2, 2)))

    # System 2: another pure delay
    A2 = np.asfortranarray(np.zeros((2, 2)))
    B2 = np.asfortranarray(np.eye(2))
    C2 = np.asfortranarray(np.eye(2))
    D2 = np.asfortranarray(np.zeros((2, 2)))

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 0
    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == 0
    assert n_out == 4

    # A matrix should be all zeros
    assert np.allclose(A_out[:n1, :n1], 0.0)
    assert np.allclose(A_out[n1:, n1:], 0.0)
    # Coupling: B2 @ C1 = I @ I = I
    assert np.allclose(A_out[n1:, :n1], np.eye(2))

    # B: B1 on top, B2@D1=I@0=0 on bottom
    assert np.allclose(B_out[:n1, :], B1)
    assert np.allclose(B_out[n1:, :], 0.0)

    # C: D2@C1=0@I=0 on left, C2=I on right
    assert np.allclose(C_out[:, :n1], 0.0)
    assert np.allclose(C_out[:, n1:], C2)

    # D: D2@D1 = 0@0 = 0
    assert np.allclose(D_out, 0.0)


def test_ab05md_zero_dimensions():
    """Test with one system having zero states."""
    n1, m1, p1 = 0, 1, 1
    n2, p2 = 2, 1

    A1 = np.asfortranarray(np.zeros((1, 1)))
    B1 = np.asfortranarray(np.zeros((1, 1)))
    C1 = np.asfortranarray(np.zeros((1, 1)))
    D1 = np.asfortranarray([[2.0]])  # Non-zero feedthrough

    A2 = np.asfortranarray(np.eye(2))
    B2 = np.asfortranarray([[1.0], [2.0]])
    C2 = np.asfortranarray([[1.0, 1.0]])
    D2 = np.asfortranarray([[0.0]])

    n = n1 + n2
    A = np.zeros((max(1, n), max(1, n)), dtype=np.float64, order='F')
    B = np.zeros((max(1, n), m1), dtype=np.float64, order='F')
    C = np.zeros((p2, max(1, n)), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 0
    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == 0
    assert n_out == 2

    # A should just be A2 (since n1=0)
    assert np.allclose(A_out[:n2, :n2], A2)

    # B should be B2*D1 = [[1],[2]] * 2 = [[2],[4]]
    expected_B = B2 * D1[0, 0]
    assert np.allclose(B_out[:n2, :], expected_B)

    # C should just be C2
    assert np.allclose(C_out[:, :n2], C2)

    # D should be D2*D1 = 0*2 = 0
    assert np.allclose(D_out, 0.0)


def test_ab05md_cascade_property():
    """Test that cascade preserves expected system structure."""
    n1, m1, p1 = 2, 1, 1
    n2, p2 = 2, 1

    # System 1 with distinct eigenvalues
    A1 = np.asfortranarray([[-1.0, 0.0], [0.0, -2.0]])
    B1 = np.asfortranarray([[1.0], [1.0]])
    C1 = np.asfortranarray([[1.0, 1.0]])
    D1 = np.asfortranarray([[0.0]])

    # System 2 with distinct eigenvalues
    A2 = np.asfortranarray([[-3.0, 0.0], [0.0, -4.0]])
    B2 = np.asfortranarray([[1.0], [1.0]])
    C2 = np.asfortranarray([[1.0, 1.0]])
    D2 = np.asfortranarray([[0.0]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 0
    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == 0
    assert n_out == 4

    # Verify block diagonal structure is preserved
    assert np.allclose(A_out[:n1, :n1], A1)
    assert np.allclose(A_out[n1:, n1:], A2)

    # Verify coupling term
    expected_coupling = B2 @ C1  # [[1],[1]] @ [[1,1]] = [[1,1],[1,1]]
    assert np.allclose(A_out[n1:, :n1], expected_coupling)

    # For D1=0, B should be [B1; 0]
    assert np.allclose(B_out[:n1, :], B1)
    assert np.allclose(B_out[n1:, :], 0.0)

    # For D2=0, C should be [0, C2]
    assert np.allclose(C_out[:, :n1], 0.0)
    assert np.allclose(C_out[:, n1:], C2)

    # D should be 0
    assert np.allclose(D_out, 0.0)


def test_ab05md_invalid_uplo():
    """Test error handling for invalid uplo parameter."""
    n1, m1, p1 = 2, 1, 1
    n2, p2 = 2, 1

    A1 = np.asfortranarray(np.eye(2))
    B1 = np.asfortranarray(np.ones((2, 1)))
    C1 = np.asfortranarray(np.ones((1, 2)))
    D1 = np.asfortranarray(np.zeros((1, 1)))

    A2 = np.asfortranarray(np.eye(2))
    B2 = np.asfortranarray(np.ones((2, 1)))
    C2 = np.asfortranarray(np.ones((1, 2)))
    D2 = np.asfortranarray(np.zeros((1, 1)))

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 5  # Invalid value
    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == -1


def test_ab05md_invalid_over():
    """Test error handling for invalid over parameter."""
    n1, m1, p1 = 2, 1, 1
    n2, p2 = 2, 1

    A1 = np.asfortranarray(np.eye(2))
    B1 = np.asfortranarray(np.ones((2, 1)))
    C1 = np.asfortranarray(np.ones((1, 2)))
    D1 = np.asfortranarray(np.zeros((1, 1)))

    A2 = np.asfortranarray(np.eye(2))
    B2 = np.asfortranarray(np.ones((2, 1)))
    C2 = np.asfortranarray(np.ones((1, 2)))
    D2 = np.asfortranarray(np.zeros((1, 1)))

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p2, n), dtype=np.float64, order='F')
    D = np.zeros((p2, m1), dtype=np.float64, order='F')
    dwork = np.zeros(max(1, p1 * max(n1, m1, n2, p2)), dtype=np.float64)

    uplo = 0
    over = 3  # Invalid value

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05md(
        uplo, over, n1, m1, p1, n2, p2,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, dwork
    )

    assert info == -2
