import numpy as np
import slicutlet


def test_ab05nd_basic_negative_feedback():
    """Test basic negative feedback interconnection."""
    n1, m1, p1 = 2, 1, 1
    n2 = 2
    alpha = -1.0  # Negative feedback

    # System 1: simple plant
    A1 = np.asfortranarray([[-1.0, 0.0], [0.0, -2.0]])
    B1 = np.asfortranarray([[1.0], [1.0]])
    C1 = np.asfortranarray([[1.0, 0.0]])
    D1 = np.asfortranarray([[0.0]])

    # System 2: simple controller
    A2 = np.asfortranarray([[-3.0, 0.0], [0.0, -4.0]])
    B2 = np.asfortranarray([[1.0], [1.0]])
    C2 = np.asfortranarray([[1.0, 0.0]])
    D2 = np.asfortranarray([[0.5]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    # Workspace calculation for over=0: max(1, p1*p1, m1*m1, n1*p1)
    ldwork = max(1, p1 * p1, m1 * m1, n1 * p1)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 0  # No overlap

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0, f"ab05nd failed with info={info}"
    assert n_out == 4

    # With D1=0, D2=0.5, alpha=-1:
    # E21 = (I + alpha*D1*D2)^(-1) = (I + (-1)*0*0.5)^(-1) = I
    # E12 = (I + alpha*D2*D1)^(-1) = (I + (-1)*0.5*0)^(-1) = I
    # So E21*D1 = 0, E21*C1 = C1, E12 = I

    # Check D matrix: E21*D1 = I*0 = 0
    assert np.allclose(D_out, 0.0)

    # Check C matrix: [E21*C1, -alpha*E21*D1*C2] = [C1, 0]
    assert np.allclose(C_out[:, :n1], C1)
    assert np.allclose(C_out[:, n1:], 0.0)

    # Check B matrix: [B1*E12; B2*E21*D1] = [B1; 0]
    assert np.allclose(B_out[:n1, :], B1)
    assert np.allclose(B_out[n1:, :], 0.0)

    # Check A matrix structure
    # A11 = A1 - alpha*B1*E12*D2*C1 = A1 - (-1)*B1*I*0.5*C1 = A1 + 0.5*B1*C1
    expected_A11 = A1 + 0.5 * (B1 @ C1)
    assert np.allclose(A_out[:n1, :n1], expected_A11)

    # A12 = -alpha*B1*E12*C2 = -(-1)*B1*I*C2 = B1*C2
    expected_A12 = B1 @ C2
    assert np.allclose(A_out[:n1, n1:], expected_A12)

    # A21 = B2*E21*C1 = B2*I*C1 = B2*C1
    expected_A21 = B2 @ C1
    assert np.allclose(A_out[n1:, :n1], expected_A21)

    # A22 = A2 - alpha*B2*E21*D1*C2 = A2 - 0 = A2
    assert np.allclose(A_out[n1:, n1:], A2)


def test_ab05nd_positive_feedback():
    """Test positive feedback interconnection."""
    n1, m1, p1 = 2, 1, 1
    n2 = 2
    alpha = 1.0  # Positive feedback

    A1 = np.asfortranarray([[0.0, 1.0], [-2.0, -3.0]])
    B1 = np.asfortranarray([[0.0], [1.0]])
    C1 = np.asfortranarray([[1.0, 0.0]])
    D1 = np.asfortranarray([[0.0]])

    A2 = np.asfortranarray([[-1.0, 0.0], [0.0, -1.0]])
    B2 = np.asfortranarray([[1.0], [0.0]])
    C2 = np.asfortranarray([[2.0, 0.0]])
    D2 = np.asfortranarray([[0.0]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    ldwork = max(1, p1 * p1, m1 * m1, n1 * p1)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0
    assert n_out == 4

    # With D1=0, D2=0, alpha=1: E21=I, E12=I
    assert np.allclose(D_out, 0.0)
    assert np.allclose(C_out[:, :n1], C1)
    assert np.allclose(C_out[:, n1:], 0.0)
    assert np.allclose(B_out[:n1, :], B1)
    assert np.allclose(B_out[n1:, :], 0.0)

    # A11 = A1 - alpha*B1*E12*D2*C1 = A1
    assert np.allclose(A_out[:n1, :n1], A1)

    # A12 = -alpha*B1*E12*C2 = -1*B1*C2 = -B1*C2
    expected_A12 = -(B1 @ C2)
    assert np.allclose(A_out[:n1, n1:], expected_A12)

    # A21 = B2*E21*C1 = B2*C1
    expected_A21 = B2 @ C1
    assert np.allclose(A_out[n1:, :n1], expected_A21)

    # A22 = A2
    assert np.allclose(A_out[n1:, n1:], A2)


def test_ab05nd_nonzero_feedthrough():
    """Test with non-zero D1 and D2 matrices."""
    n1, m1, p1 = 2, 2, 2
    n2 = 2
    alpha = -1.0

    A1 = np.asfortranarray([[1.0, 0.0], [0.0, 2.0]])
    B1 = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])
    C1 = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])
    D1 = np.asfortranarray([[0.1, 0.0], [0.0, 0.1]])

    A2 = np.asfortranarray([[-1.0, 0.0], [0.0, -1.0]])
    B2 = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])
    C2 = np.asfortranarray([[2.0, 0.0], [0.0, 2.0]])
    D2 = np.asfortranarray([[0.2, 0.0], [0.0, 0.2]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    ldwork = max(1, p1 * p1, m1 * m1, n1 * p1)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0
    assert n_out == 4

    # With D1=0.1*I, D2=0.2*I, alpha=-1:
    # I + alpha*D1*D2 = I - 0.1*0.2*I = I - 0.02*I = 0.98*I
    # E21 = (0.98*I)^(-1) = (1/0.98)*I
    E21_scalar = 1.0 / 0.98
    E21_diag = E21_scalar * np.eye(p1)

    # Check D matrix: E21*D1
    expected_D = E21_diag @ D1
    assert np.allclose(D_out, expected_D, rtol=1e-10)

    # Check C matrix: [E21*C1, -alpha*E21*D1*C2]
    expected_C_left = E21_diag @ C1
    expected_C_right = -alpha * (E21_diag @ D1 @ C2)
    assert np.allclose(C_out[:, :n1], expected_C_left, rtol=1e-10)
    assert np.allclose(C_out[:, n1:], expected_C_right, rtol=1e-10)


def test_ab05nd_with_overlap():
    """Test with overlap mode enabled."""
    n1, m1, p1 = 2, 1, 1
    n2 = 2
    alpha = -1.0

    A1 = np.asfortranarray([[-1.0, 0.0], [0.0, -2.0]])
    B1 = np.asfortranarray([[1.0], [1.0]])
    C1 = np.asfortranarray([[1.0, 0.0]])
    D1 = np.asfortranarray([[0.0]])

    A2 = np.asfortranarray([[-3.0, 0.0], [0.0, -4.0]])
    B2 = np.asfortranarray([[1.0], [1.0]])
    C2 = np.asfortranarray([[1.0, 0.0]])
    D2 = np.asfortranarray([[0.5]])

    n = n1 + n2
    # For over=1, need larger workspace: n1*p1 + max(p1*p1, m1*m1, n1*p1)
    ldwork = n1 * p1 + max(p1 * p1, m1 * m1, n1 * p1)
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 1  # Overlap mode

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0
    assert n_out == 4

    # Results should match non-overlap case
    assert np.allclose(D_out, 0.0)
    assert np.allclose(C_out[:, :n1], C1)
    assert np.allclose(C_out[:, n1:], 0.0)
    assert np.allclose(B_out[:n1, :], B1)
    assert np.allclose(B_out[n1:, :], 0.0)


def test_ab05nd_zero_order_system1():
    """Test with zero-order first system (constant plant)."""
    n1, m1, p1 = 0, 2, 2
    n2 = 2
    alpha = -1.0

    # System 1: just feedthrough (no states)
    A1 = np.zeros((0, 0), dtype=np.float64, order='F')
    B1 = np.zeros((0, m1), dtype=np.float64, order='F')
    C1 = np.zeros((p1, 0), dtype=np.float64, order='F')
    D1 = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])

    # System 2: simple controller
    A2 = np.asfortranarray([[-1.0, 0.0], [0.0, -2.0]])
    B2 = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])
    C2 = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])
    D2 = np.asfortranarray([[0.1, 0.0], [0.0, 0.1]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    ldwork = max(1, p1 * p1, m1 * m1, n1 * p1)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0
    assert n_out == n2

    # With n1=0, the system reduces to:
    # I + alpha*D1*D2 = I - 1.0*0.1*I = 0.9*I
    # E21 = (1/0.9)*I
    E21_scalar = 1.0 / 0.9

    # D = E21*D1
    expected_D = E21_scalar * D1
    assert np.allclose(D_out, expected_D, rtol=1e-10)

    # C = [E21*C1, -alpha*E21*D1*C2] = [0, 1*E21*D1*C2] (since C1 is empty)
    expected_C = E21_scalar * (D1 @ C2)
    assert np.allclose(C_out[:, :], expected_C, rtol=1e-10)


def test_ab05nd_zero_order_system2():
    """Test with zero-order second system (constant feedback)."""
    n1, m1, p1 = 2, 1, 1
    n2 = 0
    alpha = -1.0

    # System 1
    A1 = np.asfortranarray([[-1.0, 0.0], [0.0, -2.0]])
    B1 = np.asfortranarray([[1.0], [1.0]])
    C1 = np.asfortranarray([[1.0, 1.0]])
    D1 = np.asfortranarray([[0.2]])  # Changed from 0.5 to avoid singularity

    # System 2: just feedthrough
    A2 = np.zeros((0, 0), dtype=np.float64, order='F')
    B2 = np.zeros((0, p1), dtype=np.float64, order='F')
    C2 = np.zeros((m1, 0), dtype=np.float64, order='F')
    D2 = np.asfortranarray([[2.0]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    ldwork = max(1, p1 * p1, m1 * m1, n1 * p1)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0
    assert n_out == n1

    # With n2=0, I + alpha*D1*D2 = I - 0.2*2.0*I = I - 0.4*I = 0.6*I
    # E21 = (1/0.6)*I
    E21_scalar = 1.0 / 0.6

    # D = E21*D1
    expected_D = E21_scalar * D1
    assert np.allclose(D_out, expected_D, rtol=1e-10)

    # C = E21*C1 (since n2=0, no C2 part)
    expected_C = E21_scalar * C1
    assert np.allclose(C_out, expected_C, rtol=1e-10)

    # B = B1*E12 where E12 = I - alpha*D2*E21*D1 = I - (-1)*2.0*(1/0.6)*0.2*I
    # = I + 2.0*(1/0.6)*0.2*I = I + 0.4/0.6*I = I + (2/3)*I = (5/3)*I
    E12_scalar = 1.0 + 2.0 * (1.0 / 0.6) * 0.2
    expected_B = E12_scalar * B1
    assert np.allclose(B_out, expected_B, rtol=1e-10)

    # A = A1 - alpha*B1*E12*D2*C1 = A1 - (-1)*B*(2.0)*C1 = A1 + 2*B*C1
    # where B = B1*E12 (already computed)
    expected_A = A1 + 2.0 * (expected_B @ C1)
    assert np.allclose(A_out, expected_A, rtol=1e-10)


def test_ab05nd_large_feedback_gain():
    """Test with large feedback gain."""
    n1, m1, p1 = 2, 1, 1
    n2 = 2
    alpha = -10.0  # Large negative feedback

    A1 = np.asfortranarray([[0.0, 1.0], [-1.0, -0.1]])
    B1 = np.asfortranarray([[0.0], [1.0]])
    C1 = np.asfortranarray([[1.0, 0.0]])
    D1 = np.asfortranarray([[0.0]])

    A2 = np.asfortranarray([[-1.0, 0.0], [0.0, -1.0]])
    B2 = np.asfortranarray([[1.0], [0.0]])
    C2 = np.asfortranarray([[1.0, 0.0]])
    D2 = np.asfortranarray([[0.0]])

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    ldwork = max(1, p1 * p1, m1 * m1, n1 * p1)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0
    assert n_out == 4

    # With D1=0, D2=0: E21=I, E12=I regardless of alpha
    assert np.allclose(D_out, 0.0)
    assert np.allclose(C_out[:, :n1], C1)
    assert np.allclose(B_out[:n1, :], B1)

    # A11 = A1
    assert np.allclose(A_out[:n1, :n1], A1)

    # A12 = -alpha*B1*C2 = -(-10)*B1*C2 = 10*B1*C2
    expected_A12 = 10.0 * (B1 @ C2)
    assert np.allclose(A_out[:n1, n1:], expected_A12)


def test_ab05nd_identity_feedback():
    """Test feedback of identity systems."""
    n1, m1, p1 = 2, 2, 2
    n2 = 2
    alpha = -1.0

    # System 1: identity
    A1 = np.asfortranarray(np.zeros((2, 2)))
    B1 = np.asfortranarray(np.eye(2))
    C1 = np.asfortranarray(np.eye(2))
    D1 = np.asfortranarray(np.zeros((2, 2)))

    # System 2: identity
    A2 = np.asfortranarray(np.zeros((2, 2)))
    B2 = np.asfortranarray(np.eye(2))
    C2 = np.asfortranarray(np.eye(2))
    D2 = np.asfortranarray(np.zeros((2, 2)))

    n = n1 + n2
    A = np.zeros((n, n), dtype=np.float64, order='F')
    B = np.zeros((n, m1), dtype=np.float64, order='F')
    C = np.zeros((p1, n), dtype=np.float64, order='F')
    D = np.zeros((p1, m1), dtype=np.float64, order='F')
    iwork = np.zeros(p1, dtype=np.int32)
    ldwork = max(1, p1 * p1, m1 * m1, n1 * p1)
    dwork = np.zeros(ldwork, dtype=np.float64)

    over = 0

    n_out, A_out, B_out, C_out, D_out, info = slicutlet.ab05nd(
        over, n1, m1, p1, n2, alpha,
        A1, B1, C1, D1, A2, B2, C2, D2,
        A, B, C, D, iwork, dwork
    )

    assert info == 0
    assert n_out == 4

    # All D matrices are zero, so E21=I, E12=I
    assert np.allclose(D_out, 0.0)
    assert np.allclose(C_out[:, :n1], C1)
    assert np.allclose(C_out[:, n1:], 0.0)
    assert np.allclose(B_out[:n1, :], B1)
    assert np.allclose(B_out[n1:, :], 0.0)

    # A11 = A1 = 0
    assert np.allclose(A_out[:n1, :n1], 0.0)

    # A12 = -alpha*B1*C2 = -(-1)*I*I = I
    assert np.allclose(A_out[:n1, n1:], np.eye(2))

    # A21 = B2*C1 = I*I = I
    assert np.allclose(A_out[n1:, :n1], np.eye(2))

    # A22 = A2 = 0
    assert np.allclose(A_out[n1:, n1:], 0.0)
