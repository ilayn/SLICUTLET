import numpy as np
from slicutlet import ma02bd


class TestMA02BD:
    """Test MA02BD - Reverse order of rows and/or columns of a matrix"""

    def test_reverse_rows_only(self):
        """Test reversing rows only (SIDE=0)"""
        # Original matrix
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0]], order='F')

        # Expected: rows in reverse order
        expected = np.array([[10.0, 11.0, 12.0],
                            [7.0, 8.0, 9.0],
                            [4.0, 5.0, 6.0],
                            [1.0, 2.0, 3.0]], order='F')

        result = ma02bd(0, A)
        np.testing.assert_allclose(result, expected)

    def test_reverse_columns_only(self):
        """Test reversing columns only (SIDE=1)"""
        A = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]], order='F')

        # Expected: columns in reverse order
        expected = np.array([[4.0, 3.0, 2.0, 1.0],
                            [8.0, 7.0, 6.0, 5.0],
                            [12.0, 11.0, 10.0, 9.0]], order='F')

        result = ma02bd(1, A)
        np.testing.assert_allclose(result, expected)

    def test_reverse_both(self):
        """Test reversing both rows and columns (SIDE=2)"""
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], order='F')

        # Expected: both rows and columns reversed
        expected = np.array([[9.0, 8.0, 7.0],
                            [6.0, 5.0, 4.0],
                            [3.0, 2.0, 1.0]], order='F')

        result = ma02bd(2, A)
        np.testing.assert_allclose(result, expected)

    def test_single_row_reverse_columns(self):
        """Test with single row matrix, reversing columns"""
        A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], order='F')
        expected = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]], order='F')

        result = ma02bd(1, A)
        np.testing.assert_allclose(result, expected)

    def test_single_column_reverse_rows(self):
        """Test with single column matrix, reversing rows"""
        A = np.array([[1.0], [2.0], [3.0], [4.0]], order='F')
        expected = np.array([[4.0], [3.0], [2.0], [1.0]], order='F')

        result = ma02bd(0, A)
        np.testing.assert_allclose(result, expected)

    def test_square_matrix_both(self):
        """Test with square matrix, reversing both"""
        A = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]], order='F')

        expected = np.array([[16.0, 15.0, 14.0, 13.0],
                            [12.0, 11.0, 10.0, 9.0],
                            [8.0, 7.0, 6.0, 5.0],
                            [4.0, 3.0, 2.0, 1.0]], order='F')

        result = ma02bd(2, A)
        np.testing.assert_allclose(result, expected)

    def test_odd_dimensions_rows(self):
        """Test with odd number of rows"""
        A = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], order='F')

        expected = np.array([[5.0, 6.0],
                            [3.0, 4.0],
                            [1.0, 2.0]], order='F')

        result = ma02bd(0, A)
        np.testing.assert_allclose(result, expected)

    def test_odd_dimensions_columns(self):
        """Test with odd number of columns"""
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], order='F')

        expected = np.array([[3.0, 2.0, 1.0],
                            [6.0, 5.0, 4.0]], order='F')

        result = ma02bd(1, A)
        np.testing.assert_allclose(result, expected)

    def test_even_dimensions(self):
        """Test with even dimensions"""
        A = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]], order='F')

        # Reverse rows
        expected_L = np.array([[5.0, 6.0, 7.0, 8.0],
                              [1.0, 2.0, 3.0, 4.0]], order='F')
        result_L = ma02bd(0, A.copy("F"))
        np.testing.assert_allclose(result_L, expected_L)

        # Reverse columns
        expected_R = np.array([[4.0, 3.0, 2.0, 1.0],
                              [8.0, 7.0, 6.0, 5.0]], order='F')
        result_R = ma02bd(1, A.copy("F"))
        np.testing.assert_allclose(result_R, expected_R)

    def test_2x2_matrix(self):
        """Test with 2x2 matrix"""
        A = np.array([[1.0, 2.0],
                      [3.0, 4.0]], order='F')

        # Both sides
        expected = np.array([[4.0, 3.0],
                            [2.0, 1.0]], order='F')

        result = ma02bd(2, A)
        np.testing.assert_allclose(result, expected)

    def test_double_application_identity(self):
        """Test that applying the operation twice gives back the original"""
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], order='F')

        # Apply twice for each mode
        for side in [0, 1, 2]:
            result = ma02bd(side, A.copy("F"))
            result = ma02bd(side, result)
            np.testing.assert_allclose(result, A,
                                      err_msg=f"Double application with SIDE='{side}' should be identity")

    def test_empty_matrix(self):
        """Test with 0x0 matrix"""
        A = np.array([[]], order='F').reshape(0, 0)
        result = ma02bd(2, A)
        assert result.shape == (0, 0)

    def test_single_element(self):
        """Test with 1x1 matrix"""
        A = np.array([[5.0]], order='F')

        for side in [0, 1, 2]:
            result = ma02bd(side, A.copy("F"))
            np.testing.assert_allclose(result, A)

    def test_rectangular_tall(self):
        """Test with tall rectangular matrix (more rows than columns)"""
        A = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0],
                      [9.0, 10.0]], order='F')

        expected_B = np.array([[10.0, 9.0],
                              [8.0, 7.0],
                              [6.0, 5.0],
                              [4.0, 3.0],
                              [2.0, 1.0]], order='F')

        result = ma02bd(2, A)
        np.testing.assert_allclose(result, expected_B)

    def test_rectangular_wide(self):
        """Test with wide rectangular matrix (more columns than rows)"""
        A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0, 10.0]], order='F')

        expected_B = np.array([[10.0, 9.0, 8.0, 7.0, 6.0],
                              [5.0, 4.0, 3.0, 2.0, 1.0]], order='F')

        result = ma02bd(2, A)
        np.testing.assert_allclose(result, expected_B)

    def test_preserves_values(self):
        """Test that all original values are preserved, just reordered"""
        A = np.random.rand(5, 4)
        A = np.asfortranarray(A)
        original_values = np.sort(A.flatten())

        for side in [0, 1, 2]:
            result = ma02bd(side, A.copy("F"))
            result_values = np.sort(result.flatten())
            np.testing.assert_allclose(result_values, original_values,
                                      err_msg=f"SIDE='{side}' should preserve all values")

    def test_large_matrix(self):
        """Test with larger matrix"""
        A = np.arange(1, 101).reshape(10, 10, order='F').astype(float)

        # Reverse rows
        result_L = ma02bd(0, A.copy("F"))
        np.testing.assert_allclose(result_L[0, :], A[-1, :])
        np.testing.assert_allclose(result_L[-1, :], A[0, :])

        # Reverse columns
        result_R = ma02bd(1, A.copy("F"))
        np.testing.assert_allclose(result_R[:, 0], A[:, -1])
        np.testing.assert_allclose(result_R[:, -1], A[:, 0])
