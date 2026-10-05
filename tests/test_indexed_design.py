"""索引设计的列序、切片、只读与延迟物化合同。"""
import unittest
import numpy as np
from forecasting_core.design import IndexedDesign


class IndexedDesignTest(unittest.TestCase):
    def test_gathers_columns_only_when_consumed_and_slices_rows(self):
        values = np.arange(20, dtype=float)
        matrix = IndexedDesign(((values, 2), (values, 5)), start=1, stop=7)
        self.assertEqual(matrix.shape, (6, 2))
        self.assertLess(matrix.retained_bytes, matrix.nbytes + values.nbytes)
        expected = np.column_stack((np.arange(3, 9), np.arange(6, 12)))
        np.testing.assert_array_equal(np.asarray(matrix), expected)
        np.testing.assert_array_equal(np.asarray(matrix[1:4]), expected[1:4])
        np.testing.assert_array_equal(matrix[:, 1], expected[:, 1])
        np.testing.assert_array_equal(np.asarray(matrix[[4, 1]]), expected[[4, 1]])
        with self.assertRaises(ValueError):
            matrix.columns[0][0][0] = 99
        self.assertTrue(matrix.is_finite())

    def test_out_of_bounds_and_nonfinite_values_are_not_silently_filled(self):
        with self.assertRaises(ValueError):
            IndexedDesign(((np.arange(3.0), 2),), start=0, stop=2)
        matrix = IndexedDesign(((np.array([1.0, np.nan, 2.0]), 0),), start=0, stop=3)
        self.assertFalse(matrix.is_finite())

    def test_empty_and_negative_slice_match_numpy(self):
        matrix = IndexedDesign(((np.arange(8.0), 0),), start=1, stop=7)
        for selector in (slice(0, 0), slice(-2, None), slice(None, None, -1)):
            np.testing.assert_array_equal(np.asarray(matrix[selector]), np.asarray(matrix)[selector])
