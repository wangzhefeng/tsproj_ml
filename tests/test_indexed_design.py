"""索引设计的列序、切片、只读与延迟物化合同。"""
import unittest
import numpy as np
from forecasting_core.execution.design import IndexedDesign, retained_array_bytes


class IndexedDesignTest(unittest.TestCase):
    def test_distinct_views_of_one_bytes_buffer_are_counted_once(self):
        backing = np.arange(100.0).tobytes()
        left = np.frombuffer(backing, dtype=float, count=10)
        right = np.frombuffer(backing, dtype=float, count=10, offset=80)
        self.assertEqual(retained_array_bytes([left, right]), len(backing))

    def test_readonly_view_cannot_alias_mutable_input_storage(self):
        original = np.arange(10.0)
        view = original[:]
        view.flags.writeable = False
        design = IndexedDesign(((view, 0),), start=0, stop=4)
        original[0] = 999
        np.testing.assert_array_equal(np.asarray(design)[:, 0], np.arange(4.0))
        with self.assertRaises(ValueError):
            design.columns[0][0].flags.writeable = True

    def test_retained_bytes_counts_backing_storage_once(self):
        original = np.arange(100.0)
        original.flags.writeable = False
        design = IndexedDesign(((original[:10], 0), (original[10:20], 0)), start=0, stop=5)
        blocks = [v for v, _ in design.columns]
        self.assertEqual(design.retained_bytes, retained_array_bytes(blocks))
        sliced = design[1:3]
        self.assertIs(sliced.columns[0][0], design.columns[0][0])

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
