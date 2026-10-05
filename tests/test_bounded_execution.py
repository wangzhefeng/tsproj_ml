"""回测执行必须按序消费且限制未消费的任务数。"""
import unittest
from model_testing.loops.execution import ordered_bounded_map


class BoundedExecutionTest(unittest.TestCase):
    def test_input_is_lazy_and_inflight_is_bounded(self):
        pulled = []
        def inputs():
            for index in range(9):
                pulled.append(index)
                yield index
        results = ordered_bounded_map(lambda value: value * 2, inputs(), workers=2)
        self.assertEqual(pulled, [])
        self.assertEqual(next(results), 0)
        self.assertLessEqual(len(pulled), 2)
        self.assertEqual(list(results), list(range(2, 18, 2)))

    def test_serial_is_lazy_and_errors_propagate(self):
        def fail(value):
            if value == 1:
                raise ValueError("fit failed")
            return value
        results = ordered_bounded_map(fail, range(3), workers=1)
        self.assertEqual(next(results), 0)
        with self.assertRaisesRegex(ValueError, "fit failed"):
            next(results)
