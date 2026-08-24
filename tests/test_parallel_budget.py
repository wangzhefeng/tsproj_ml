# -*- coding: utf-8 -*-
"""窗口并行下内部并行预算测试。"""

import unittest
from types import SimpleNamespace

from utils.parallel_budget import apply_window_parallel_budget


class ParallelBudgetTest(unittest.TestCase):
    def test_window_parallelism_forces_every_internal_dimension_to_one(self):
        args = SimpleNamespace(
            multi_output_n_jobs=8,
            quantile_parallel_workers=3,
            ensemble_parallel_workers=4,
            model_thread_count=6,
        )

        apply_window_parallel_budget(args, window_workers=8)

        self.assertEqual(args.multi_output_n_jobs, 1)
        self.assertEqual(args.quantile_parallel_workers, 1)
        self.assertEqual(args.ensemble_parallel_workers, 1)
        self.assertEqual(args.model_thread_count, 1)

    def test_serial_windows_preserve_internal_parallelism(self):
        args = SimpleNamespace(
            multi_output_n_jobs=4,
            quantile_parallel_workers=3,
            ensemble_parallel_workers=2,
            model_thread_count=1,
        )

        apply_window_parallel_budget(args, window_workers=1)

        self.assertEqual(args.multi_output_n_jobs, 4)
        self.assertEqual(args.quantile_parallel_workers, 3)
        self.assertEqual(args.ensemble_parallel_workers, 2)
        self.assertEqual(args.model_thread_count, 1)


if __name__ == "__main__":
    unittest.main()
