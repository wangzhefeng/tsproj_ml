"""Regression gates for nested batch resource limits."""
from dataclasses import replace
import unittest
import numpy as np

from forecasting_core.runtime_resources import RuntimeResourceBudget
from model_performance.transform_cache import FoldTransformCache


class BatchResourceLimitsTest(unittest.TestCase):
    def test_transform_cache_evicts_and_does_not_retain_oversized_entry(self):
        cache = FoldTransformCache(max_bytes=4096, max_entries=2)
        value = np.ones(128)
        self.assertIs(cache.get_or_create("a", lambda: value), value)
        cache.get_or_create("b", lambda: value.copy())
        cache.get_or_create("c", lambda: value.copy())
        self.assertEqual(cache.payload()["entries"], 2)
        self.assertGreaterEqual(cache.payload()["evictions"], 1)
        self.assertLessEqual(cache.payload()["resident_bytes"], 4096)
        large = np.ones(8192)
        self.assertIs(cache.get_or_create("large", lambda: large), large)
        self.assertEqual(cache.payload()["oversized_skips"], 1)
        self.assertLessEqual(cache.payload()["resident_bytes"], 4096)

    def test_parent_with_no_available_thread_is_rejected(self):
        budget = RuntimeResourceBudget(8, 8, 8, 1_000_000, parent_concurrency=9)
        with self.assertRaisesRegex(ValueError, "parent_concurrency"):
            _ = budget.available_threads

    def test_child_budget_composes_ancestors(self):
        budget = RuntimeResourceBudget(8, 8, 8, 1_000_000, parent_concurrency=2)
        child = budget.for_children(2)
        self.assertEqual(child.parent_concurrency, 4)
        self.assertEqual(child.available_threads, 2)
        self.assertEqual(child.memory_limit_bytes, budget.memory_limit_bytes)
        with self.assertRaisesRegex(ValueError, "available"):
            budget.for_children(5)
        with self.assertRaises((ValueError, TypeError)):
            budget.for_children(True)


if __name__ == "__main__":
    unittest.main()
