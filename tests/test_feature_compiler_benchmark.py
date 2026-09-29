"""性能基准的正确性门禁，不对墙钟速度设易抖动阈值。"""
import unittest

from benchmark_feature_compiler import run_benchmark


class FeatureCompilerBenchmarkTest(unittest.TestCase):
    def test_benchmark_verifies_values_and_evidence_before_reporting(self):
        for global_scope in (False, True):
            with self.subTest(global_scope=global_scope):
                result = run_benchmark(global_scope=global_scope, horizon=3, origins=2, repeats=1)
                self.assertTrue(result["equivalent"])
                self.assertEqual(result["origin_count"], 2)
                self.assertEqual(result["horizon"], 3)
                self.assertEqual(len(result["single_seconds"]), 1)
                self.assertEqual(len(result["batch_seconds"]), 1)
                self.assertGreater(result["single_median_seconds"], 0)
                self.assertGreater(result["batch_median_seconds"], 0)

    def test_benchmark_rejects_empty_workloads(self):
        with self.assertRaises(ValueError):
            run_benchmark(global_scope=False, horizon=3, origins=0, repeats=1)


if __name__ == "__main__":
    unittest.main()
