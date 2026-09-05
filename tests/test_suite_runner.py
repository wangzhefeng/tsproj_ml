"""分组入口不得隐藏发现失败、重复 ID 或未分类的新测试。"""

import io
import unittest
from unittest.mock import patch

from run_suite import classify, flatten, main, partition


class SuiteRunnerTest(unittest.TestCase):
    def test_partition_is_disjoint_and_exhaustive(self):
        ids = [
            "test_forecast_tensor.TensorTest.test_shape",
            "test_new_runtime.RuntimeTest.test_new",
            "test_runtime_asset_audit.AuditTest.test_assets",
        ]
        groups = partition(ids)
        self.assertEqual(groups["fast"], [ids[0]])
        self.assertEqual(groups["integration"], [ids[1]])
        self.assertEqual(groups["audit"], [ids[2]])
        self.assertEqual(sorted(sum(groups.values(), [])), sorted(ids))

    def test_mixed_module_audit_is_selected_by_method(self):
        prefix = "test_config_entrypoints.Task27ExecutionMatrixTest."
        self.assertEqual(classify(prefix + "test_task27_checker_validates_all_model_configs_read_only"), "audit")
        self.assertEqual(classify(prefix + "test_task27_full_execution_matrix_reports_exact_counts"), "integration")

    def test_duplicate_ids_raise(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            partition(["test_x.C.test_a", "test_x.C.test_a"])

    def test_flatten_keeps_every_nested_case(self):
        case = unittest.FunctionTestCase(lambda: None)
        self.assertEqual(list(flatten(unittest.TestSuite([unittest.TestSuite([case])]))), [case])

    def test_empty_selection_is_an_error(self):
        with patch("run_suite.discover", return_value=[]), patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(["fast"]), 2)

    def test_discovery_error_is_an_error_even_for_list(self):
        with patch("run_suite.discover", side_effect=RuntimeError("broken import")), patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(["fast", "--list"]), 2)

    def test_failing_test_returns_nonzero(self):
        def failure():
            raise AssertionError("intentional failure")
        case = unittest.FunctionTestCase(failure)
        with patch("run_suite.discover", return_value=[case]), patch("sys.stderr", new_callable=io.StringIO), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["all"]), 1)


if __name__ == "__main__":
    unittest.main()
