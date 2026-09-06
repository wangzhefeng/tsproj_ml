"""本地入口必须显式选择配置；解析阶段不启动训练。"""

import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import run
from config.config_loader import load_yaml_config

class RunExplicitConfigTest(unittest.TestCase):
    def test_missing_config_is_usage_error_before_environment_check(self):
        with patch.object(sys, "argv", ["run.py"]), \
                patch.object(run, "ensure_runtime_environment") as environment, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                run.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--config-yaml", stderr.getvalue())
        environment.assert_not_called()

    def test_explicit_config_reaches_single_model_runtime(self):
        path = "config/aidc_load_15min_short/route_A/baseline/st_recursive.yaml"
        expected = load_yaml_config(path)
        with patch.object(sys, "argv", ["run.py", "--config-yaml", path]), \
                patch.object(run, "ensure_runtime_environment"), \
                patch.object(run, "run_canonical_config") as runtime:
            run.main()
        runtime.assert_called_once()
        self.assertEqual(runtime.call_args.args[0].fingerprint(), expected.fingerprint())


if __name__ == "__main__":
    unittest.main()
