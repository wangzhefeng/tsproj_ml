"""仅回测入口合同；真实拟合只使用临时合成数据，不运行业务场景。"""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import test_canonical_runtime_smoke as smoke
from model_pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from model_pipeline.lifecycle import run_lifecycle
import run as entrypoint


class BacktestOnlyTest(unittest.TestCase):
    def test_cli_rejects_ensemble_before_execution(self):
        root = Path(__file__).resolve().parents[1]
        # 读取现役融合配置，仅验证 CLI 分派，不拟合成员。
        path = next((root / "config/aidc_load_15min_daily/route_A/add_ensemble").glob("*.yaml"))
        args = SimpleNamespace(config_yaml=str(path), output_root=None, backtest_only=True)
        with patch.object(entrypoint, "run_ensemble_config_file") as execute:
            with self.assertRaisesRegex(ValueError, "not Ensemble"):
                entrypoint.run(args)
            execute.assert_not_called()

    def test_cli_subprocess_writes_only_backtest_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "target.csv"
            pd.DataFrame({
                "time": pd.date_range("2026-01-01", periods=48, freq="1h"),
                "load": 100 + np.arange(48, dtype=float),
            }).to_csv(source, index=False)
            config = smoke.CanonicalRuntimeSmokeTest().build_config(source, mode="point")
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config.canonical_payload()))
            completed = subprocess.run([
                sys.executable, "run.py", "--config-yaml", str(path),
                "--backtest-only", "--output-root", str(root / "results"),
            ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(len(list(root.rglob("test_scores_df.csv"))), 1)
            self.assertEqual(list(root.rglob("prediction.csv")), [])
            self.assertEqual(list(root.rglob("model.pkl")), [])

    def test_cli_parses_and_dispatches_backtest_only(self):
        with patch("sys.argv", ["run.py", "--config-yaml", "fixture.yaml", "--backtest-only"]):
            args = entrypoint.args_parse()
        self.assertTrue(args.backtest_only)
        with patch("sys.argv", ["run.py", "--config-yaml", "fixture.yaml"]):
            self.assertFalse(entrypoint.args_parse().backtest_only)
        config = smoke.CanonicalRuntimeSmokeTest().build_config(Path("fixture.csv"), mode="point")
        with patch.object(entrypoint, "_load_config", return_value=config), patch.object(
            entrypoint, "run_canonical_config", return_value="backtest-result"
        ) as execute:
            self.assertEqual(entrypoint.run(args), "backtest-result")
        execute.assert_called_once_with(config, output_root=None, backtest_only=True)

    def test_failure_and_cancellation_do_not_touch_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (root, root / "model", root / "test", root / "forecast")
            paths[1].mkdir()
            model_state = paths[1] / "run_state.json"
            model_state.write_text('{"status":"completed","existing":true}')
            runner = SimpleNamespace(config=SimpleNamespace(fingerprint=lambda: "fixture"))
            for error in (RuntimeError("fixture failure"), KeyboardInterrupt("fixture cancellation")):
                with patch("model_pipeline.lifecycle._output_paths", return_value=paths), patch(
                    "model_pipeline.lifecycle.execute_lifecycle", side_effect=error
                ):
                    with self.assertRaises(type(error)) as caught:
                        run_lifecycle(runner, backtest_only=True)
                self.assertIs(caught.exception, error)
                self.assertEqual(json.loads((paths[2] / "backtest_only/run_state.json").read_text())["status"], "failed")
                self.assertEqual(model_state.read_text(), '{"status":"completed","existing":true}')

    def test_real_backtest_stops_before_final_fit_and_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "target.csv"
            pd.DataFrame({
                "time": pd.date_range("2026-01-01", periods=48, freq="1h"),
                "load": 100 + np.arange(48, dtype=float),
            }).to_csv(source, index=False)
            config = smoke.CanonicalRuntimeSmokeTest().build_config(source, mode="point")
            forbidden = AssertionError("backtest-only entered final lifecycle")
            with patch.object(CanonicalBaseModelRunner, "final_bundle_inputs", side_effect=forbidden), patch.object(
                CanonicalBaseModelRunner, "fit_final", side_effect=forbidden
            ), patch("model_pipeline.lifecycle.persist_model_bundle", side_effect=forbidden), patch(
                "model_pipeline.lifecycle.write_forecast_results", side_effect=forbidden
            ):
                result = run_canonical_config(config, output_root=root / "results", backtest_only=True)
            self.assertEqual(type(result).__name__, "BacktestRuntimeResult")
            self.assertEqual(result.test_dir.name, "backtest_only")
            self.assertFalse(pd.read_csv(result.test_dir / "test_scores_df.csv").empty)
            state = json.loads((result.test_dir / "run_state.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(result.fingerprint, config.fingerprint())
            self.assertEqual(list((root / "results").rglob("model.pkl")), [])
            self.assertEqual(list((root / "results").rglob("prediction.csv")), [])
            self.assertEqual(list((root / "results").rglob("pretrained_models")), [])
            self.assertEqual(list((root / "results").rglob("results_forecast")), [])


if __name__ == "__main__":
    unittest.main()
