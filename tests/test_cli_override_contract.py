# -*- coding: utf-8 -*-
"""CLI 的阶段、原点、持久化与非语义运行参数合同。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import run


ROOT = Path(__file__).resolve().parents[1]


class CliContractTest(unittest.TestCase):
    @staticmethod
    def _write_single_model_config(root: Path) -> Path:
        times = pd.date_range("2026-01-01", periods=72, freq="1h")
        values = 100.0 + np.arange(len(times), dtype=float)
        data_path = root / "load.csv"
        pd.DataFrame({"time": times, "load": values}).to_csv(data_path, index=False)
        config_path = root / "model.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "problem": {
                        "time_col": "time",
                        "freq": "1h",
                        "horizon": 2,
                        "targets": ["load"],
                        "training_scope": "local",
                        "series_id_cols": [],
                    },
                    "data": {
                        "sources": [
                            {
                                "name": "target_history",
                                "source_type": "file",
                                "history_path": str(data_path),
                                "time_col": "time",
                                "availability": "source_time",
                                "columns": [
                                    {
                                        "name": "load",
                                        "role": "target",
                                        "categorical": False,
                                    }
                                ],
                            }
                        ]
                    },
                    "features": {
                        "target_lags": {"load": [2, 3]},
                        "observed_past_lags": {},
                        "datetime_features": [],
                        "transformations": {},
                    },
                    "strategy": {"name": "direct"},
                    "estimator": {
                        "model_type": "ridge",
                        "target_adapter": "independent",
                        "params": {"alpha": 1e-8},
                    },
                    "probabilistic": {"mode": "point"},
                    "validation": {
                        "forecast_origin": "2026-01-03T22:00:00",
                        'training_window': {'kind': 'rolling', 'history_steps': 24},
                        "fold_count": 2,
                        "stride_steps": 2,
                    },
                    "output": {
                        "identity": {
                            "scenario_subpath": "cli-contract",
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return config_path

    def test_parser_accepts_only_declared_runtime_arguments(self):
        argv = [
            "run.py",
            "--config-yaml",
            "config/model.yaml",
            "--seed",
            "7",
            "--output-root",
            "/tmp/forecast-results",
        ]
        with patch.object(sys, "argv", argv):
            args = run.args_parse()

        self.assertEqual(
            vars(args),
            {
                "config_yaml": "config/model.yaml",
                "seed": 7,
                "output_root": "/tmp/forecast-results",
                "mode": "backtest-fit-forecast",
                "no_bundle": False,
                "issue_date": None,
            },
        )

    def test_parser_accepts_explicit_backtest_only(self):
        with patch.object(sys, "argv", ["run.py", "--config-yaml", "config/model.yaml", "--mode", "backtest-only"]):
            self.assertEqual(run.args_parse().mode, "backtest-only")

    def test_parser_production_requires_issue_date_and_compatible_modes(self):
        valid = [
            "run.py", "--config-yaml", "config/model.yaml",
            "--mode", "production", "--issue-date", "2026-07-25",
        ]
        with patch.object(sys, "argv", valid):
            args = run.args_parse()
        self.assertEqual(args.mode, "production")
        self.assertEqual(args.issue_date, "2026-07-25")
        for argv in (
            ["run.py", "--config-yaml", "config/model.yaml", "--mode", "production"],
            ["run.py", "--config-yaml", "config/model.yaml", "--mode", "backtest-only", "--no-bundle"],
            ["run.py", "--config-yaml", "config/model.yaml", "--mode", "production", "--issue-date", "2026-07-25", "--backtest-only"],
        ):
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                run.args_parse()

    def test_issue_date_and_no_bundle_mode_matrix(self):
        for mode in ("backtest-only", "backtest-fit-forecast", "production"):
            for date in (None, "2026-07-10", "2026-02-30", "20260710"):
                for no_bundle in (False, True):
                    with self.subTest(mode=mode, date=date, no_bundle=no_bundle):
                        argv = ["run.py", "--config-yaml", "fixture.yaml", "--mode", mode]
                        if date is not None:
                            argv += ["--issue-date", date]
                        if no_bundle:
                            argv += ["--no-bundle"]
                        valid = (date in (None, "2026-07-10")
                                 and not (mode == "production" and date is None)
                                 and not (mode == "backtest-only" and no_bundle))
                        with patch.object(sys, "argv", argv):
                            if valid:
                                args = run.args_parse()
                                self.assertEqual((args.mode, args.issue_date, args.no_bundle), (mode, date, no_bundle))
                            else:
                                with self.assertRaises(SystemExit) as caught:
                                    run.args_parse()
                                self.assertEqual(caught.exception.code, 2)
        for old in ("--production", "--backtest-only"):
            with patch.object(sys, "argv", ["run.py", "--config-yaml", "fixture.yaml", old]):
                with self.assertRaises(SystemExit) as caught:
                    run.args_parse()
                self.assertEqual(caught.exception.code, 2)

    def test_direct_run_rejects_invalid_options_before_config_io(self):
        for options in ({"mode": "production"}, {"mode": "bad"},
                        {"mode": "backtest-only", "no_bundle": True}, {"issue_date": "2026-02-30"}):
            with self.subTest(options=options), patch.object(run, "_load_config") as load:
                with self.assertRaises(ValueError):
                    run.run(SimpleNamespace(config_yaml="absent.yaml", **options))
                load.assert_not_called()

    def test_production_cli_really_skips_backtest_and_supports_no_bundle(self):
        import subprocess
        for no_bundle in (False, True):
            with self.subTest(no_bundle=no_bundle), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = self._write_single_model_config(root)
                payload = yaml.safe_load(path.read_text())
                payload["validation"]["fold_count"] = 100
                path.write_text(yaml.safe_dump(payload))
                argv = [sys.executable, "run.py", "--config-yaml", str(path), "--mode", "production",
                        "--issue-date", "2026-01-01", "--output-root", str(root / "results")]
                if no_bundle:
                    argv.append("--no-bundle")
                completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                predictions = list((root / "results").rglob("prediction.csv"))
                self.assertEqual(len(predictions), 1)
                self.assertEqual(len(pd.read_csv(predictions[0])), 2)
                self.assertEqual(len(list((root / "results").rglob("model.pkl"))), 0 if no_bundle else 1)
                self.assertEqual(list((root / "results").rglob("cv_plot_df.csv")), [])

    def test_output_root_is_used_by_single_model_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = self._write_single_model_config(root)
            output_root = root / "artifacts"

            result = run.run(
                SimpleNamespace(
                    config_yaml=str(config_path),
                    seed=7,
                    output_root=str(output_root),
                )
            )

            self.assertTrue(result.run_dir.is_relative_to(output_root))
            relative = result.run_dir.relative_to(output_root / "results_forecast")
            self.assertEqual(result.run_dir, result.forecast_dir)
            self.assertEqual(result.model_dir, output_root / "pretrained_models" / relative)
            self.assertEqual(result.test_dir, output_root / "results_test" / relative)
            self.assertTrue((result.model_dir / "model.pkl").exists())
            self.assertTrue((result.test_dir / "cv_plot_df.csv").exists())
            self.assertTrue((result.forecast_dir / "prediction.csv").exists())

    def test_parser_rejects_removed_model_override(self):
        argv = [
            "run.py",
            "--config-yaml",
            "config/model.yaml",
            "--model-type",
            "ridge",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                run.args_parse()


if __name__ == "__main__":
    unittest.main()
