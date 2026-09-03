# -*- coding: utf-8 -*-
"""CLI 只接受配置路径、随机种子与统一输出根目录。"""

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
                        "history_steps": 48,
                        "train_window_steps": 24,
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
            },
        )

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
