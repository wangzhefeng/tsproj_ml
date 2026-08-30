# -*- coding: utf-8 -*-
"""Reference-based ensemble runtime lifecycle, artifact, and leakage contracts (v4)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from model_ensemble.loader import load_ensemble_config
from model_ensemble.runtime import run_ensemble_config


def _member_doc(strategy: str, estimator: str, subpath: str) -> dict:
    return {
        "schema_version": 2,
        "problem": {
            "time_col": "time",
            "freq": "1h",
            "horizon": 2,
            "targets": ["load"],
            "information_mode": "forecast",
            "training_scope": "local",
            "series_id_cols": [],
        },
        "data": {
            "sources": [
                {
                    "name": "targets",
                    "source_type": "file",
                    "columns": [{"name": "load", "role": "target", "categorical": False}],
                    "history_path": "data.csv",
                    "time_col": "time",
                    "series_id_cols": [],
                    "availability": "source_time",
                }
            ]
        },
        "features": {
            "target_lags": {"load": [2, 3]},
            "observed_past_lags": {},
            "datetime_features": [],
            "transformations": {},
        },
        "strategy": {"name": strategy},
        "estimator": {
            "model_type": estimator,
            "target_adapter": "independent",
            "params": {"alpha": 1e-8} if estimator == "ridge" else {},
        },
        "probabilistic": {"mode": "point"},
        "validation": {"forecast_origin": "2026-01-03T23:00:00"},
        "output": {"scenario_subpath": subpath},
    }


def _ensemble_doc(method: str) -> dict:
    return {
        "schema_version": 2,
        "problem": _member_doc("direct", "ridge", "x")["problem"],
        "data": _member_doc("direct", "ridge", "x")["data"],
        "probabilistic": {"mode": "point"},
        "ensemble": {
            "members": [
                {"name": "direct", "config_ref": "member_direct.yaml"},
                {"name": "recursive", "config_ref": "member_recursive.yaml"},
            ],
            "oof": {"train_window_length": 6, "fold_count": 2, "stride": 1},
            "method": {"name": method},
        },
        "validation": {"forecast_origin": "2026-01-03T23:00:00"},
        "output": {"scenario_subpath": f"ens-{method}"},
    }


class ReferenceEnsembleRuntimeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        times = pd.date_range("2026-01-01", periods=72, freq="1h")
        pd.DataFrame(
            {"time": times, "load": 20.0 + 0.5 * np.arange(len(times))}
        ).to_csv(self.root / "data.csv", index=False)
        (self.root / "member_direct.yaml").write_text(
            yaml.safe_dump(_member_doc("direct", "ridge", "direct")), encoding="utf-8"
        )
        (self.root / "member_recursive.yaml").write_text(
            yaml.safe_dump(_member_doc("recursive", "ridge", "recursive")),
            encoding="utf-8",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, method: str):
        path = self.root / f"ens_{method}.yaml"
        path.write_text(yaml.safe_dump(_ensemble_doc(method)), encoding="utf-8")
        config = load_ensemble_config(path)
        return run_ensemble_config(
            config, output_root=self.root, base_dir=self.root
        )

    def test_averaging_persists_oof_cache_and_matches_manual_mean(self):
        result = self._run("averaging")
        members = result["member_final_values"]
        manual = (members["direct"] + members["recursive"]) / 2.0
        np.testing.assert_allclose(result["combined_values"], manual)
        cache_dir = self.root / "_ensemble_oof" / result["oof_fingerprint"]
        self.assertTrue(cache_dir.exists())
        metadata = json.loads((cache_dir / "oof_metadata.json").read_text("utf-8"))
        self.assertEqual(metadata["member_order"], ["direct", "recursive"])

    def test_stacking_weights_use_oof_not_outer_holdout(self):
        result = self._run("linear_blending")
        audit = result["audit"]
        # folds used for fitting come from inside the outer holdout window:
        # the newest fold origin is strictly before the forecast origin
        folds = result["oof"].folds
        self.assertTrue(folds)
        newest = max(pd.Timestamp(f["origin"]) for f in folds)
        self.assertLess(newest, pd.Timestamp("2026-01-03T23:00:00"))
        self.assertEqual(audit["method"], "linear_blending")

    def test_bundleless_result_contract_is_complete(self):
        result = self._run("averaging")
        for key in ("artifact", "oof", "oof_fingerprint", "member_final_values",
                    "combined_values", "forecast_times", "audit"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
