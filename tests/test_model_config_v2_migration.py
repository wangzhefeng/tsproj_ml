# -*- coding: utf-8 -*-
"""Forecast model-schema migration audit tests."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_forecast_configs import REQUIRED_KEYS, build_catalog, main


class ForecastConfigAuditTest(unittest.TestCase):
    def _write_fixture_configs(self, root: Path) -> None:
        (root / "nested").mkdir()
        (root / "nested" / "z_dirrec.yaml").write_text(
            """\
base_config: config.univariate_config
overrides:
  runtime:
    predict_steps: 4
  target_series:
    target: load_kw
    target_series_numeric_features: [temperature]
  time_lag_features:
    lags: [2, 6]
  model_strategy:
    pred_method: univariate-single-multistep-direct-recursive
    block_size: 0
  output:
    scenario_subpath: fixtures/dirrec
    setting_suffix: -audit
""",
            encoding="utf-8",
        )
        (root / "a_direct.yaml").write_text(
            """\
base_config: config.multivariate_config
overrides:
  runtime:
    predict_steps: 3
  target_series:
    target: target_value
    target_series_numeric_features: [feature_a, feature_b]
  time_lag_features:
    lags: [3, 9]
  model_strategy:
    pred_method: msmd
    direct_strategy: horizon_feature
    blend_weight_strategy: ridge_stacking
    blend_weights: [0.25, 0.75]
    endogenous_backfill_strategy: auxiliary
    enable_global_training: true
    series_id_feature: meter_id
""",
            encoding="utf-8",
        )
        (root / "periodicity_detect.yaml").write_text(
            "source_path: dataset/input.csv\nseasonal_period: 96\n",
            encoding="utf-8",
        )

    def test_catalog_is_deterministic_sorted_unique_and_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_fixture_configs(root)

            first = build_catalog(root)
            second = build_catalog(root)

            self.assertEqual(first, second)
            paths = [row["path"] for row in first]
            self.assertEqual(paths, sorted(paths))
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(paths, ["a_direct.yaml", "nested/z_dirrec.yaml"])
            for row in first:
                self.assertTrue(REQUIRED_KEYS.issubset(row))

            direct, dirrec = first
            self.assertEqual(direct["legacy_method"], "msmd")
            self.assertEqual(direct["method_code"], "msmd")
            self.assertEqual(direct["base_config"], "config.multivariate_config")
            self.assertEqual(direct["target_series_numeric_features"], ["feature_a", "feature_b"])
            self.assertEqual(direct["direct_strategy"], "horizon_feature")
            self.assertIsNone(direct["effective_block_size"])
            self.assertTrue(direct["enable_global_training"])
            self.assertEqual(direct["series_id_feature"], "meter_id")

            self.assertEqual(
                dirrec["legacy_method"],
                "univariate-single-multistep-direct-recursive",
            )
            self.assertEqual(dirrec["method_code"], "usmdr")
            self.assertEqual(dirrec["block_size"], 0)
            self.assertEqual(dirrec["effective_block_size"], 2)
            self.assertEqual(dirrec["horizon"], 4)
            self.assertEqual(dirrec["lags"], [2, 6])
            self.assertEqual(dirrec["scenario_subpath"], "fixtures/dirrec")
            self.assertEqual(dirrec["setting_suffix"], "-audit")

    def test_cli_output_matches_build_catalog_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "config"
            root.mkdir()
            self._write_fixture_configs(root)
            output_path = Path(temp_dir) / "catalog.json"
            expected = build_catalog(root)

            exit_code = main(["--root", str(root), "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), expected)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )

    def test_cli_does_not_create_output_when_catalog_construction_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "config"
            root.mkdir()
            (root / "broken.yaml").write_text(
                """\
base_config: config.univariate_config
overrides:
  runtime:
    unknown_runtime_field: true
""",
                encoding="utf-8",
            )
            output_path = Path(temp_dir) / "should-not-exist.json"

            with self.assertRaises(AttributeError):
                main(["--root", str(root), "--output", str(output_path)])

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
