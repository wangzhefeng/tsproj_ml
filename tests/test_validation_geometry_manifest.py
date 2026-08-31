# -*- coding: utf-8 -*-
"""C2 方案 A：typed geometry 迁移清单与真实训练 origin 数门禁。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml

from config.config_loader import is_model_yaml, load_yaml_config
from forecasting_core.specs import FixedStepBacktestSpec, ForecastConfigSpec
from model_forecasting.design import minimum_history_rows


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/validation_geometry_migration_manifest.yaml"
LEGACY_FIELDS = {
    "history_length",
    "window_length",
    "train_window_length",
    "max_test_windows",
    "test_window_stride",
}


class ValidationGeometryManifestTest(unittest.TestCase):
    def test_manifest_matches_active_configs_and_real_subday_training_counts(self):
        manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        entries = manifest["entries"]
        by_path = {entry["path"]: entry for entry in entries}
        active_paths = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "config").rglob("*.yaml")
            if is_model_yaml(path)
        }
        self.assertEqual(len(entries), 845)
        self.assertEqual(len(by_path), 845)
        self.assertEqual(set(by_path), active_paths)
        self.assertEqual(manifest["counts"]["subday_single_models"], 615)

        timeline_cache: dict[tuple[str, str], pd.DatetimeIndex] = {}
        checked_subday = 0
        for relative in sorted(active_paths):
            path = ROOT / relative
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            validation = raw["validation"]
            self.assertFalse(set(validation) & LEGACY_FIELDS, relative)
            performance = validation.get("performance") or {}
            self.assertFalse(
                set(performance) & {"max_test_windows", "test_window_stride"},
                relative,
            )
            if "ensemble" in raw:
                oof = raw["ensemble"]["oof"]
                self.assertIn("train_window_steps", oof, relative)
                self.assertIn("stride_steps", oof, relative)
                self.assertNotIn("train_window_length", oof, relative)
                self.assertNotIn("stride", oof, relative)

            config = load_yaml_config(path)
            entry = by_path[relative]
            self.assertEqual(config.fingerprint(), entry["new_fingerprint"], relative)
            if not isinstance(config, ForecastConfigSpec):
                continue
            if config.problem.freq in {"1D", "1ME", "1MS"}:
                continue
            checked_subday += 1
            geometry = config.validation.backtest
            self.assertIsInstance(geometry, FixedStepBacktestSpec, relative)
            assert isinstance(geometry, FixedStepBacktestSpec)
            actual = self._actual_final_training_count(config, timeline_cache)
            contract = entry["expected_training_contract"]
            self.assertEqual(
                actual,
                contract["actual_final_training_origin_count"],
                relative,
            )
            self.assertEqual(
                actual,
                geometry.train_window_steps,
                relative,
            )
        self.assertEqual(checked_subday, 615)

    @staticmethod
    def _actual_final_training_count(
        config: ForecastConfigSpec,
        cache: dict[tuple[str, str], pd.DatetimeIndex],
    ) -> int:
        target_sources = [
            source
            for source in config.data.sources
            if any(column.role.value == "target" for column in source.columns)
        ]
        if len(target_sources) != 1:
            raise AssertionError("geometry gate expects one target source")
        source = target_sources[0]
        if source.history_path is None or source.time_col is None:
            raise AssertionError("target source requires history_path and time_col")
        history_path = Path(source.history_path)
        if not history_path.is_absolute():
            history_path = ROOT / history_path
        cache_key = (str(history_path), source.time_col)
        if cache_key not in cache:
            values = pd.to_datetime(
                pd.read_csv(history_path)[source.time_col]
            )
            cache[cache_key] = pd.DatetimeIndex(values).drop_duplicates().sort_values()
        origin = pd.Timestamp(config.validation["forecast_origin"])
        timeline = cache[cache_key]
        timeline = timeline[timeline <= origin]
        positions = timeline.get_indexer([origin])
        if positions[0] < 0:
            raise AssertionError(f"forecast origin is absent: {origin}")
        available = max(
            0,
            (
                int(positions[0])
                - config.problem.horizon
                + 1
                - (minimum_history_rows(config) - 1)
            ),
        )
        geometry = config.validation.backtest
        assert isinstance(geometry, FixedStepBacktestSpec)
        candidate_count = min(available, geometry.history_steps)
        return min(candidate_count, geometry.train_window_steps)


if __name__ == "__main__":
    unittest.main()
