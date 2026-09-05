# -*- coding: utf-8 -*-
"""活动配置的真实训练 origin 数门禁；历史迁移清单不再锁定当前身份。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml

from config.config_loader import is_model_yaml, load_yaml_config
from forecasting_core.specs import FixedStepBacktestSpec, ForecastConfigSpec
from model_forecasting.design import minimum_history_rows


ROOT = Path(__file__).resolve().parents[1]

LEGACY_FIELDS = {
    "history_length",
    "window_length",
    "train_window_length",
    "max_test_windows",
    "test_window_stride",
}


class ValidationGeometryManifestTest(unittest.TestCase):
    def test_active_configs_have_real_subday_training_counts(self):
        active_paths = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "config").rglob("*.yaml")
            if is_model_yaml(path)
        }
        self.assertTrue(active_paths, "no active model configs discovered")

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

            if not isinstance(config, ForecastConfigSpec):
                continue
            if config.problem.freq in {"1D", "1ME", "1MS"}:
                continue
            checked_subday += 1
            geometry = config.validation.backtest
            self.assertIsInstance(geometry, FixedStepBacktestSpec, relative)
            assert isinstance(geometry, FixedStepBacktestSpec)
            actual = self._actual_final_training_count(config, timeline_cache)

            self.assertEqual(
                actual,
                geometry.train_window_steps,
                relative,
            )
        self.assertGreater(checked_subday, 0)

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
