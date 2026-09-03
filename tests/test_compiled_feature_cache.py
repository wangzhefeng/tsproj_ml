# -*- coding: utf-8 -*-
"""监督特征编译产物缓存的命中与失效测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering.cache import COMPILED_CACHE_DIR_NAME
from forecasting_core.specs import (
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
    EstimatorSpec,
    FeatureSpec,
    ForecastConfigSpec,
    ForecastProblemSpec,
    ForecastStrategySpec,
)
from model_forecasting.runtime import CanonicalBaseModelRunner


class CompiledFeatureCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "load.csv"
        self.times = pd.date_range("2026-01-01", periods=64, freq="1h")
        self._write_values(np.arange(len(self.times), dtype=float))
        self.origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[-4]))
        self.config = self._config()
        self.cache_root = self.root / "results"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_values(self, values: np.ndarray) -> None:
        pd.DataFrame(
            {
                "time": self.times,
                "load": 100.0 + values,
            }
        ).to_csv(self.data_path, index=False)

    def _config(self) -> ForecastConfigSpec:
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=2,
                targets=("load",),
                information_mode="forecast",
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="target_history",
                        source_type="file",
                        columns=(ColumnSpec("load", "target"),),
                        history_path=str(self.data_path),
                        time_col="time",
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": (2, 3, 4)},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations={},
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(
                model_type="ridge",
                target_adapter="independent",
                params={"alpha": 1e-6},
            ),
            probabilistic={"mode": "point"},
            validation=validation,
            output={"scenario_subpath": "compiled-cache"},
        )

    def _runner(self) -> CanonicalBaseModelRunner:
        return CanonicalBaseModelRunner(
            self.config,
            SourceRegistry(self.config.data, self.root),
            self.origin,
            compiled_cache_root=self.cache_root,
        )

    def test_cache_hit_is_fast_and_source_change_invalidates(self) -> None:
        first = self._runner()
        self.assertFalse(first.compiled_cache_hit)
        cache_parent = self.cache_root / COMPILED_CACHE_DIR_NAME
        first_entries = tuple(cache_parent.iterdir())
        self.assertEqual(len(first_entries), 1)

        start = perf_counter()
        second = self._runner()
        elapsed = perf_counter() - start
        self.assertTrue(second.compiled_cache_hit)
        self.assertLess(elapsed, 1.0)
        for actual, expected in zip(second.X_all, first.X_all):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(second.Y_all, first.Y_all)
        self.assertEqual(second.feature_schema, first.feature_schema)

        changed = np.arange(len(self.times), dtype=float)
        changed[0] = -999.0
        self._write_values(changed)
        third = self._runner()
        self.assertFalse(third.compiled_cache_hit)
        self.assertEqual(len(tuple(cache_parent.iterdir())), 2)

    def test_performance_controls_share_compiled_design_cache(self) -> None:
        first = self._runner()
        self.assertFalse(first.compiled_cache_hit)
        configured = self._config(
            {
                "window_parallel_workers": 2,
                "model_thread_count": 4,
            }
        )

        second = CanonicalBaseModelRunner(
            configured,
            SourceRegistry(configured.data, self.root),
            self.origin,
            compiled_cache_root=self.cache_root,
        )

        self.assertTrue(second.compiled_cache_hit)
        cache_parent = self.cache_root / COMPILED_CACHE_DIR_NAME
        self.assertEqual(len(tuple(cache_parent.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
