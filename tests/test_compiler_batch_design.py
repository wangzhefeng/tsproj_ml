# -*- coding: utf-8 -*-
"""监督设计批编译接线的等价性测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
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
from model_forecasting.design import _RegistryDesignBuilder


class _CountingSourceRegistry(SourceRegistry):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.materialize_calls = 0

    def materialize(self, *args, **kwargs):
        self.materialize_calls += 1
        return super().materialize(*args, **kwargs)


class CompilerBatchDesignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.times = pd.date_range("2026-01-01", periods=48, freq="1h")
        self.data_path = self.root / "load.csv"
        pd.DataFrame(
            {
                "time": self.times,
                "load": 100.0 + np.arange(len(self.times), dtype=float),
            }
        ).to_csv(self.data_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _config(
        self,
        strategy: str,
        *,
        expanding: bool = False,
        horizon: int = 3,
        output_chunk_length: int | None = None,
        safe_lags: bool = False,
    ) -> ForecastConfigSpec:
        target_lags = (
            tuple(range(horizon, horizon + 3))
            if safe_lags or strategy not in {"recursive", "recmo", "dirrec", "dirrecmo"}
            else (1, 2, 3)
        )
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=horizon,
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
                target_lags={"load": target_lags},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations=(
                    {
                        "advanced": {
                            "expanding": {
                                "columns": ["load"],
                                "stats": ["mean", "std"],
                            }
                        }
                    }
                    if expanding
                    else {}
                ),
            ),
            strategy=ForecastStrategySpec(
                strategy,
                output_chunk_length=output_chunk_length,
            ),
            estimator=EstimatorSpec(
                model_type="ridge",
                target_adapter="independent",
                params={"alpha": 1e-6},
            ),
            probabilistic={"mode": "point"},
            validation={
                "forecast_origin": self.times[-4].isoformat(),
                "history_steps": 20,
                "train_window_steps": 10,
                "fold_count": 1,
                "stride_steps": horizon,
            },
            output={"scenario_subpath": "compiler-batch-design"},
        )

    def _assert_rows_equal(
        self,
        strategy: str,
        *,
        expanding: bool = False,
        horizon: int = 3,
        output_chunk_length: int | None = None,
        safe_lags: bool = False,
        expect_batch: bool | None = None,
    ) -> None:
        config = self._config(
            strategy,
            expanding=expanding,
            horizon=horizon,
            output_chunk_length=output_chunk_length,
            safe_lags=safe_lags,
        )
        origins = tuple(pd.Timestamp(value) for value in self.times[12:18])
        expected_builder = _RegistryDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        expected = tuple(expected_builder.training_row(origin) for origin in origins)
        batch_builder = _RegistryDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        if expect_batch is not None:
            call_steps = tuple(
                coordinates[0].horizon_step
                for coordinates in batch_builder.plan.call_coordinates
            )
            self.assertEqual(
                batch_builder._can_batch_training(call_steps),
                expect_batch,
            )

        actual = batch_builder.training_rows(origins)

        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            self.assertEqual(len(actual_row[0]), len(expected_row[0]))
            for actual_design, expected_design in zip(actual_row[0], expected_row[0]):
                np.testing.assert_array_equal(actual_design, expected_design)
            np.testing.assert_array_equal(actual_row[1], expected_row[1])
        self.assertEqual(batch_builder.feature_schema, expected_builder.feature_schema)
        self.assertEqual(
            batch_builder.categorical_schema,
            expected_builder.categorical_schema,
        )

    def test_direct_training_rows_match_per_origin_compilation(self) -> None:
        self._assert_rows_equal("direct")

    def test_direct_expanding_training_rows_use_batch_and_match(self) -> None:
        self._assert_rows_equal("direct", expanding=True, expect_batch=True)

    def test_recmo_safe_lags_use_batch_and_match(self) -> None:
        self._assert_rows_equal(
            "recmo",
            horizon=4,
            output_chunk_length=2,
            safe_lags=True,
            expect_batch=True,
        )

    def test_recursive_training_rows_preserve_provider_fallback(self) -> None:
        self._assert_rows_equal("recursive", expect_batch=False)

    def test_recursive_safe_lags_use_batch_and_match(self) -> None:
        self._assert_rows_equal(
            "recursive",
            horizon=4,
            safe_lags=True,
            expect_batch=True,
        )

    def test_dirrec_safe_lags_use_batch_and_match(self) -> None:
        self._assert_rows_equal(
            "dirrec",
            horizon=4,
            safe_lags=True,
            expect_batch=True,
        )

    def test_dirrecmo_safe_lags_use_batch_and_match(self) -> None:
        self._assert_rows_equal(
            "dirrecmo",
            horizon=4,
            output_chunk_length=2,
            safe_lags=True,
            expect_batch=True,
        )

    def test_training_row_does_not_retain_forecast_audit(self) -> None:
        config = self._config("recursive")
        builder = _RegistryDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )

        origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[20]))
        builder.training_row(origin)

        self.assertEqual(builder.audit, ())

    def test_forecast_design_retains_visibility_audit(self) -> None:
        config = self._config("recursive")
        builder = _RegistryDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )

        origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[20]))
        builder.forecast_designs(origin)

        self.assertTrue(builder.audit)

    def test_training_row_materializes_information_set_once(self) -> None:
        config = self._config("recursive")
        registry = _CountingSourceRegistry(config.data, self.root)
        builder = _RegistryDesignBuilder(config, registry)
        origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[20]))

        builder.training_row(origin)

        self.assertEqual(registry.materialize_calls, 1)


if __name__ == "__main__":
    unittest.main()
