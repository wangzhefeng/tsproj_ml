# -*- coding: utf-8 -*-
"""监督设计批编译接线的等价性测试。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering.compiler import FeatureCompiler
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
from model_pipeline.supervised_design import SupervisedDesignBuilder, _split_batch_designs


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
        self.global_data_path = self.root / "panel.csv"
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "series_id": series_id,
                        "time": self.times,
                        "load": offset + np.arange(len(self.times), dtype=float),
                        "power": offset * 2.0
                        + np.arange(len(self.times), dtype=float),
                    }
                )
                for series_id, offset in (("A", 100.0), ("B", 1000.0))
            ],
            ignore_index=True,
        ).to_csv(self.global_data_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_split_batch_designs_preserves_global_identity_major_order(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": [10, 10, 20, 20],
                "horizon_step": [1, 3, 1, 3],
                "feature": [101.0, 103.0, 201.0, 203.0],
            }
        )

        designs = _split_batch_designs(
            frame,
            schema=("series_id", "feature"),
            call_steps=(1, 3),
            n_series=2,
        )

        self.assertEqual(len(designs), 2)
        np.testing.assert_array_equal(
            designs[0],
            np.asarray([[10.0, 101.0], [20.0, 201.0]]),
        )
        np.testing.assert_array_equal(
            designs[1],
            np.asarray([[10.0, 103.0], [20.0, 203.0]]),
        )

    def test_label_extraction_does_not_use_dataframe_row_access(self) -> None:
        config = self._config("mimo")
        builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[20]))
        request = builder.request(origin, target_access="supervised_labels")
        information_set = builder.registry.materialize(request)

        with patch.object(
            pd.DataFrame,
            "iloc",
            new_callable=PropertyMock,
            side_effect=AssertionError("label extraction used DataFrame.iloc"),
        ):
            values, trajectories = builder._labels_from_information_set(
                request,
                information_set,
            )

        np.testing.assert_array_equal(
            values[0, :, 0],
            np.asarray([121.0, 122.0, 123.0]),
        )
        self.assertEqual(
            trajectories["__local__"]["load"],
            (121.0, 122.0, 123.0),
        )

    def test_batch_training_validates_without_materializing_proofs(self) -> None:
        config = self._config("direct")
        builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value)) for value in self.times[12:18]
        )

        with patch(
            "feature_engineering.compiler.VisibilityProof",
            side_effect=AssertionError("batch training materialized VisibilityProof"),
        ):
            rows = builder.training_rows(origins)

        self.assertEqual(len(rows), len(origins))

    def test_batch_training_validate_only_rejects_late_availability(self) -> None:
        config = self._config("direct")
        builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value)) for value in self.times[12:18]
        )
        add_proof_column = FeatureCompiler._add_batch_proof_column

        def inject_late_availability(*args, **kwargs) -> None:
            add_proof_column(*args, **kwargs)
            item = args[0]
            feature_name = args[1]
            source_name, role, source_times, available_at = item["proof_columns"][
                feature_name
            ]
            item["proof_columns"][feature_name] = (
                source_name,
                role,
                source_times,
                tuple(
                    pd.Timestamp(item["request"].forecast_origin) + pd.Timedelta(days=1)
                    for _ in available_at
                ),
            )

        with patch.object(
            FeatureCompiler,
            "_add_batch_proof_column",
            side_effect=inject_late_availability,
        ), self.assertRaisesRegex(ValueError, "available after forecast_origin"):
            builder.training_rows(origins)

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
                "forecast_origin": cast(
                    pd.Timestamp,
                    pd.Timestamp(self.times.to_numpy()[-4]),
                ).isoformat(),
                "history_steps": 20,
                "train_window_steps": 10,
                "fold_count": 1,
                "stride_steps": horizon,
            },
            output={"scenario_subpath": "compiler-batch-design"},
        )

    def _global_config(
        self,
        strategy: str,
        output_chunk_length: int | None,
    ) -> ForecastConfigSpec:
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=4,
                targets=("load", "power"),
                training_scope="global",
                series_id_cols=("series_id",),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="target_history",
                        source_type="file",
                        columns=(
                            ColumnSpec("series_id", "key", categorical=True),
                            ColumnSpec("load", "target"),
                            ColumnSpec("power", "target"),
                        ),
                        history_path=str(self.global_data_path),
                        time_col="time",
                        series_id_cols=("series_id",),
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": (4, 5, 6), "power": (4, 5, 6)},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations={},
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
                "forecast_origin": cast(
                    pd.Timestamp,
                    pd.Timestamp(self.times.to_numpy()[-5]),
                ).isoformat(),
                "history_steps": 20,
                "train_window_steps": 10,
                "fold_count": 1,
                "stride_steps": 4,
                "training_scope": {
                    "series_order": ["A", "B"],
                    "incomplete_series_policy": "raise",
                    "unknown_series_policy": "raise",
                },
            },
            output={"scenario_subpath": "compiler-batch-design-global"},
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
        origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value)) for value in self.times[12:18]
        )
        expected_builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        expected = tuple(expected_builder.training_row(origin) for origin in origins)
        batch_builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        if expect_batch is not None:
            call_steps = tuple(
                coordinates[0].horizon_step
                for coordinates in batch_builder.plan.call_coordinates
            )
            requests = tuple(
                batch_builder.request(origin) for origin in origins
            )
            self.assertEqual(
                batch_builder.compiler.batch_eligibility(
                    requests,
                    horizon_steps=call_steps,
                ).eligible,
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

    def test_mimo_training_rows_match_per_origin_compilation(self) -> None:
        self._assert_rows_equal("mimo", expect_batch=True)

    def test_dirmo_training_rows_match_per_origin_compilation(self) -> None:
        self._assert_rows_equal(
            "dirmo",
            horizon=4,
            output_chunk_length=2,
            expect_batch=True,
        )

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

    def test_recursive_direct_alignment_hint_does_not_bypass_provider_fallback(
        self,
    ) -> None:
        config = self._config("recursive")
        config = replace(
            config,
            features=replace(
                config.features,
                transformations={
                    "direct": {
                        "layout": "independent_models",
                        "align_to_target": False,
                    }
                },
            ),
        )
        builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )
        origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value)) for value in self.times[12:18]
        )
        requests = tuple(builder.request(origin) for origin in origins)
        call_steps = tuple(
            coordinates[0].horizon_step
            for coordinates in builder.plan.call_coordinates
        )

        eligibility = builder.compiler.batch_eligibility(
            requests,
            horizon_steps=call_steps,
        )

        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.reason_code, "provider_dependent_lag")
        self.assertEqual(eligibility.origin_count, len(origins))
        self.assertEqual(eligibility.call_count, len(call_steps))
        self.assertEqual(
            eligibility.estimated_origin_call_count,
            len(origins) * len(call_steps),
        )
        self.assertIn("target:load:lag=1", eligibility.trigger_fields)
        with patch.object(
            builder.compiler,
            "compile_batch",
            wraps=builder.compiler.compile_batch,
        ) as compile_batch:
            actual = builder.training_rows(origins)

        self.assertEqual(len(actual), len(origins))
        compile_batch.assert_not_called()

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

    def test_global_k2_all_strategies_batch_match_row_without_series_mixing(self) -> None:
        cases = (
            ("recursive", None),
            ("direct", None),
            ("mimo", None),
            ("recmo", 2),
            ("dirrec", None),
            ("dirmo", 2),
            ("dirrecmo", 2),
        )
        origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value)) for value in self.times[12:16]
        )
        for strategy, chunk in cases:
            with self.subTest(strategy=strategy):
                config = self._global_config(strategy, chunk)
                row_builder = SupervisedDesignBuilder(
                    config,
                    SourceRegistry(config.data, self.root),
                )
                expected = tuple(
                    row_builder.training_row(origin) for origin in origins
                )
                batch_builder = SupervisedDesignBuilder(
                    config,
                    SourceRegistry(config.data, self.root),
                )
                actual = batch_builder.training_rows(origins)

                for actual_row, expected_row in zip(actual, expected):
                    for actual_design, expected_design in zip(
                        actual_row[0], expected_row[0]
                    ):
                        np.testing.assert_array_equal(actual_design, expected_design)
                    np.testing.assert_array_equal(actual_row[1], expected_row[1])
                self.assertEqual(batch_builder.series_ids, ("A", "B"))
                self.assertEqual(batch_builder.feature_schema, row_builder.feature_schema)

    def test_training_row_does_not_retain_forecast_audit(self) -> None:
        config = self._config("recursive")
        builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )

        origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[20]))
        builder.training_row(origin)

        self.assertEqual(builder.audit, ())

    def test_forecast_design_retains_visibility_audit(self) -> None:
        config = self._config("recursive")
        builder = SupervisedDesignBuilder(
            config,
            SourceRegistry(config.data, self.root),
        )

        origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[20]))
        builder.forecast_designs(origin)

        self.assertTrue(builder.audit)

    def test_training_row_materializes_information_set_once(self) -> None:
        config = self._config("recursive")
        registry = _CountingSourceRegistry(config.data, self.root)
        builder = SupervisedDesignBuilder(config, registry)
        origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[20]))

        builder.training_row(origin)

        self.assertEqual(registry.materialize_calls, 1)


if __name__ == "__main__":
    unittest.main()
