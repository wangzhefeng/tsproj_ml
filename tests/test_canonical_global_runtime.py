# -*- coding: utf-8 -*-
"""Canonical Global Panel N×H×K runtime contracts."""

import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from model_pipeline.runner import run_canonical_config
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


STRATEGIES = (
    ("recursive", None),
    ("direct", None),
    ("mimo", None),
    ("recmo", 2),
    ("dirrec", None),
    ("dirmo", 2),
    ("dirrecmo", 2),
)


class CanonicalGlobalRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.times = pd.date_range("2026-01-01", periods=96, freq="1h")
        rows = []
        for series_id, offset, slope in (("B", 1000.0, 7.0), ("A", 10.0, 2.0)):
            for step, timestamp in enumerate(self.times):
                rows.append(
                    {
                        "series_id": series_id,
                        "time": timestamp,
                        "load": offset + slope * step,
                        "power": -2.0 * offset + (slope + 3.0) * step,
                    }
                )
        self.targets = pd.DataFrame(rows)
        self.target_path = self.root / "targets.csv"
        self.targets.to_csv(self.target_path, index=False)
        self.static_path = self.root / "static.csv"
        pd.DataFrame(
            {
                "series_id": ["A", "B"],
                "region": ["east", "west"],
                "capacity": [100.0, 5000.0],
            }
        ).to_csv(self.static_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_config(
        self,
        strategy,
        chunk,
        *,
        mode="point",
        series_order=("A", "B"),
        incomplete_policy="raise",
    ):
        horizon = 4
        recursive = strategy in {"recursive", "recmo", "dirrec", "dirrecmo"}
        lags = (1, 2, 3) if recursive else (horizon, horizon + 1, horizon + 2)
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=horizon,
                targets=("load", "power"),
                training_scope="global",
                series_id_cols=("series_id",),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="targets",
                        source_type="file",
                        columns=(
                            ColumnSpec("series_id", "key", categorical=True),
                            ColumnSpec("load", "target"),
                            ColumnSpec("power", "target"),
                        ),
                        history_path=str(self.target_path),
                        time_col="time",
                        series_id_cols=("series_id",),
                        availability="source_time",
                    ),
                    DataSourceSpec(
                        name="static",
                        source_type="file",
                        columns=(
                            ColumnSpec("series_id", "key", categorical=True),
                            ColumnSpec("region", "static", categorical=True),
                            ColumnSpec("capacity", "static"),
                        ),
                        history_path=str(self.static_path),
                        series_id_cols=("series_id",),
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": lags, "power": lags},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations={
                    "feature_scaling": {
                        "method": "standard",
                        "grouped": True,
                        "encode_categorical": True,
                    },
                    "target": {
                        "calendar_normalization": {"method": "none"},
                        "decomposition": {"method": "none"},
                        "scaling": {"method": "standard", "inverse": True},
                    },
                },
            ),
            strategy=ForecastStrategySpec(strategy, output_chunk_length=chunk),
            estimator=EstimatorSpec(
                model_type="ridge" if mode == "point" else "qr",
                target_adapter="independent",
                params={"alpha": 1e-8} if mode == "point" else {},
            ),
            probabilistic=(
                {"mode": "point"}
                if mode == "point"
                else {
                    "mode": "quantile",
                    "quantiles": [0.1, 0.5, 0.9],
                    "point_quantile": 0.5,
                }
            ),
            validation={
                "forecast_origin": "2026-01-04T19:00:00",
                "history_steps": 10_000,
                "train_window_steps": 9_999,
                "fold_count": 1,
                "stride_steps": 4,
                "training_scope": {
                    **(
                        {"series_order": list(series_order)}
                        if series_order is not None
                        else {}
                    ),
                    "incomplete_series_policy": incomplete_policy,
                    "unknown_series_policy": "raise",
                },
            },
            output={"scenario_subpath": "global-runtime"},
        )

    def test_all_seven_point_strategies_run_n2_h2_k2_without_crossing(self):
        expected_times = pd.date_range("2026-01-04T20:00:00", periods=4, freq="1h")
        expected = self.targets.loc[
            self.targets["time"].isin(expected_times)
        ].sort_values(["series_id", "time"], kind="stable")
        for strategy, chunk in STRATEGIES:
            with self.subTest(strategy=strategy):
                output_root = self.root / strategy
                result = run_canonical_config(
                    self.build_config(strategy, chunk),
                    output_root=output_root,
                )
                prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
                self.assertEqual(len(prediction), 2 * 4 * 2)
                self.assertEqual(tuple(prediction["series_id"].drop_duplicates()), ("A", "B"))
                pivot = prediction.pivot(
                    index=["series_id", "time"],
                    columns="target",
                    values="predict_value",
                ).reset_index()
                pivot["time"] = pd.to_datetime(pivot["time"])
                merged = pivot.merge(expected, on=["series_id", "time"], how="inner")
                self.assertEqual(len(merged), 8)
                np.testing.assert_allclose(merged["load_x"], merged["load_y"], atol=0.25)
                np.testing.assert_allclose(merged["power_x"], merged["power_y"], atol=0.25)

                scores = pd.read_csv(result.test_dir / "test_scores_df.csv")
                self.assertEqual(scores["scope"].tolist()[:3], ["target", "target", "aggregate"])
                self.assertEqual(scores["n_points"].tolist()[:3], [8, 8, 16])
                # per-horizon 明细拆分到独立文件（2026-09-03 方案 B）
                self.assertEqual(set(scores["scope"]), {"target", "aggregate"})
                horizon_scores = pd.read_csv(
                    result.test_dir / "test_scores_horizon_df.csv"
                )
                self.assertEqual(
                    set(horizon_scores["scope"]),
                    {"horizon", "aggregate_horizon"},
                )
                # K=2、H=4 → 每 target 4 行 + 池化 4 行
                self.assertEqual((horizon_scores["scope"] == "horizon").sum(), 2 * 4)
                self.assertEqual(
                    (horizon_scores["scope"] == "aggregate_horizon").sum(), 4
                )
                self.assertTrue(
                    (
                        horizon_scores.loc[
                            horizon_scores["scope"] == "horizon", "n_points"
                        ]
                        == 2
                    ).all()
                )
                self.assertTrue(
                    (
                        horizon_scores.loc[
                            horizon_scores["scope"] == "aggregate_horizon",
                            "n_points",
                        ]
                        == 4  # 池化跨 target：N(2) × K(2)
                    ).all()
                )
                self.assertEqual(result.bundle.dimensions, (2, 4, 2))
                self.assertEqual(result.bundle.input_schema["panel"]["known_series_ids"], ["A", "B"])
                self.assertIn("series_id", result.bundle.selected_features)
                self.assertIn("region", result.bundle.selected_features)
                self.assertIn("capacity", result.bundle.selected_features)
                self.assertEqual(result.bundle.model.N, 2)
                self.assertEqual(
                    result.bundle.feature_scaler.category_mappings["series_id"],
                    {"A": 0, "B": 1},
                )
                self.assertEqual(
                    result.bundle.feature_scaler.category_mappings["region"],
                    {"east": 0, "west": 1},
                )
                self.assertEqual(
                    set(result.bundle.feature_scaler.scalers),
                    {"lag", "datetime", "other"},
                )

                with (result.model_dir / "model.pkl").open("rb") as file:
                    loaded = pickle.load(file)
                self.assertEqual(loaded.input_schema["panel"]["known_series_ids"], ["A", "B"])
                resolved = json.loads(
                    (result.forecast_dir / "resolved_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(resolved["runtime"]["series_order"], ["A", "B"])

    def test_independent_quantile_runtime_preserves_n_h_k_q(self):
        base = self.build_config("direct", None, mode="quantile")
        result = run_canonical_config(
            replace(
                base,
                validation={
                    "forecast_origin": base.validation["forecast_origin"],
                    "training_scope": {
                        **dict(base.validation["training_scope"]),
                        "series_order": list(
                            base.validation["training_scope"]["series_order"]
                        ),
                    },
                    "history_steps": 48,
                    "train_window_steps": 24,
                    "fold_count": 2,
                    "stride_steps": 4,
                },
                probabilistic={
                    "mode": "quantile",
                    "quantiles": [0.1, 0.5, 0.9],
                    "point_quantile": 0.5,
                },
            ),
            output_root=self.root / "quantile",
        )
        prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
        cv = pd.read_csv(result.test_dir / "cv_plot_df.csv")
        scores = pd.read_csv(result.test_dir / "test_scores_df.csv")
        self.assertEqual(len(prediction), 16)
        self.assertTrue({"predict_q10", "predict_q50", "predict_q90"}.issubset(prediction))
        self.assertEqual(result.bundle.dimensions, (2, 4, 2))
        self.assertEqual(sorted(cv["window"].unique().tolist()), [1, 2])
        self.assertEqual(len(cv), 2 * 2 * 4 * 2)
        self.assertTrue({"predict_q10", "predict_q50", "predict_q90"}.issubset(cv))
        # K=2、H=4、2 窗：汇总每窗 2 target + 1 aggregate = 3；horizon 明细每窗 12
        self.assertEqual(len(scores), 2 * 3)
        horizon_scores = pd.read_csv(result.test_dir / "test_scores_horizon_df.csv")
        self.assertEqual(len(horizon_scores), 2 * 12)

    def test_series_order_defaults_to_target_source_first_occurrence(self):
        result = run_canonical_config(
            self.build_config("direct", None, series_order=None),
            output_root=self.root / "inferred-order",
        )
        self.assertEqual(result.bundle.series_ids, ("B", "A"))
        self.assertEqual(
            result.bundle.input_schema["panel"]["known_series_ids"],
            ["B", "A"],
        )

    def test_explicit_series_order_rejects_unknown_target_series(self):
        extra = self.targets.copy()
        extra.loc[len(extra)] = ["C", self.times[0], 1.0, 2.0]
        extra.to_csv(self.target_path, index=False)
        with self.assertRaisesRegex(ValueError, "unknown series"):
            run_canonical_config(
                self.build_config("direct", None),
                output_root=self.root / "unknown",
            )

    def test_incomplete_series_policy_raise_and_drop(self):
        incomplete = self.targets.loc[
            ~((self.targets["series_id"] == "B") & (self.targets["time"] == self.times[30]))
        ]
        incomplete.to_csv(self.target_path, index=False)
        with self.assertRaisesRegex(ValueError, "series 'B' is incomplete"):
            run_canonical_config(
                self.build_config("direct", None),
                output_root=self.root / "incomplete-raise",
            )

        result = run_canonical_config(
            self.build_config("direct", None, incomplete_policy="drop"),
            output_root=self.root / "incomplete-drop",
        )
        prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
        self.assertEqual(tuple(prediction["series_id"].drop_duplicates()), ("A",))
        self.assertEqual(result.bundle.dimensions, (1, 4, 2))

    def test_drop_policy_keeps_complete_series_when_first_series_is_incomplete(self):
        incomplete = self.targets.loc[
            ~((self.targets["series_id"] == "A") & (self.targets["time"] == self.times[30]))
        ]
        incomplete.to_csv(self.target_path, index=False)
        result = run_canonical_config(
            self.build_config("direct", None, incomplete_policy="drop"),
            output_root=self.root / "incomplete-first-drop",
        )
        self.assertEqual(result.bundle.series_ids, ("B",))
        self.assertEqual(result.bundle.dimensions, (1, 4, 2))

    def test_two_series_id_columns_are_pinned_in_declared_order(self):
        targets = self.targets.assign(region=lambda frame: frame["series_id"].map({"A": "east", "B": "west"}))
        targets = targets[["region", "series_id", "time", "load", "power"]]
        targets.to_csv(self.target_path, index=False)
        config = self.build_config("direct", None, series_order=(["east", "A"], ["west", "B"]))
        target_source = DataSourceSpec(
            name="targets",
            source_type="file",
            columns=(
                ColumnSpec("region", "key", categorical=True),
                ColumnSpec("series_id", "key", categorical=True),
                ColumnSpec("load", "target"),
                ColumnSpec("power", "target"),
            ),
            history_path=str(self.target_path),
            time_col="time",
            series_id_cols=("region", "series_id"),
            availability="source_time",
        )
        config = ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=4,
                targets=("load", "power"),
                training_scope="global",
                series_id_cols=("region", "series_id"),
            ),
            data=DataSpec((target_source,)),
            features=config.features,
            strategy=config.strategy,
            estimator=config.estimator,
            probabilistic=config.probabilistic,
            validation={
                "forecast_origin": "2026-01-04T19:00:00",
                "history_steps": 10_000,
                "train_window_steps": 9_999,
                "fold_count": 1,
                "stride_steps": 4,
                "training_scope": {
                    "series_order": [["east", "A"], ["west", "B"]],
                    "incomplete_series_policy": "raise",
                    "unknown_series_policy": "raise",
                },
            },
            output=config.output,
        )
        result = run_canonical_config(config, output_root=self.root / "composite")
        self.assertEqual(result.bundle.input_schema["panel"]["known_series_ids"], [["east", "A"], ["west", "B"]])


if __name__ == "__main__":
    unittest.main()
