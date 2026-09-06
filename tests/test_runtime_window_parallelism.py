# -*- coding: utf-8 -*-
"""Fixed-step backtest window parallelism contracts."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

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
from fixtures.runtime_planning import plan_for_config
from model_pipeline.runner import CanonicalBaseModelRunner


class _TrackingRunner(CanonicalBaseModelRunner):
    lock = threading.Lock()
    active = 0
    max_active = 0

    @classmethod
    def reset(cls) -> None:
        with cls.lock:
            cls.active = 0
            cls.max_active = 0

    def fit(self, train_indices, **kwargs):
        with type(self).lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            time.sleep(0.03)
            return super().fit(train_indices, **kwargs)
        finally:
            with type(self).lock:
                type(self).active -= 1


class RuntimeWindowParallelismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "load.csv"
        times = pd.date_range("2026-01-01", periods=96, freq="1h")
        values = 100.0 + np.arange(len(times), dtype=float) * 0.5
        pd.DataFrame({"time": times, "load": values}).to_csv(
            self.data_path,
            index=False,
        )
        self.origin = cast(pd.Timestamp, pd.Timestamp(times.to_numpy()[-1]))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _config(
        self,
        workers: int,
        *,
        mode: str = "point",
    ) -> ForecastConfigSpec:
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=2,
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
                target_lags={"load": (2, 3, 4)},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations={
                    "direct": {
                        "layout": "independent_models",
                        "align_to_target": False,
                    }
                },
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(
                model_type="ridge" if mode == "point" else "qr",
                target_adapter="independent",
                params={"alpha": 1e-6 if mode == "point" else 0.01},
            ),
            probabilistic=(
                {"mode": "point"}
                if mode == "point"
                else {
                    "mode": "quantile",
                    "quantiles": [0.1, 0.5, 0.9],
                    "point_quantile": 0.5,
                    "intervals": [
                        {
                            "name": "q10_q90",
                            "lower_quantile": 0.1,
                            "upper_quantile": 0.9,
                        }
                    ],
                    "calibration": {
                        "method": "cqr",
                        "interval": "q10_q90",
                        "target_coverage": 0.8,
                        "calibration_windows": 3,
                        "min_windows": 1,
                        "min_scores": 1,
                        "label_availability_delay_steps": 0,
                    },
                }
            ),
            validation={
                "forecast_origin": self.origin.isoformat(),
                "history_steps": 50,
                "train_window_steps": 20,
                "fold_count": 3,
                "stride_steps": 2,
                "performance": {
                    "window_parallel_workers": workers,
                    "multi_output_n_jobs": 1 if workers > 1 else 2,
                },
            },
            output={"scenario_subpath": "window-parallel"},
        )

    def _runner(self, workers: int, *, mode: str = "point") -> _TrackingRunner:
        config = self._config(workers, mode=mode)
        return _TrackingRunner(
            config,
            SourceRegistry(config.data, self.root),
            self.origin,
        )

    def test_worker_resolver_reads_explicit_positive_value(self) -> None:
        self.assertEqual(plan_for_config(self._config(2)).window_workers, 2)

    def test_parallel_windows_overlap_fit_and_match_serial_results(self) -> None:
        serial = self._runner(1)
        serial_result = serial.run(self.root / "serial")
        serial_cv = pd.read_csv(serial_result.test_dir / "cv_plot_df.csv")

        _TrackingRunner.reset()
        parallel = self._runner(2)
        parallel_result = parallel.run(self.root / "parallel")
        parallel_cv = pd.read_csv(parallel_result.test_dir / "cv_plot_df.csv")

        self.assertGreaterEqual(_TrackingRunner.max_active, 2)
        pd.testing.assert_frame_equal(parallel_cv, serial_cv, check_exact=True)
        self.assertEqual(parallel_cv["window"].drop_duplicates().tolist(), [1, 2, 3])
        resolved = json.loads(
            (parallel_result.forecast_dir / "resolved_config.json").read_text(
                encoding="utf-8"
            )
        )
        resources = resolved["runtime"]["resources"]
        self.assertEqual(resources["execution_plan"]["selected_axis"], "window")
        self.assertIn(resources["cache"]["status"], {"disabled", "hit", "miss"})
        self.assertEqual(
            set(resources["stage_wall_seconds"]),
            {"raw_design", "backtest", "final_fit", "forecast_persist", "total"},
        )
        self.assertTrue(
            all(value >= 0.0 for value in resources["stage_wall_seconds"].values())
        )

    def test_identity_parallel_windows_do_not_materialize_target_history(self) -> None:
        runner = self._runner(2)
        original_target_history = runner.builder.target_history
        target_history_calls = 0

        def count_target_history(cutoff):
            nonlocal target_history_calls
            target_history_calls += 1
            return original_target_history(cutoff)

        cast(Any, runner.builder).target_history = count_target_history

        runner.run(self.root / "parallel-identity")

        self.assertEqual(
            target_history_calls,
            len(runner.backtest_windows()) + 1,
        )


    def test_parallel_quantile_windows_preserve_cqr_order(self) -> None:
        serial = self._runner(1, mode="quantile")
        serial_result = serial.run(self.root / "serial-quantile")
        serial_cv = pd.read_csv(serial_result.test_dir / "cv_plot_df.csv")

        parallel = self._runner(2, mode="quantile")
        parallel_result = parallel.run(self.root / "parallel-quantile")
        parallel_cv = pd.read_csv(parallel_result.test_dir / "cv_plot_df.csv")

        pd.testing.assert_frame_equal(parallel_cv, serial_cv, check_exact=True)
        self.assertIn("predict_pi80_lower", parallel_cv.columns)
        self.assertEqual(parallel_cv["window"].drop_duplicates().tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
