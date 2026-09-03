# -*- coding: utf-8 -*-
"""生产时间几何必须直接支持完整自然月折。"""

import pickle
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from model_forecasting.runtime import CanonicalBaseModelRunner, run_canonical_config
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.specs.config import parse_model_config
from forecasting_core.tensors import PointForecastTensor
from model_testing import validation
from model_testing.backtest import seasonal_naive_tensor


ROOT = Path(__file__).resolve().parents[1]


class _CountingRunner(CanonicalBaseModelRunner):
    created_horizons = []
    lock = threading.Lock()
    active = 0
    max_active = 0

    @classmethod
    def reset(cls) -> None:
        cls.created_horizons = []
        with cls.lock:
            cls.active = 0
            cls.max_active = 0

    def __init__(self, config, registry, origin, **kwargs):
        type(self).created_horizons.append(config.problem.horizon)
        super().__init__(config, registry, origin, **kwargs)

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


class CalendarMonthRuntimeGeometryTest(unittest.TestCase):
    def test_calendar_month_folds_use_each_target_month_horizon(self):
        self.assertTrue(
            hasattr(validation, "calendar_month_folds"),
            "model_testing.validation must own the production calendar-month geometry",
        )
        times = pd.date_range("2025-10-01", "2026-07-31", freq="1D")
        folds = validation.calendar_month_folds(
            times,
            train_window_days=120,
            fold_count=6,
            stride_months=1,
        )
        self.assertEqual(
            [fold.horizon for fold in folds],
            [28, 31, 30, 31, 30, 31],
        )
        self.assertTrue(
            all(len(fold.train_indices) == 120 for fold in folds)
        )
        self.assertTrue(
            all(fold.forecast_times[0].day == 1 for fold in folds)
        )

    def test_runtime_writes_six_complete_calendar_month_windows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "daily.csv"
            times = pd.date_range("2025-10-01", "2026-07-31", freq="1D")
            pd.DataFrame(
                {"time": times, "load": 100.0 + np.arange(len(times))}
            ).to_csv(data_path, index=False)
            config = parse_model_config(
                {
                    "schema_version": 2,
                    "problem": {
                        "time_col": "time",
                        "freq": "1D",
                        "horizon": 31,
                        "targets": ["load"],
                        "training_scope": "local",
                        "series_id_cols": [],
                    },
                    "data": {
                        "sources": [
                            {
                                "name": "target_history",
                                "source_type": "file",
                                "columns": [
                                    {
                                        "name": "load",
                                        "role": "target",
                                        "categorical": False,
                                    }
                                ],
                                "history_path": str(data_path),
                                "time_col": "time",
                                "series_id_cols": [],
                                "availability": "source_time",
                            }
                        ]
                    },
                    "features": {
                        "target_lags": {"load": [1, 7]},
                        "observed_past_lags": {},
                        "datetime_features": [],
                        "transformations": {},
                    },
                    "strategy": {"name": "recursive"},
                    "estimator": {
                        "model_type": "ridge",
                        "target_adapter": "independent",
                        "params": {"alpha": 1e-8},
                    },
                    "probabilistic": {"mode": "point"},
                    "validation": {
                        "forecast_origin": "2026-07-31T00:00:00",
                        "schedule_mode": "daily",
                        "horizon_mode": "calendar_month",
                        "train_window_days": 120,
                        "fold_count": 6,
                        "stride_months": 1,
                        "performance": {
                            "window_parallel_workers": 2,
                            "multi_output_n_jobs": 2,
                        },
                    },
                    "output": {"scenario_subpath": "calendar-runtime"},
                },
                source="<calendar-runtime-test>",
            )
            _CountingRunner.reset()
            with patch(
                "model_forecasting.runtime.CanonicalBaseModelRunner",
                _CountingRunner,
            ):
                result = run_canonical_config(
                    config,
                    output_root=root / "results",
                )
            self.assertEqual(
                _CountingRunner.created_horizons,
                [31, 28, 31, 30],
            )
            self.assertGreaterEqual(_CountingRunner.max_active, 2)
            cv = pd.read_csv(result.test_dir / "cv_plot_df.csv")
            horizons = cv.groupby("window")["time"].nunique().tolist()
            self.assertEqual(horizons, [28, 31, 30, 31, 30, 31])
            for _, frame in cv.groupby("window"):
                month = pd.to_datetime(frame["time"]).dt.to_period("M")
                self.assertEqual(month.nunique(), 1)

    def test_active_calendar_month_quantile_writes_complete_target_month(self):
        config_path = (
            ROOT
            / "config/aidc_power_month/route_A/freq_1day/baseline"
            / "lgbm_usmr_prob_mean_conformal.yaml"
        )
        config = load_yaml_config(config_path)
        assert isinstance(config, ForecastConfigSpec)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_canonical_config(config, output_root=Path(tmp_dir) / "results")

            cv = pd.read_csv(result.test_dir / "cv_plot_df.csv")
            self.assertEqual(
                cv.groupby("window")["time"].nunique().tolist(),
                [28, 31, 30, 31, 30, 31],
            )
            prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
            forecast_times = pd.to_datetime(prediction["time"])
            self.assertEqual(len(forecast_times), 31)
            self.assertEqual(forecast_times.min(), pd.Timestamp("2026-08-01"))
            self.assertEqual(forecast_times.max(), pd.Timestamp("2026-08-31"))
            self.assertTrue(
                {"predict_q10", "predict_q50", "predict_q90"}.issubset(prediction)
            )
            quantiles = prediction[["predict_q10", "predict_q50", "predict_q90"]]
            self.assertTrue((np.diff(quantiles.to_numpy(), axis=1) >= 0.0).all())
            self.assertTrue(
                (result.test_dir / "test_scores_probabilistic_df.csv").is_file()
            )
            with (result.model_dir / "model.pkl").open("rb") as file:
                restored = pickle.load(file)
            self.assertEqual(restored.dimensions, (1, 31, 1))

    def test_active_month_end_runtime_writes_reloadable_month_axis_bundle(self):
        config_path = (
            ROOT
            / "config/aidc_power_month/route_A/freq_1month/window_length_7"
            / "st_usmd_mean.yaml"
        )
        config = load_yaml_config(config_path)
        assert isinstance(config, ForecastConfigSpec)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_canonical_config(config, output_root=Path(tmp_dir) / "results")

            prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
            self.assertEqual(len(prediction), 1)
            self.assertEqual(pd.Timestamp(prediction["time"].iloc[0]), pd.Timestamp("2026-08-31"))
            cv = pd.read_csv(result.test_dir / "cv_plot_df.csv")
            self.assertEqual(cv.groupby("window")["time"].nunique().tolist(), [1, 1, 1, 1])
            with (result.model_dir / "model.pkl").open("rb") as file:
                restored = pickle.load(file)
            self.assertEqual(restored.dimensions, (1, 1, 1))
            self.assertEqual(restored.canonical_problem["freq"], "1ME")

    def test_month_end_seasonal_naive_uses_previous_month_step(self):
        history_times = pd.DatetimeIndex([pd.Timestamp("2026-07-31")])
        history = PointForecastTensor(
            values=np.array([[[42.0]]]),
            series_ids=("__local__",),
            forecast_times=history_times,
            targets=("load",),
        )
        builder = SimpleNamespace(
            config=SimpleNamespace(validation={}),
            offset=pd.tseries.frequencies.to_offset("1ME"),
            target_history=lambda _origin: history,
        )
        result = seasonal_naive_tensor(
            builder,
            pd.Timestamp("2026-07-31"),
            pd.DatetimeIndex([pd.Timestamp("2026-08-31")]),
        )
        np.testing.assert_array_equal(result.values, np.array([[[42.0]]]))


if __name__ == "__main__":
    unittest.main()
