# -*- coding: utf-8 -*-
"""E1: shared rolling-origin splitter and CanonicalBaseModelRunner contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_testing import validation
from model_forecasting.runtime import CanonicalBaseModelRunner, SourceRegistry


def _origins(count: int) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range("2026-01-01", periods=count, freq="1h"))


def _geometry(horizon: int = 2) -> validation.TimeGeometry:
    return validation.TimeGeometry(
        offset=pd.tseries.frequencies.to_offset("1h"),
        horizon=horizon,
    )


class RollingOriginFoldContractTest(unittest.TestCase):
    def test_folds_exclude_overlapping_training_samples(self):
        origins = _origins(24)
        folds = validation.rolling_origin_folds(
            origins,
            _geometry(horizon=2),
            history_length=None,
            window_length=10,
            max_windows=3,
            stride=2,
        )
        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[-1].window, 3)
        for fold in folds:
            holdout_label_start = fold.origin + pd.Timedelta(hours=1)
            for index in fold.train_indices:
                self.assertLess(
                    origins[index] + pd.Timedelta(hours=2),
                    holdout_label_start,
                )
            self.assertLess(
                max(origins[i] for i in fold.train_indices) + pd.Timedelta(hours=2),
                holdout_label_start,
            )

    def test_folds_are_chronologically_ordered(self):
        origins = _origins(24)
        folds = validation.rolling_origin_folds(
            origins,
            _geometry(),
            history_length=None,
            window_length=5,
            max_windows=4,
            stride=3,
        )
        origins_seq = [fold.origin for fold in folds]
        self.assertEqual(origins_seq, sorted(origins_seq))

    def test_no_training_samples_raises(self):
        # H=2 hourly: the only candidate (immediate predecessor) has
        # label_end == holdout label_start -> excluded -> empty train set
        origins = _origins(2)
        with self.assertRaises(ValueError):
            validation.rolling_origin_folds(
                origins,
                _geometry(),
                history_length=None,
                window_length=10,
                max_windows=1,
                stride=1,
            )

    def test_validate_no_overlap_rejects_overlap(self):
        origins = _origins(6)
        geometry = _geometry(horizon=2)
        # last origin overlaps the holdout labels
        with self.assertRaises(ValueError):
            validation.validate_no_overlap(
                origins,
                (0, 4),
                origins[5],
                geometry,
            )


class CanonicalBaseModelRunnerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _runner(self, strategy: str = "recursive") -> CanonicalBaseModelRunner:
        from tests.test_ensemble_parity import (
            _parity_data,
            _single_model_config,
        )

        data_path = _parity_data(self.root / "local.csv")
        config = _single_model_config(data_path)
        from dataclasses import replace

        config = replace(config, strategy=_strategy(strategy))
        registry = SourceRegistry(config.data, self.root)
        origin = pd.Timestamp("2026-01-03T23:00:00")
        return CanonicalBaseModelRunner(config, registry, origin)

    def test_runner_matches_run_canonical_config_outputs(self):
        runner = self._runner("recursive")
        windows = runner.backtest_windows()
        self.assertEqual(len(windows), 1)
        window = windows[0]
        (
            scaler,
            transform,
            _X,
            _Y,
            artifact,
        ) = runner.fit(window.train_indices)
        designs, provider = runner.forecast_designs(
            window.origin, scaler, transform
        )
        times = runner.forecast_times(window.origin)
        prediction = runner.predict(
            artifact, designs, provider, times, transform
        )
        self.assertEqual(prediction.values.shape, (1, 2, 1))
        self.assertAlmostEqual(
            float(prediction.values[0, 0, 0]), 54.99999999999561, places=9
        )

    def test_rejects_missing_strategy(self):
        from dataclasses import replace

        from tests.test_ensemble_parity import (
            _parity_data,
            _single_model_config,
        )

        data_path = _parity_data(self.root / "local.csv")
        # v4: base-only spec rejects missing strategy at construction time
        with self.assertRaises(ValueError):
            replace(_single_model_config(data_path), strategy=None)


def _strategy(name: str):
    from model_forecasting.specs.strategy import ForecastStrategySpec

    return ForecastStrategySpec(name)


if __name__ == "__main__":
    unittest.main()
