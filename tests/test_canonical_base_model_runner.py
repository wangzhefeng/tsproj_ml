# -*- coding: utf-8 -*-
"""E1: shared rolling-origin splitter and CanonicalBaseModelRunner contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_testing import geometry as validation
from model_pipeline.runner import CanonicalBaseModelRunner, SourceRegistry


def _origins(count: int) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range("2026-01-01", periods=count, freq="1h"))


def _geometry(horizon: int = 2) -> validation.TimeGeometry:
    return validation.TimeGeometry(
        offset=pd.tseries.frequencies.to_offset("1h"),
        horizon=horizon,
    )


class RollingOriginFoldContractTest(unittest.TestCase):
    @staticmethod
    def folds(origins, raw_steps, fold_count, stride_steps):
        from types import SimpleNamespace
        from forecasting_core.specs.config import parse_model_config
        from model_pipeline.supervised_design import temporal_backtest_windows
        from tests.test_ensemble_runtime import _member_doc
        document = _member_doc("direct", "ridge", "geometry")
        document["validation"] = {
            "history_steps": len(origins), "fold_count": fold_count, "stride_steps": stride_steps,
            "training_window": {"kind": "rolling", "history_steps": raw_steps},
        }
        config = parse_model_config(document, source="geometry-test")
        builder = SimpleNamespace(config=config, offset=_geometry().offset,
            registry=SimpleNamespace(target_history_coverage=lambda: (SimpleNamespace(times=pd.DatetimeIndex(origins)),)))
        return temporal_backtest_windows(builder, origins[-1])

    def test_folds_exclude_overlapping_training_samples(self):
        folds = self.folds(_origins(24), 14, 3, 2)
        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[-1].window, 3)
        for fold in folds:
            self.assertLess(pd.Timestamp(fold.metadata["training_label_end_max"]),
                            pd.Timestamp(fold.metadata["label_start"]))
            self.assertEqual(fold.train_indices, tuple(range(10)))

    def test_folds_are_chronologically_ordered(self):
        folds = self.folds(_origins(24), 9, 4, 3)
        self.assertEqual([f.origin for f in folds], sorted(f.origin for f in folds))

    def test_no_training_samples_raises(self):
        with self.assertRaises(ValueError):
            self.folds(_origins(2), 14, 1, 1)

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
        ) = runner.for_backtest_window(window).fit(window.train_indices)
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

    def test_final_bundle_inputs_use_configured_training_window(self):
        from dataclasses import replace

        runner = self._runner("recursive")
        validation_payload = dict(runner.config.validation)
        validation_payload["training_window"] = {"kind": "rolling", "history_steps": 10}
        config = replace(runner.config, validation=validation_payload)
        registry = SourceRegistry(config.data, self.root)
        windowed = CanonicalBaseModelRunner(config, registry, runner.origin)

        _scaler, _transform, X_by_call, Y = windowed.final_bundle_inputs()

        self.assertTrue(X_by_call)
        self.assertEqual(X_by_call[0].shape[0], 5 * len(windowed.series_ids))
        self.assertEqual(Y.shape[0], 5 * len(windowed.series_ids))

    def test_supervised_design_is_limited_to_configured_history_origins(self):
        from dataclasses import replace

        runner = self._runner("recursive")
        validation_payload = {
            **dict(runner.config.validation),
            "history_steps": 8,
            'training_window': {'kind': 'rolling', 'history_steps': 13},
            "fold_count": 1,
            "stride_steps": 2,
        }
        config = replace(runner.config, validation=validation_payload)
        limited = CanonicalBaseModelRunner(
            config,
            SourceRegistry(config.data, self.root),
            runner.origin,
        )

        self.assertEqual(len(limited.supervised_origins), 8)

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
    from forecasting_core.specs.strategy import ForecastStrategySpec

    return ForecastStrategySpec(name)


if __name__ == "__main__":
    unittest.main()
