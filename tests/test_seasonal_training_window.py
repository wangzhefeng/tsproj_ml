"""季节残差迁移到显式训练窗：真实拟合与窗口外污染隔离。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from model_pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from model_pipeline.lifecycle import run_lifecycle
import test_canonical_runtime_smoke as smoke


class SeasonalTrainingWindowTest(unittest.TestCase):
    def test_residual_final_entrypoints_reject_before_training_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'target.csv'
            times = pd.date_range('2026-01-01', periods=360, freq='1h')
            pd.DataFrame({'time': times, 'load': 100 + np.sin(np.arange(360) / 12)}).to_csv(path, index=False)
            base = smoke.CanonicalRuntimeSmokeTest().build_config(
                path, mode='point', strategy='direct', horizon=24)
            transforms = base.features.canonical_payload()['transformations']
            transforms['seasonal_baseline'] = {'column': 'load', 'period': 24, 'days': 2}
            config = replace(base, features=replace(base.features, transformations=transforms),
                validation={'forecast_origin': times[302].isoformat(), 'history_steps': 300,
                            'fold_count': 2, 'stride_steps': 24,
                            'training_window': {'kind': 'rolling', 'history_steps': 120}})
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, root), times[302])
            output = root / 'output'
            operations = {
                'inputs': runner.final_bundle_inputs,
                'fit': lambda: runner.fit_final((), np.empty((0, 24, 1))),
                'bundle': lambda: runner.build_final_bundle(None, None, None, None, None),
                'lifecycle': lambda: run_lifecycle(runner, output_root=output),
                'entry': lambda: run_canonical_config(config, output_root=output),
            }
            for name, operation in operations.items():
                with self.subTest(entry=name):
                    with self.assertRaisesRegex(ValueError, 'seasonal_baseline.*backtest-only'):
                        operation()
                    self.assertFalse(output.exists())

    def test_rolling_residual_fit_preserves_history_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'target.csv'
            times = pd.date_range('2026-01-01', periods=360, freq='1h')
            frame = pd.DataFrame({'time': times, 'load': 100 + np.sin(np.arange(360) / 12)})
            frame.to_csv(path, index=False)
            base = smoke.CanonicalRuntimeSmokeTest().build_config(
                path, mode='point', strategy='direct', horizon=24)
            transforms = base.features.canonical_payload()['transformations']
            transforms['seasonal_baseline'] = {'column': 'load', 'period': 24, 'days': 2}
            config = replace(base, features=replace(base.features, transformations=transforms),
                validation={'forecast_origin': times[302].isoformat(), 'history_steps': 300,
                            'fold_count': 2, 'stride_steps': 24,
                            'training_window': {'kind': 'rolling', 'history_steps': 120}})

            def predict():
                runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, root), times[302])
                window = runner.backtest_windows()[0]
                fold = runner.for_backtest_window(window)
                scaler, transform, _, _, artifact = fold.fit(window.train_indices)
                designs, provider = fold.forecast_designs(fold.origin, scaler, transform)
                prediction = fold.predict(artifact, designs, provider, fold.forecast_times(fold.origin), transform)
                return window, prediction.values

            window, before = predict()
            self.assertTrue(np.isfinite(before).all())
            self.assertEqual(before.shape, (1, 24, 1))
            start = pd.Timestamp(window.metadata['raw_history_start'])
            frame.loc[frame.time < start, 'load'] += 1e6
            frame.to_csv(path, index=False)
            _, after = predict()
            np.testing.assert_array_equal(before, after)
