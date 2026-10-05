"""批统计不得逐原点重扫；数值参照来自独立窗口切片。"""
from dataclasses import replace
from pathlib import Path
from typing import cast
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering.indexed import compile_indexed_history
from pipeline.supervised_design import SupervisedDesignBuilder
from tests.test_raw_history_window import make_config


class CompilerArrayKernelsTest(unittest.TestCase):
    def test_all_history_statistics_match_independent_windows_in_all_paths(self):
        stats = ('mean', 'std', 'min', 'max', 'median', 'skew', 'kurt', 'max_diff', 'min_diff', 'entropy')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'target.csv'
            times = pd.date_range('2026-01-01', periods=52, freq='h')
            cases = {
                'constant': np.full(52, 7.0),
                'zero': np.zeros(52),
                'near_constant': 7 + np.sin(np.arange(52)) * 0.001,
                'nonconstant': np.sin(np.arange(52)) + np.arange(52) / 10,
                'transition': np.r_[np.full(27, 7.0), np.arange(25.)],
            }
            for case, values in cases.items():
                pd.DataFrame({'time': times, 'load': values}).to_csv(path, index=False)
                parsed = pd.read_csv(path)['load']
                config = make_config(path, strategy='direct')
                transforms = config.features.canonical_payload()['transformations']
                transforms['advanced'] = {
                    'rolling': {'columns': ['load'], 'windows': [1, 2, 4, 8], 'stats': list(stats)},
                    'expanding': {'columns': ['load'], 'stats': list(stats)},
                }
                config = replace(config, features=replace(config.features, transformations=transforms))
                builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, directory),
                    history_start=cast(pd.Timestamp, times[10]))
                origins = tuple(times[25:30])
                # 短窗的std/skew/kurt/diff按既有合同警告并返回0；显式验证该警告。
                with self.assertWarnsRegex(RuntimeWarning, 'falling back to 0.0'):
                    single = [builder.training_row(origin)[0] for origin in origins]
                batch = [row[0] for row in builder.training_rows(origins)]
                indexed = compile_indexed_history(builder.compiler,
                    builder.registry.materialize(builder.request(cast(pd.Timestamp, times[47]))), pd.DatetimeIndex(origins), (1, 2),
                    registry=builder.registry)
                self.assertIsNotNone(indexed)
                assert indexed is not None
                for i, origin in enumerate(origins):
                    end = times.get_loc(origin)
                    assert isinstance(end, (int, np.integer))
                    history = parsed.iloc[10:end + 1]
                    windows = {'expanding': history}
                    windows.update({f'rolling_{window}': history.iloc[-window:] for window in (1, 2, 4, 8)})
                    for kind, sample in windows.items():
                        for stat in stats:
                            if stat == 'entropy':
                                mass = np.abs(sample.to_numpy())
                                probabilities = mass[mass > 0] / mass.sum() if mass.sum() else np.array([])
                                expected = -np.sum(probabilities * np.log2(probabilities))
                            elif stat in {'max_diff', 'min_diff'}:
                                diffs = sample.diff().dropna()
                                expected = 0.0 if diffs.empty else getattr(diffs, stat.split('_')[0])()
                            else:
                                expected = getattr(sample, stat)()
                                expected = 0.0 if pd.isna(expected) else expected
                            name = (f'load_expanding_{stat}' if kind == 'expanding'
                                    else f'load_rolling_{stat}_{kind.split("_")[1]}')
                            column = builder.feature_schema.index(name)
                            for mode, calls in (('single', single[i]), ('batch', batch[i]),
                                                ('indexed', tuple(design[i:i + 1] for design in indexed[0]))):
                                with self.subTest(case=case, origin=origin, feature=name, mode=mode):
                                    for design in calls:
                                        np.testing.assert_allclose(design[0, column], expected, rtol=1e-7, atol=1e-7)

    def test_batch_statistics_use_vector_kernels_with_fold_lower_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.csv"
            times = pd.date_range("2026-01-01", periods=52, freq="h")
            for values in (100 + np.arange(52, dtype=float) ** 1.5,
                           np.full(52, 7.0),
                           1e9 + np.sin(np.arange(52)) * 0.01):
                pd.DataFrame({"time": times, "load": values}).to_csv(path, index=False)
                config = make_config(path, strategy="direct")
                for lower in (times[10], times[20]):
                    builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, directory), history_start=lower)
                    origins = tuple(times[25:30])
                    # This is a complexity contract: batch must not invoke the scalar
                    # reducer for every origin/window after already computing rolling.
                    with patch("feature_engineering.compiler.history_statistic",
                               side_effect=AssertionError("scalar history rescan")):
                        rows = builder.training_rows(origins)
                    parsed = pd.read_csv(path)["load"].to_numpy()
                    for row, origin in zip(rows, origins):
                        end = times.get_loc(origin)
                        begin = times.get_loc(lower)
                        history = pd.Series(parsed[begin:end + 1])
                        expected = {
                            "load_rolling_mean_3": history.iloc[-3:].mean(),
                            "load_expanding_mean": history.mean(),
                            "load_expanding_std": history.std(),
                        }
                        for design in row[0]:
                            for name, value in expected.items():
                                # Std is small even for a large baseline: do not hide
                                # catastrophic cancellation under a relative-to-1e9 bound.
                                np.testing.assert_allclose(design[0, builder.feature_schema.index(name)], value,
                                                           rtol=2e-5 if name.endswith("std") else 1e-12,
                                                           atol=2e-7 if name.endswith("std") else 1e-10)


if __name__ == "__main__":
    unittest.main()
