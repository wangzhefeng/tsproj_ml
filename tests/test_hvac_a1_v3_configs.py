"""A1_all v3 的完整配置矩阵、真实资产与生产设计，不拟合正式模型。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from config.aidc_hvac_load_5min.scripts.v3.forecast_data.build_model_configs import build_documents
from config.aidc_hvac_load_5min.scripts.forecast_data.build_model_configs import publish
from tests.test_hvac_model_configs import builder_for, expected_paths as legacy_paths
from model_pipeline.supervised_design import raw_history_backtest_windows

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'config/aidc_hvac_load_5min'
GROUPS = ('baseline', 'add_weather', 'add_endogenous_it', 'add_endogenous_route')
DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
METHODS = ('direct', 'direct-pointwise', 'direct-pointwise-horizon', 'recursive',
           'mimo', 'recmo', 'dirrec', 'dirmo', 'dirrecmo')


def expected_paths():
    return {Path(g) / 'A1_all/v3' / d / f'lgbm_{m}.yaml'
            for g in GROUPS for d in DEVICES for m in METHODS}


class A1V3ConfigsTest(unittest.TestCase):
    def test_generator_matrix_and_version_scoped_publication(self):
        documents = build_documents()
        self.assertEqual(set(documents), expected_paths())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = Path('baseline/old.yaml')
            publish({legacy: 'old\n'}, root)
            self.assertEqual(publish(documents, root, config_version='v3')['created'], len(documents))
            self.assertEqual(publish({legacy: 'old\n'}, root, check=True)['created'], 0)
            self.assertEqual(publish(documents, root, check=True, config_version='v3')['created'], 0)
            first = next(iter(documents))
            with self.assertRaises(FileExistsError):
                publish({**documents, first: 'different'}, root, config_version='v3')

    def test_real_matrix_and_production_designs(self):
        physical = {p.relative_to(DIRECTORY) for p in DIRECTORY.rglob('*.yaml')}
        self.assertEqual(physical, legacy_paths() | expected_paths())
        frames = {}
        for relative in sorted(expected_paths()):
            with self.subTest(path=relative):
                config = load_yaml_config(DIRECTORY / relative)
                group = relative.parts[0]
                self.assertEqual(config.problem.targets, ('hvac_total_load_AB',))
                self.assertEqual((config.problem.freq, config.problem.horizon), ('5min', 288))
                self.assertEqual(config.validation['train_history_steps'], 14 * 288)
                self.assertEqual(config.validation['fold_count'], 148)
                sources = {s.name: s for s in config.data.sources}
                self.assertEqual(set(sources), {'target_history', 'chinese_holiday'} | (
                    {'weather'} if group == 'add_weather' else
                    {'covariate_history'} if group in ('add_endogenous_it', 'add_endogenous_route') else set()))
                covariates = {'add_endogenous_it': ['it_subset_load'],
                              'add_endogenous_route': ['hvac_total_load_A', 'hvac_total_load_B']}.get(group, [])
                self.assertEqual(dict(config.features.observed_past_lags), {c: (288, 576) for c in covariates})
                for source in config.data.sources:
                    if source.source_type != 'file':
                        continue
                    if source.history_path not in frames:
                        frames[source.history_path] = pd.read_csv(ROOT / source.history_path)
                    frame = frames[source.history_path]
                    self.assertEqual(len(frame), 46656)
                    self.assertTrue(np.isfinite(frame[[c.name for c in source.columns]].to_numpy(dtype=float)).all())
                    times = pd.DatetimeIndex(pd.to_datetime(frame[source.time_col]))
                    self.assertTrue(times.equals(pd.date_range('2026-04-08', '2026-09-16 23:55', freq='5min')))
                folds = raw_history_backtest_windows(builder_for(config, None), pd.Timestamp('2026-09-16 23:55'))
                self.assertEqual(len(folds), 148)
                self.assertEqual(folds[0].origin, pd.Timestamp('2026-04-21 23:55'))
                self.assertEqual(folds[-1].origin, pd.Timestamp('2026-09-15 23:55'))
                for previous, current in zip(folds, folds[1:]):
                    self.assertEqual(current.origin - previous.origin, pd.Timedelta(days=1))
                for fold in (folds[0], folds[-1]):
                    builder = builder_for(config, pd.Timestamp(fold.metadata['raw_history_start']))
                    designs, labels = builder.training_row(fold.origin - pd.Timedelta(days=1))
                    self.assertTrue(all(np.isfinite(x).all() for x in designs))
                    self.assertTrue(np.isfinite(labels).all())
                    request = builder.request(fold.origin)
                    compiled = builder.compiler.compile(builder.registry.materialize(request), request,
                                                        horizon_steps=(1, 144, 288))
                    self.assertTrue(np.isfinite(compiled.frame[list(compiled.schema.feature_names)].to_numpy(dtype=float)).all())
                    for proof in compiled.visibility_proof:
                        if proof.source_name == 'covariate_history':
                            self.assertLessEqual(proof.source_time, fold.origin)
                            self.assertIsNone(proof.provider)

    def test_cli_check_outside_repo(self):
        with TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, str(DIRECTORY / 'scripts/v3/forecast_data/build_model_configs.py'),
                                      '--check'], cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == '__main__':
    unittest.main()
