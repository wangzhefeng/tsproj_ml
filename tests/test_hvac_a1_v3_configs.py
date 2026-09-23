"""A1_all v3 的完整配置矩阵、真实资产与生产设计，不拟合正式模型。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from forecasting_core.specs import ForecastConfigSpec
from config.aidc_hvac_load_5min.scripts.v3.forecast_data.build_model_configs import build_documents
from config.aidc_hvac_load_5min.scripts.forecast_data.build_model_configs import publish
from tests.test_hvac_model_configs import builder_for, expected_paths as legacy_paths
from model_pipeline.supervised_design import minimum_history_rows, raw_history_backtest_windows

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'config/aidc_hvac_load_5min'
GROUPS = ('baseline', 'add_datetime_holiday', 'add_weather', 'add_endogenous_it',
          'add_endogenous_route', 'add_context')
DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
METHODS = ('direct', 'direct-pointwise', 'direct-pointwise-horizon', 'recursive',
           'mimo', 'recmo', 'dirrec', 'dirmo', 'dirrecmo')


def expected_paths():
    paths = {Path(g) / 'A1_all' / d / f'lgbm_{m}.yaml'
             for g in GROUPS for d in DEVICES for m in METHODS}
    for group in ('baseline', 'add_context'):
        for device in DEVICES:
            parent = Path(group) / 'A1_all' / device
            paths.add(parent / 'ets.yaml')
            paths.update(parent / f'{model}_{method}.yaml'
                         for model in ('ridge', 'xgboost', 'st') for method in METHODS[:4])
    return paths


class A1V3ConfigsTest(unittest.TestCase):
    def test_generator_matrix_and_version_scoped_publication(self):
        documents = build_documents()
        self.assertEqual(set(documents), expected_paths())
        self.assertEqual(len(documents), 160)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = Path('baseline/old.yaml')
            publish({legacy: 'old\n'}, root)
            self.assertEqual(publish(documents, root, config_version='a1')['created'], len(documents))
            self.assertEqual(publish({legacy: 'old\n'}, root, check=True)['created'], 0)
            self.assertEqual(publish(documents, root, check=True, config_version='a1')['created'], 0)
            first = next(iter(documents))
            with self.assertRaises(FileExistsError):
                publish({**documents, first: 'different'}, root, config_version='a1')
            for invalid in (Path('baseline/A1_all/v3/hvac_all_devices/bad.yaml'),
                            Path('baseline/../bad.yaml'), Path('baseline/old.yaml')):
                with self.assertRaises(ValueError):
                    publish({invalid: 'bad'}, root, config_version='a1')
            extra = root / 'add_context/A1_all/hvac_all_devices/unknown.yaml'
            extra.write_text('unknown')
            with self.assertRaises(ValueError):
                publish(documents, root, config_version='a1')

    def test_group_orthogonality_and_context_only_changes_window(self):
        documents = {p: yaml.safe_load(text) for p, text in build_documents().items()}
        for relative, payload in documents.items():
            group = relative.parts[0]
            baseline = documents[Path('baseline', *relative.parts[1:])]
            if group == 'add_context':
                self.assertEqual({k: v for k, v in payload.items() if k not in ('validation', 'output')},
                                 {k: v for k, v in baseline.items() if k not in ('validation', 'output')})
                continue
            candidate = yaml.safe_load(yaml.safe_dump(payload))
            candidate['output'] = baseline['output']
            candidate['data']['sources'] = candidate['data']['sources'][:1]
            candidate['features']['datetime_features'] = []
            candidate['features']['observed_past_lags'] = {}
            self.assertEqual(candidate, baseline, str(relative))

    def test_real_matrix_and_production_designs(self):
        physical = {p.relative_to(DIRECTORY) for p in DIRECTORY.rglob('*.yaml')}
        self.assertEqual(physical, legacy_paths() | expected_paths())
        frames = {}
        for relative in sorted(expected_paths()):
            with self.subTest(path=relative):
                config = load_yaml_config(DIRECTORY / relative)
                assert isinstance(config, ForecastConfigSpec)
                group = relative.parts[0]
                self.assertEqual(config.problem.targets, ('hvac_total_load_AB',))
                self.assertEqual((config.problem.freq, config.problem.horizon), ('5min', 288))
                train_days, fold_count = (150, 12) if group == 'add_context' else (90, 72)
                self.assertEqual(config.validation['train_history_steps'], train_days * 288)
                self.assertEqual(config.validation['fold_count'], fold_count)
                self.assertEqual(config.validation['train_window_steps'],
                                 train_days * 288 - minimum_history_rows(config) - 288 + 1)
                self.assertEqual(config.output['scenario_subpath'],
                                 'aidc_hvac_load_5min/' + str(relative.parent))
                self.assertEqual(bool(config.features.datetime_features), group == 'add_datetime_holiday')
                if relative.name == 'ets.yaml':
                    self.assertEqual(config.estimator.model_type, 'ets')
                    self.assertEqual(dict(config.features.target_lags), {})
                    self.assertEqual(dict(config.features.transformations), {})
                    self.assertEqual(config.estimator.params['seasonal_periods'], 288)
                else:
                    self.assertEqual(config.estimator.model_type, {
                        'lgbm': 'lightgbm', 'ridge': 'ridge', 'xgboost': 'xgboost', 'st': 'st',
                    }[relative.stem.split('_')[0]])
                    self.assertEqual(dict(config.features.target_lags), {
                        'hvac_total_load_AB': tuple(d * 288 for d in (1, 2, 3, 4, 5, 6, 7, 14, 21, 28))})
                    advanced = config.features.transformations['advanced']
                    self.assertEqual(tuple(advanced['rolling']['windows']),
                                     tuple(d * 288 for d in (1, 2, 4, 7, 14, 28)))
                sources = {s.name: s for s in config.data.sources}
                self.assertEqual(set(sources), {'target_history'} | (
                    {'chinese_holiday'} if group == 'add_datetime_holiday' else
                    {'weather'} if group == 'add_weather' else
                    {'covariate_history'} if group in ('add_endogenous_it', 'add_endogenous_route') else set()))
                covariates = {'add_endogenous_it': ['it_subset_load'],
                              'add_endogenous_route': ['hvac_total_load_A', 'hvac_total_load_B']}.get(group, [])
                self.assertEqual(dict(config.features.observed_past_lags), {
                    c: tuple(d * 288 for d in (1, 2, 7, 14, 28)) for c in covariates})
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
                self.assertEqual(len(folds), fold_count)
                self.assertEqual(folds[0].origin, pd.Timestamp('2026-09-04 23:55')
                                 if group == 'add_context' else pd.Timestamp('2026-07-06 23:55'))
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
