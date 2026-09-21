"""HVAC 四组配置、真实资产与 as-of 设计验证；不拟合模型。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from config.aidc_hvac_load_5min.scripts.forecast_data.build_model_configs import publish
from data_loading import BUILTIN_GENERATORS, SourceRegistry
from model_pipeline.supervised_design import (
    SupervisedDesignBuilder, minimum_history_rows, raw_history_backtest_windows,
)
from scripts.check_model_configs import check_model_yaml

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'config/aidc_hvac_load_5min'
TEMPLATES = ROOT / 'config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/baseline'
GROUPS = ('baseline', 'add_weather', 'add_endogenous_it', 'add_endogenous_route')
VERSIONS = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')
BUILDINGS = ('A1', 'A2', 'A3', 'ALL')
VARIANTS = ('direct', 'direct-pointwise', 'direct-pointwise-horizon', 'recursive',
            'mimo', 'recmo', 'dirrec', 'dirmo', 'dirrecmo')
# 独立黄金值：总天数、训练天数、滚动折数，不从生成脚本导入。
WINDOWS = {
    'A1_data': (58, 14, 44), 'A1_data_with_it': (22, 14, 8),
    'A2_data': (54, 14, 40), 'A2_data_with_it': (47, 14, 33),
    'A3_data': (35, 7, 28), 'A3_data_with_it': (16, 7, 9),
    'data': (24, 7, 17), 'data_with_it': (14, 7, 7),
}


def expected_paths():
    return {Path(g) / v / r / b / f'lgbm_{method}.yaml'
            for g in GROUPS for v in VERSIONS for r in ROUTES
            for b in BUILDINGS for method in VARIANTS}


def builder_for(config, start):
    return SupervisedDesignBuilder(
        config, SourceRegistry(config.data, ROOT, generators=BUILTIN_GENERATORS),
        history_start=start,
    )


class HvacModelConfigsTest(unittest.TestCase):
    def paths(self):
        paths = sorted(p for p in DIRECTORY.rglob('*.yaml')
                       if p.relative_to(DIRECTORY).parts[1:3] != ('A1_all', 'v3'))
        self.assertEqual({p.relative_to(DIRECTORY) for p in paths}, expected_paths())
        return paths

    def test_complete_physical_matrix(self):
        self.assertEqual(len(self.paths()), 576)

    def test_generator_check_from_another_directory_and_write_protection(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            process = subprocess.run(
                [sys.executable, str(DIRECTORY / 'scripts/forecast_data/build_model_configs.py'), '--check'],
                cwd=output, capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            first = Path('baseline/example.yaml')
            second = Path('add_weather/example.yaml')
            documents = {first: 'first\n'}
            self.assertEqual(publish(documents, output)['created'], 1)
            before = (output / first).stat().st_mtime_ns
            self.assertEqual(publish(documents, output)['created'], 0)
            self.assertEqual((output / first).stat().st_mtime_ns, before)
            with self.assertRaises(FileExistsError):
                publish({first: 'different\n', second: 'second\n'}, output)
            self.assertFalse((output / second).exists())
            self.assertEqual((output / first).read_text(), 'first\n')
            with self.assertRaises(FileNotFoundError):
                publish({**documents, second: 'second\n'}, output, check=True)
            self.assertFalse((output / second).exists())
            with self.assertRaises(ValueError):
                publish({second: 'second\n'}, output)
            self.assertFalse((output / second).exists())

    def test_config_contract_assets_and_all_fold_dates(self):
        frames = {}
        for path in self.paths():
            with self.subTest(path=path):
                group, version, route, building, filename = path.relative_to(DIRECTORY).parts
                config = load_yaml_config(path)
                _, errors = check_model_yaml(str(path))
                self.assertEqual(errors, [])
                template = load_yaml_config(TEMPLATES / filename)
                self.assertEqual(config.strategy, template.strategy)
                self.assertEqual(config.estimator, template.estimator)
                self.assertEqual(config.features.datetime_features, template.features.datetime_features)
                self.assertEqual(config.problem.targets, ('hvac_total_load_' + route[-1],))
                self.assertEqual((config.problem.freq, config.problem.horizon), ('5min', 288))
                self.assertEqual(config.probabilistic['mode'], 'point')
                self.assertEqual(config.output['scenario_subpath'],
                                 'aidc_hvac_load_5min/' + str(path.relative_to(DIRECTORY).parent))
                source = config.data.sources[0]
                stem = ('data' if building == 'ALL' else building + '_data')
                if group == 'add_endogenous_it':
                    stem += '_with_it'
                self.assertEqual(source.history_path,
                                 f'dataset/aidc_hvac_load_5min/forecast_data/data_v1/{version}/{route}/{stem}.csv')
                total_days, train_days, fold_count = WINDOWS[stem]
                for s in config.data.sources:
                    if s.source_type != 'file':
                        continue
                    if s.history_path not in frames:
                        frames[s.history_path] = pd.read_csv(ROOT / s.history_path)
                    frame = frames[s.history_path]
                    self.assertEqual(len(frame), total_days * 288)
                    ts = pd.DatetimeIndex(pd.to_datetime(frame[s.time_col]))
                    self.assertTrue(ts.equals(pd.date_range(ts[0], periods=len(ts), freq='5min')))
                    cols = [c.name for c in s.columns]
                    self.assertTrue(np.isfinite(frame[cols].to_numpy(dtype=float)).all())
                self.assertEqual(config.validation['train_history_steps'], train_days * 288)
                n_train = train_days * 288 - minimum_history_rows(config) - 288 + 1
                self.assertGreater(n_train, 0)
                self.assertEqual(config.validation['train_window_steps'], n_train)
                self.assertEqual(config.validation['history_steps'], n_train + fold_count * 288)
                self.assertEqual(config.validation['fold_count'], fold_count)
                self.assertEqual(config.validation['stride_steps'], 288)
                ts = pd.DatetimeIndex(pd.to_datetime(frames[source.history_path]['time']))
                cutoff = pd.Timestamp(config.validation['forecast_origin'])
                self.assertEqual(cutoff, ts[-1])
                folds = raw_history_backtest_windows(builder_for(config, None), cutoff)
                self.assertEqual(len(folds), fold_count)
                self.assertEqual(folds[0].origin, ts[0] + pd.Timedelta(days=train_days, minutes=-5))
                self.assertEqual(folds[-1].origin, ts[-1] - pd.Timedelta(days=1))
                for i, fold in enumerate(folds):
                    self.assertEqual(fold.origin, folds[0].origin + pd.Timedelta(days=i))
                    self.assertEqual(pd.Timestamp(fold.metadata['raw_history_start']),
                                     ts[0] + pd.Timedelta(days=i))
                    self.assertEqual(fold.metadata['train_history_steps'], train_days * 288)

    def test_group_projection_and_template_feature_contract(self):
        for path in self.paths():
            group, version, route, building, filename = path.relative_to(DIRECTORY).parts
            with self.subTest(path=path):
                config = load_yaml_config(path)
                baseline = load_yaml_config(DIRECTORY / 'baseline' / version / route / building / filename)
                payload = deepcopy(config.canonical_payload())
                reference = baseline.canonical_payload()
                sources = {s.name: s for s in config.data.sources}
                self.assertIn('chinese_holiday', sources)
                self.assertEqual([c.name for c in sources['chinese_holiday'].columns],
                                 ['is_holiday', 'next_holiday_days'])
                expected = {'target_history', 'chinese_holiday'}
                added = []
                if group == 'add_weather':
                    expected.add('weather')
                    self.assertEqual(len(sources['weather'].inference_columns), 6)
                if group in ('add_endogenous_it', 'add_endogenous_route'):
                    expected.add('covariate_history')
                    added = ['it_total_load'] if group == 'add_endogenous_it' else [
                        'hvac_total_load_' + ('B' if route == 'route_A' else 'A')]
                    if group == 'add_endogenous_route' and building == 'ALL':
                        added.append('hvac_total_load_AB')
                    s = sources['covariate_history']
                    self.assertEqual([c.name for c in s.columns], added)
                    self.assertEqual({c.role for c in s.columns}, {'observed_past'})
                    self.assertEqual(s.provider, 'persistence')
                self.assertEqual(set(sources), expected)
                self.assertEqual(dict(config.features.observed_past_lags), {c: (288, 576) for c in added})
                target = config.problem.targets[0]
                self.assertEqual(tuple(config.features.target_lags[target]),
                                 tuple(range(288, (8 if building in ('A1', 'A2') else 4) * 288, 288)))
                template_features = load_yaml_config(TEMPLATES / filename).canonical_payload()['features']
                template_features['target_lags'] = {target: list(config.features.target_lags[target])}
                advanced = template_features['transformations']['advanced']
                for kind in ('rolling', 'expanding'):
                    advanced[kind]['columns'] = [target]
                if building in ('A3', 'ALL'):
                    advanced['rolling']['windows'] = [288, 576]
                own_features = deepcopy(payload['features'])
                own_features['observed_past_lags'] = {}
                self.assertEqual(own_features, template_features)
                payload['data']['sources'] = [s for s in payload['data']['sources']
                                               if s['name'] in ('target_history', 'chinese_holiday')]
                payload['features']['observed_past_lags'] = {}
                # IT 文件使用独立窗口；只允许数据路径、validation 和 output 不同。
                if group == 'add_endogenous_it':
                    payload['data']['sources'][0]['history_path'] = reference['data']['sources'][0]['history_path']
                    payload['validation'] = reference['validation']
                payload['output'] = reference['output']
                self.assertEqual(payload, reference)

    def test_real_designs_and_covariate_visibility(self):
        # 每份配置完整编译一个训练 origin；预测直接编译首/中/末步，不拟合 estimator。
        for path in self.paths():
            with self.subTest(path=path):
                config = load_yaml_config(path)
                end = pd.Timestamp(config.validation['forecast_origin'])
                origin = end - pd.Timedelta(days=1)
                start = origin - pd.Timedelta(minutes=5 * (config.validation['train_history_steps'] - 1))
                builder = builder_for(config, start)
                designs, labels = builder.training_row(origin - pd.Timedelta(days=1))
                self.assertTrue(designs)
                self.assertTrue(all(np.isfinite(d).all() for d in designs))
                self.assertTrue(np.isfinite(labels).all())
                request = builder.request(origin)
                steps = (1, 144, 288)
                compiled = builder.compiler.compile(builder.registry.materialize(request), request, horizon_steps=steps)
                self.assertTrue(np.isfinite(compiled.frame[list(compiled.schema.feature_names)].to_numpy(dtype=float)).all())
                covariates = [s for s in config.data.sources if s.name == 'covariate_history']
                if covariates:
                    source = covariates[0]
                    values = pd.read_csv(ROOT / source.history_path, parse_dates=['time']).set_index('time')
                    proofs = [p for p in compiled.visibility_proof if p.source_name == source.name]
                    self.assertEqual(len(proofs), len(source.columns) * 2 * len(steps))
                    for proof in proofs:
                        self.assertLessEqual(proof.source_time, origin)
                        self.assertLessEqual(proof.available_at, origin)
                        self.assertGreaterEqual(proof.source_time, start)
                        self.assertIsNone(proof.provider)
                        column = proof.feature_name.rsplit('__lag_', 1)[0]
                        self.assertAlmostEqual(compiled.frame.iloc[steps.index(proof.horizon_step)][proof.feature_name],
                                               values.loc[proof.source_time, column])

    def test_covariate_future_and_outside_window_perturbation(self):
        for group in ('add_endogenous_it', 'add_endogenous_route'):
            for variant in VARIANTS:
                path = DIRECTORY / group / 'hvac_all_devices/route_A/ALL' / f'lgbm_{variant}.yaml'
                config = load_yaml_config(path)
                origin = pd.Timestamp(config.validation['forecast_origin']) - pd.Timedelta(days=1)
                start = origin - pd.Timedelta(minutes=5 * (config.validation['train_history_steps'] - 1))
                builder = builder_for(config, start)
                request = builder.request(origin)
                before = builder.compiler.compile(builder.registry.materialize(request), request,
                                                  horizon_steps=(1, 144, 288))
                with TemporaryDirectory() as temp:
                    sources = []
                    for source in config.data.sources:
                        if source.name != 'covariate_history':
                            sources.append(source)
                            continue
                        frame = pd.read_csv(ROOT / source.history_path, parse_dates=['time'])
                        frame.loc[(frame.time < start) | (frame.time > origin),
                                  [c.name for c in source.columns]] += 1000000
                        changed_path = Path(temp) / 'mutated.csv'
                        frame.to_csv(changed_path, index=False)
                        sources.append(replace(source, history_path=str(changed_path)))
                    changed_config = replace(config, data=replace(config.data, sources=tuple(sources)))
                    changed = builder_for(changed_config, start)
                    after = changed.compiler.compile(changed.registry.materialize(request), request,
                                                     horizon_steps=(1, 144, 288))
                    pd.testing.assert_frame_equal(before.frame, after.frame, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
