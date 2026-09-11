"""联通优化公共通路；全部使用临时合成数据，不运行正式配置。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from data_loading import SourceRegistry
from feature_engineering.compiler import FeatureCompiler
from model_pipeline.supervised_design import SupervisedDesignBuilder, minimum_history_rows

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT'


def synthetic_config(root, variant='mimo', weather=False):
    path = DIRECTORY / f'lgbm_{variant}.yaml'
    if not path.exists():
        path = DIRECTORY / 'add_weather' / path.name
    config = load_yaml_config(path)
    times = pd.date_range('2026-08-01', periods=72, freq='1h')
    pd.DataFrame({'time': times, 'value': np.arange(72, dtype=float)}).to_csv(root / 'target.csv', index=False)
    source = replace(config.data.sources[0], history_path=str(root / 'target.csv'))
    sources = [source]
    if weather:
        original = next(s for s in config.data.sources if s.name == 'weather')
        frame = pd.DataFrame({'ts': times})
        for measured, forecast in dict(original.inference_columns).items():
            frame[measured] = np.arange(72, dtype=float)
            frame[forecast] = np.arange(72, dtype=float) + 100
        frame.to_csv(root / 'weather.csv', index=False)
        sources.append(replace(original, history_path=str(root / 'weather.csv'), inference_columns=dict(original.inference_columns)))
    transforms = {'advanced': {
        'same_slot': {'columns': ['value'], 'period': 12, 'days': [2, 3], 'stats': ['mean', 'std']},
        'recent_state': {'columns': ['value'], 'windows': [3, 6], 'stats': ['level', 'std', 'diff', 'slope']},
    }}
    direct = config.features.canonical_payload()['transformations'].get('direct')
    if direct:
        transforms['direct'] = direct
    return replace(config,
        problem=replace(config.problem, freq='1h', horizon=4),
        data=replace(config.data, sources=tuple(sources)),
        features=replace(config.features, target_lags={}, datetime_features=[], transformations=transforms),
        strategy=replace(config.strategy, output_chunk_length=2) if config.strategy.output_chunk_length else config.strategy,
        estimator=replace(config.estimator, params={'n_estimators': 3, 'num_leaves': 3, 'min_child_samples': 2, 'verbosity': -1}),
        validation={'forecast_origin': str(times[-1]), 'schedule_mode': 'intraday',
                    'history_steps': 60, 'train_window_steps': 9, 'fold_count': 2, 'stride_steps': 4,
                    'performance': {'total_thread_limit': 1}}), times


class OptimizationCompilerTest(unittest.TestCase):
    def test_cyclical_consumes_causal_features_in_single_and_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, times = synthetic_config(root, weather=True)
            transforms = config.features.canonical_payload()['transformations']
            transforms['advanced']['block_weather'] = {'columns': ['rt_tt2'], 'stats': ['mean']}
            columns = ['value_slot_mean_3d', 'value_rs_level_3', 'rt_tt2_blk_mean']
            transforms['advanced']['cyclical'] = {'columns': columns, 'period': 24}
            config = replace(config, features=replace(config.features, transformations=transforms))
            builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
            request = builder.request(times[47])
            information = builder.registry.materialize(request)
            single = builder.compiler.compile(information, request)
            batch = builder.compiler.compile_batch((information,), (request,))[0]
            pd.testing.assert_frame_equal(single.frame, batch.frame)
            self.assertEqual(single.schema, batch.schema)
            # 独立黄金值：同槽均值 24..27、原点水平 47、预报块均值 149.5。
            for column, values in zip(columns, (np.arange(24, 28), np.full(4, 47), np.full(4, 149.5))):
                np.testing.assert_allclose(single.frame[f'{column}_sin'], np.sin(2 * np.pi * values / 24))
                np.testing.assert_allclose(single.frame[f'{column}_cos'], np.cos(2 * np.pi * values / 24))

    def test_cache_identity_causal_parameters_and_implementation(self):
        from feature_engineering.cache import compute_raw_design_fingerprint, raw_design_provenance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, times = synthetic_config(root)
            arguments = dict(base_dir=root, origin=times[67], generators={})
            config = replace(config, validation={**dict(config.validation), 'train_history_steps': 48})
            before = compute_raw_design_fingerprint(config, **arguments)
            transforms = config.features.canonical_payload()['transformations']
            transforms['advanced']['recent_state']['windows'] = [3, 6, 12]
            updated = replace(config, features=replace(config.features, transformations=transforms))
            self.assertNotEqual(before, compute_raw_design_fingerprint(updated, **arguments))
            transforms['seasonal_baseline'] = {'column': 'value', 'period': 12, 'days': 3}
            residual = replace(config, features=replace(config.features, transformations=transforms))
            self.assertNotEqual(compute_raw_design_fingerprint(updated, **arguments), compute_raw_design_fingerprint(residual, **arguments))
            self.assertIn('feature_engineering/seasonal.py', raw_design_provenance(config, **arguments)['compilation_implementation_hashes'])

    def test_block_weather_compilation_work_is_linear(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('mimo', 'recmo', 'dirmo', 'dirrecmo', 'direct'):
                config, _ = synthetic_config(root, variant, weather=True)
                times = pd.date_range('2026-08-01', periods=600, freq='1h')
                pd.DataFrame({'time': times, 'value': np.arange(600)}).to_csv(root / 'target.csv', index=False)
                weather_source = next(source for source in config.data.sources if source.name == 'weather')
                weather = pd.DataFrame({'ts': times})
                for measured, forecast in dict(weather_source.inference_columns).items():
                    weather[measured] = np.arange(600)
                    weather[forecast] = np.arange(600) + 100
                weather.to_csv(root / 'weather.csv', index=False)
                transforms = {'advanced': {'block_weather': {'columns': ['rt_tt2'], 'stats': ['mean', 'min', 'max']}}}
                for horizon in (4, 12, 288):
                    with self.subTest(variant=variant, horizon=horizon):
                        current = replace(config, problem=replace(config.problem, horizon=horizon),
                                          features=replace(config.features, transformations=transforms))
                        builder = SupervisedDesignBuilder(current, SourceRegistry(current.data, root))
                        request = builder.request(times[300])
                        information = builder.registry.materialize(request)
                        compiler = builder.compiler
                        with patch.object(compiler, '_compile_known_future', wraps=compiler._compile_known_future) as calls:
                            single = compiler.compile(information, request)
                            # 每步基础列一次、块摘要取数至多一次；不能按块宽重复取整块。
                            self.assertLessEqual(calls.call_count, 2 * horizon)
                        with patch.object(compiler, '_compile_known_future', wraps=compiler._compile_known_future) as calls:
                            batch = compiler.compile_batch((information,), (request,))[0]
                            self.assertLessEqual(calls.call_count, 2 * horizon)
                        pd.testing.assert_frame_equal(single.frame, batch.frame)
                        self.assertEqual(single.schema, batch.schema)
                        width = current.strategy.resolve(horizon).steps_per_call
                        blocks = np.arange(401, 401 + horizon).reshape(-1, width)
                        for stat in ('mean', 'min', 'max'):
                            expected = np.repeat(getattr(np, stat)(blocks, axis=1), width)
                            np.testing.assert_allclose(single.frame[f'rt_tt2_blk_{stat}'], expected)

    def test_block_weather_cache_isolates_information_sets_and_series(self):
        from forecasting_core.specs import ColumnSpec
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, times = synthetic_config(root, weather=True)
            for filename in ('target.csv', 'weather.csv'):
                frame = pd.read_csv(root / filename)
                other = frame.copy()
                numeric = other.select_dtypes('number').columns
                other[numeric] += 1000
                pd.concat([frame.assign(site='a'), other.assign(site='b')]).to_csv(root / filename, index=False)
            config = replace(config,
                             problem=replace(config.problem, training_scope='global', series_id_cols=('site',)),
                             data=replace(config.data, sources=tuple(replace(source, series_id_cols=('site',),
                                                                           columns=(*source.columns, ColumnSpec('site', 'key')),
                                                                           inference_columns=dict(source.inference_columns) if source.inference_columns else None)
                                                                    for source in config.data.sources)),
                             features=replace(config.features, transformations={'advanced': {
                                 'block_weather': {'columns': ['rt_tt2'], 'stats': ['mean', 'min', 'max']}}}))
            builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
            requests, information_sets, expected_means = [], [], []
            # 同原点实测/预报、后续原点、再回到原原点，均复用同一个 compiler。
            for index, actual in ((47, True), (47, False), (51, False), (47, True)):
                requests.append(builder.request(times[index]))
                information_sets.append(builder.registry.materialize(builder.request(
                    times[index], target_access='supervised_labels' if actual else 'history_only')))
                expected_means.append(index + 2.5 + (0 if actual else 100))
            single = [builder.compiler.compile(info, request) for info, request in zip(information_sets, requests)]
            batch = builder.compiler.compile_batch(tuple(information_sets), tuple(requests))
            for left, right, expected in zip(single, batch, expected_means):
                pd.testing.assert_frame_equal(left.frame, right.frame)
                for site, shift in (('a', 0), ('b', 1000)):
                    np.testing.assert_allclose(left.frame.loc[left.frame['site'] == site, 'rt_tt2_blk_mean'], expected + shift)
                left_proofs = [proof for proof in left.visibility_proof if proof.source_name == 'block_known_future']
                right_proofs = [proof for proof in right.visibility_proof if proof.source_name == 'block_known_future']
                key = lambda proof: (proof.feature_name, proof.horizon_step)
                self.assertEqual(sorted(left_proofs, key=key), sorted(right_proofs, key=key))
            # 只请求块中一步仍必须使用完整块，而不是对筛选后的行求均值。
            subset = builder.compiler.compile(information_sets[1], requests[1], horizon_steps=(2,))
            np.testing.assert_allclose(subset.frame['rt_tt2_blk_mean'], [149.5, 1149.5])

    def test_synthetic_backtest_lifecycle_ets_and_optimized_mimo(self):
        from model_pipeline.runner import run_canonical_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for native in (False, True):
                config, times = synthetic_config(root, weather=not native)
                transforms = config.features.canonical_payload()['transformations']
                transforms['seasonal_baseline'] = {'column': 'value', 'period': 12, 'days': 3}
                if not native:
                    transforms['advanced']['block_weather'] = {'columns': ['rt_tt2'], 'stats': ['mean', 'min', 'max']}
                config = replace(config,
                    features=replace(config.features, transformations=transforms),
                    validation={**dict(config.validation), 'forecast_origin': str(times[-1]),
                                'history_steps': 90, 'fold_count': 2, 'train_history_steps': 48})
                if native:
                    config = replace(config,
                        features=replace(config.features, target_lags={}, datetime_features=(), transformations={}),
                        estimator=replace(config.estimator, model_type='ets', params={'seasonal_periods': 12, 'candidates': ['ANN', 'ANA']}))
                config = replace(config, validation={**dict(config.validation),
                    'train_window_steps': 48 - minimum_history_rows(config) - 4 + 1})
                result = run_canonical_config(config, output_root=root / ('ets' if native else 'optimized'), backtest_only=True)
                scores = pd.read_csv(result.test_dir / 'test_scores_df.csv')
                self.assertFalse(scores.empty)
                self.assertTrue(np.isfinite(pd.read_csv(result.test_dir / 'cv_plot_df.csv').select_dtypes('number').to_numpy()).all())
                self.assertEqual(list(result.test_dir.parent.rglob('model.pkl')), [])

    def test_spec_rejects_malformed_causal_parameters_before_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = synthetic_config(Path(directory))
            for malformed in ({'same_slot': {'columns': ['value'], 'period': True, 'days': [3], 'stats': ['mean']}},
                              {'recent_state': {'columns': ['value'], 'windows': [1], 'stats': ['slope']}},
                              {'block_weather': {'columns': ['rt_tt2'], 'stats': ['mean'], 'typo': 1}}):
                with self.subTest(malformed=malformed), self.assertRaises((TypeError, ValueError)):
                    replace(config.features, transformations={'advanced': malformed})

    def test_residual_provider_raw_units_and_target_isolation(self):
        from forecasting_core.specs import ColumnSpec
        from model_training.strategies import TargetCoordinate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, times = synthetic_config(root, 'recursive')
            transforms = config.features.canonical_payload()['transformations']
            transforms['seasonal_baseline'] = {'column': 'value', 'period': 12, 'days': 3}
            source = config.data.sources[0]
            source = replace(source, columns=(*source.columns, ColumnSpec('other', 'target')))
            pd.DataFrame({'time': times, 'value': np.arange(72), 'other': np.arange(72) * 10}).to_csv(root / 'target.csv', index=False)
            config = replace(config, problem=replace(config.problem, targets=('value', 'other')),
                             data=replace(config.data, sources=(source,)),
                             features=replace(config.features, target_lags={'value': [1], 'other': [1]}, transformations=transforms),
                             validation={**dict(config.validation), 'train_history_steps': 48})
            builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root), history_start=times[20])
            baseline = builder.seasonal_baseline(times[67])
            np.testing.assert_allclose(baseline[0, :, 0], np.arange(44, 48))
            np.testing.assert_array_equal(baseline[0, :, 1], 0)
            _, provider = builder.forecast_designs(times[67])
            prior = {TargetCoordinate('value', 1): np.array([24.]), TargetCoordinate('other', 1): np.array([680.])}
            row = provider(1, builder.plan.call_coordinates[1], builder.plan.dependencies[1], prior)
            self.assertEqual(row[0, builder.feature_schema.index('value__lag_1')], 68)
            self.assertEqual(row[0, builder.feature_schema.index('other__lag_1')], 680)

    def test_block_weather_actual_forecast_schema_and_last_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('mimo', 'recmo', 'dirmo', 'dirrecmo', 'direct'):
                config, times = synthetic_config(root, variant, weather=True)
                transforms = config.features.canonical_payload()['transformations']
                transforms['advanced']['block_weather'] = {'columns': ['rt_tt2'], 'stats': ['mean', 'min', 'max']}
                config = replace(config, features=replace(config.features, transformations=transforms))
                builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
                request = builder.request(times[47])
                actual_information = builder.registry.materialize(builder.request(times[47], target_access='supervised_labels'))
                forecast_information = builder.registry.materialize(request)
                actual = builder.compiler.compile(actual_information, request)
                forecast = builder.compiler.compile(forecast_information, request)
                batch = builder.compiler.compile_batch((forecast_information,), (request,))[0]
                pd.testing.assert_frame_equal(forecast.frame, batch.frame)
                expected = [49.5] * 4 if variant == 'mimo' else ([48.5, 48.5, 50.5, 50.5]
                           if variant != 'direct' else [48, 49, 50, 51])
                np.testing.assert_allclose(actual.frame['rt_tt2_blk_mean'], expected)
                np.testing.assert_allclose(forecast.frame['rt_tt2_blk_mean'], np.array(expected) + 100)
                self.assertEqual(actual.schema, forecast.schema)

    def test_ets_runner_uses_complete_raw_history_not_supervised_labels(self):
        from unittest.mock import patch
        from model_pipeline.runner import CanonicalBaseModelRunner
        from model_training.trainer import CanonicalTrainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, times = synthetic_config(root)
            config = replace(config, features=replace(config.features, transformations={}),
                             estimator=replace(config.estimator, model_type='ets',
                                               params={'seasonal_periods': 12, 'candidates': ['ANN']}),
                             validation={**dict(config.validation), 'train_history_steps': 48, 'train_window_steps': 44})
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, root), times[67])
            with patch.object(CanonicalTrainer, 'train', side_effect=AssertionError('ETS cannot train from Y')):
                scaler, transform, _, _, model = runner.fit(tuple(range(44)))
            evidence = model.execution_evidence()
            self.assertEqual(evidence['history_count'], 48)
            self.assertEqual(evidence['history_start'], times[20].isoformat())
            self.assertEqual(evidence['history_end'], times[67].isoformat())
            designs, provider = runner.forecast_designs(times[67], scaler, transform)
            prediction = runner.predict(model, designs, provider, times[68:72], transform)
            self.assertEqual(prediction.values.shape, (1, 4, 1))
            self.assertEqual(runner.execution_evidence(model, transform)['status'], 'recorded')
            with self.assertRaises(ValueError):
                runner.final_bundle_inputs()

    def test_nine_residual_strategies_fit_and_restore(self):
        from model_pipeline.runner import CanonicalBaseModelRunner
        variants = ('direct-pointwise', 'direct-pointwise-horizon', 'direct', 'recursive',
                    'dirrec', 'dirmo', 'recmo', 'dirrecmo', 'mimo')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant, weather in ((v, w) for v in variants for w in (False, True)):
                with self.subTest(variant=variant, weather=weather):
                    config, times = synthetic_config(root, variant, weather=weather)
                    transforms = config.features.canonical_payload()['transformations']
                    transforms['seasonal_baseline'] = {'column': 'value', 'period': 12, 'days': 3}
                    if weather and variant in ('mimo', 'dirmo', 'recmo', 'dirrecmo'):
                        transforms['advanced']['block_weather'] = {'columns': ['rt_tt2'], 'stats': ['mean', 'min', 'max']}
                    config = replace(config, features=replace(config.features, transformations=transforms),
                                     validation={**dict(config.validation), 'train_history_steps': 48})
                    runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, root), times[67])
                    scaler, transform, X, residual, artifact = runner.fit(tuple(range(9)))
                    # y(t+h) - mean(y(t+h-12), y(t+h-24), y(t+h-36)) = 24.
                    np.testing.assert_allclose(residual, 24)
                    designs, provider = runner.forecast_designs(times[67], scaler, transform)
                    result = runner.predict(artifact, designs, provider, times[68:72], transform)
                    np.testing.assert_allclose(result.values[0, :, 0], np.arange(68, 72), atol=1e-5)

    def test_single_batch_gold_origin_anchor_and_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, times = synthetic_config(root)
            builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
            request = builder.request(times[47])
            information = builder.registry.materialize(request)
            compiler = FeatureCompiler(config)
            single = compiler.compile(information, request)
            batch = compiler.compile_batch((information,), (request,))[0]
            pd.testing.assert_frame_equal(single.frame, batch.frame)
            frame = single.frame
            np.testing.assert_allclose(frame['value_slot_mean_3d'], [24, 25, 26, 27])
            np.testing.assert_allclose(frame['value_rs_level_6'], 47)
            np.testing.assert_allclose(frame['value_rs_diff_6'], 5)
            np.testing.assert_allclose(frame['value_rs_slope_6'], 1)
            self.assertEqual(minimum_history_rows(config), 36)
            changed = pd.read_csv(root / 'target.csv')
            # 文件源合同拒绝全文件缺值；用有限巨量扰动验证因果不变性。
            changed.loc[48:, 'value'] += 1000000
            changed.to_csv(root / 'target.csv', index=False)
            changed_builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
            changed_information = changed_builder.registry.materialize(request)
            pd.testing.assert_frame_equal(single.frame, changed_builder.compiler.compile(changed_information, request).frame)
            short = builder.request(times[20])
            with self.assertRaises(ValueError):
                compiler.compile(builder.registry.materialize(short), short)


if __name__ == '__main__':
    unittest.main()
