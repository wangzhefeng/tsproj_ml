"""四组物理 YAML 的永久合同测试；不拟合业务模型。"""
from pathlib import Path
import unittest

import pandas as pd

from config.config_loader import load_yaml_config
from data_loading import SourceRegistry
from feature_engineering.compiler import FeatureCompiler
from model_pipeline.supervised_design import minimum_history_rows, SupervisedDesignBuilder, raw_history_backtest_windows
from models.factory import ModelFactory
from scripts.check_model_configs import check_model_yaml

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT'
VARIANTS = {'direct-pointwise', 'direct-pointwise-horizon', 'direct', 'recursive', 'dirrec', 'dirmo', 'recmo', 'dirrecmo', 'mimo'}


class LiantongFourGroupsTest(unittest.TestCase):
    def test_physical_matrix_parameters_and_all_fold_dates(self):
        paths = list(DIRECTORY.glob('*/*.yaml'))
        self.assertEqual(len(paths), 37)
        self.assertEqual(list(DIRECTORY.glob('*.yaml')), [])
        for group in ('baseline', 'baseline_opt', 'add_weather', 'add_weather_opt'):
            files = set(p.name for p in (DIRECTORY / group).glob('*.yaml'))
            expected = {f'lgbm_{variant}.yaml' for variant in VARIANTS}
            if group == 'baseline':
                expected.add('ets.yaml')
            self.assertEqual(files, expected)
        for path in sorted(paths):
            with self.subTest(path=str(path.relative_to(DIRECTORY))):
                config = load_yaml_config(path)
                _, errors = check_model_yaml(str(path))
                self.assertEqual(errors, [])
                FeatureCompiler(config)
                ModelFactory().create_model(config.estimator.model_type, dict(config.estimator.params), log_params=False)
                self.assertEqual((config.problem.freq, config.problem.horizon), ('5min', 288))
                self.assertEqual(config.validation['train_history_steps'], 4032)
                self.assertEqual(config.validation['train_window_steps'], 4032 - minimum_history_rows(config) - 288 + 1)
                self.assertTrue(config.output['scenario_subpath'].endswith('/' + path.parent.name))
                source_names = {s.name for s in config.data.sources}
                self.assertEqual('weather' in source_names, path.parent.name.startswith('add_weather'))
                self.assertNotIn('target', config.features.transformations)
                if path.parent.name.endswith('_opt'):
                    self.assertEqual(config.features.transformations['seasonal_baseline']['days'], 7)
                    self.assertEqual(minimum_history_rows(config), 2016)
                folds = raw_history_backtest_windows(SupervisedDesignBuilder(config, SourceRegistry(config.data, ROOT)),
                                                    pd.Timestamp(config.validation['forecast_origin']))
                self.assertEqual(len(folds), 17)
                self.assertEqual(folds[0].origin, pd.Timestamp('2026-08-14 23:55'))
                self.assertEqual(folds[-1].origin, pd.Timestamp('2026-08-30 23:55'))
                for fold in folds:
                    self.assertEqual(pd.Timestamp(fold.metadata['raw_history_end']) - pd.Timestamp(fold.metadata['raw_history_start']),
                                     pd.Timedelta(days=14) - pd.Timedelta(minutes=5))

    def test_unoptimized_pair_differs_only_by_weather_and_output(self):
        for variant in VARIANTS:
            baseline = load_yaml_config(DIRECTORY / f'baseline/lgbm_{variant}.yaml').canonical_payload()
            weather = load_yaml_config(DIRECTORY / f'add_weather/lgbm_{variant}.yaml').canonical_payload()
            weather['data']['sources'] = [s for s in weather['data']['sources'] if s['name'] != 'weather']
            baseline.pop('output')
            weather.pop('output')
            self.assertEqual(baseline, weather)


if __name__ == '__main__':
    unittest.main()
