"""无递归冷启动、命名节日及训练权重的因果边界。"""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'config/hongtaiyang_cesuan'))

from data_loading.calendar_generator.named_holidays import named_holiday_frame
from model_training.sample_weight import temporal_sample_weight
from cold_start import calendar_baseline, validate_recipe
from annual_backtest import forecast_window, schedule, window_config
from generate_configs import model_document, apply_options
from model_training.trainer import CanonicalTrainer
from forecasting_core.specs.config import parse_model_config


class HongtaiyangOptimizationTest(unittest.TestCase):
    def test_named_holiday_is_not_weekend(self):
        times = pd.to_datetime(['2025-01-19', '2025-01-28', '2025-02-04', '2025-02-05', '2025-05-01'])
        frame = named_holiday_frame(times)
        self.assertEqual(frame.is_spring_festival.tolist(), [0, 1, 1, 0, 0])
        self.assertEqual(frame.is_named_holiday.tolist(), [0, 1, 1, 0, 1])
        self.assertEqual(frame.days_after_spring_festival.iloc[3], 1)
        self.assertTrue((frame.days_after_spring_festival <= 31).all())

    def test_weight_halflife_and_rejections(self):
        origins = pd.to_datetime(['2025-01-01', '2025-01-31'])
        weights = temporal_sample_weight(origins, pd.Timestamp('2025-02-01'),
                                        {'method': 'exponential', 'halflife_days': 30})
        self.assertAlmostEqual(weights[0] / weights[1], 0.5)
        self.assertAlmostEqual(weights.mean(), 1.)
        for spec in ({'method': 'bad', 'halflife_days': 30}, {'method': 'exponential', 'halflife_days': 0}):
            with self.assertRaises(ValueError):
                temporal_sample_weight(origins, pd.Timestamp('2025-02-01'), spec)
        with self.assertRaises(ValueError):
            temporal_sample_weight(origins, pd.Timestamp('2025-01-20'), {'method': 'exponential', 'halflife_days': 30})

    def test_recipe_is_strict(self):
        with self.assertRaises(ValueError):
            validate_recipe({'recipe_version': 1, 'cold_start': {'method': 'recursive'}})
        with self.assertRaises(ValueError):
            validate_recipe({'recipe_version': 1, 'cold_start': {'method': 'calendar_baseline'}, 'unknown': 1})

    def test_cold_start_no_training_no_future_target_access(self):
        times = pd.date_range('2025-01-01', '2026-01-01', freq='1D', inclusive='left')
        actual = pd.DataFrame({'time': times, 'value': 100.})
        actual.loc[actual.time.between('2025-01-28', '2025-01-31'), 'value'] = 10.
        cfg = parse_model_config(model_document('xinnengyuan', 'demand_load', True, 'direct-pointwise'), source='test')
        recipe = {'recipe_version': 1, 'cold_start': {'method': 'calendar_baseline', 'normal_floor_ratio': 0.5, 'transition_days': 0}}
        with patch('model_training.trainer.CanonicalTrainer.train', side_effect=AssertionError('no training for statistical baseline')):
            first, audit = forecast_window(cfg, actual, schedule('1D')[0], recipe=recipe)
            actual.loc[actual.time >= '2025-02-01', 'value'] = 1e9
            second, _ = forecast_window(cfg, actual, schedule('1D')[0], recipe=recipe)
        np.testing.assert_array_equal(first.y_pred, second.y_pred)
        np.testing.assert_array_equal(first.y_pred.iloc[:4], 10.)
        np.testing.assert_array_equal(first.y_pred.iloc[4:], 100.)
        self.assertEqual(audit['model_count'], 0)
        self.assertEqual(audit['effective_strategy'], 'calendar_baseline')

    def test_training_settings_survive_dynamic_window(self):
        doc = model_document('xinnengyuan', 'demand_load', True, 'direct-pointwise')
        doc['validation']['training'] = {'sample_weight': {'method': 'exponential', 'halflife_days': 30}}
        config = parse_model_config(doc, source='test')
        dynamic = window_config(config, 30, 'direct')
        self.assertEqual(dynamic.validation['training']['sample_weight']['halflife_days'], 30)

    def test_real_weighted_named_calendar_training(self):
        times = pd.date_range('2025-01-01', '2026-01-01', freq='1D', inclusive='left')
        frame = pd.DataFrame({'time': times, 'value': 100 + np.arange(len(times)) % 7 * 5.})
        doc = apply_options(model_document('xinnengyuan', 'demand_load', True, 'direct-pointwise'),
                            {'calendar': True, 'halflife_days': 30})
        doc['estimator']['params']['n_estimators'] = 3
        cfg = parse_model_config(doc, source='test')
        original = CanonicalTrainer.train
        captured = []

        def capture(trainer, *args, **kwargs):
            captured.append(kwargs['sample_weight'])
            return original(trainer, *args, **kwargs)

        with patch.object(CanonicalTrainer, 'train', new=capture):
            prediction, audit = forecast_window(cfg, frame, schedule('1D')[1])
        self.assertEqual(len(prediction), 31)
        self.assertEqual(audit['sample_weight']['count'], audit['training_samples'])
        self.assertGreater(captured[0][-1], captured[0][0])
        self.assertIn('is_spring_festival', audit['feature_schema'])

    def test_nonrecursive_cold_pointwise(self):
        times = pd.date_range('2025-01-01', '2026-01-01', freq='1D', inclusive='left')
        frame = pd.DataFrame({'time': times, 'value': 100 + np.arange(len(times)) % 7 * 5.})
        doc = apply_options(model_document('xinnengyuan', 'demand_load', True, 'direct-pointwise'), {'calendar': True})
        doc['estimator']['params']['n_estimators'] = 3
        recipe = {'recipe_version': 1, 'cold_start': {'method': 'calendar_pointwise', 'normal_floor_ratio': .5, 'transition_days': 0}}
        prediction, audit = forecast_window(parse_model_config(doc, source='test'), frame, schedule('1D')[0], recipe=recipe)
        self.assertEqual(len(prediction), 28)
        self.assertEqual(audit['effective_strategy'], 'calendar_pointwise')
        self.assertEqual(audit['model_count'], 1)
        self.assertNotIn('dt_day', audit['feature_schema'])
        self.assertNotIn('dt_day_of_year', audit['feature_schema'])


if __name__ == '__main__':
    unittest.main()
