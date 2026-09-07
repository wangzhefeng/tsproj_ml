"""风向原生窗口与日统计按圆周计算，差值保留有符号角差。"""
from dataclasses import replace
import unittest

import pandas as pd

from data_loading.weather_generator.derivation import derive_native
from data_loading.weather_generator.resampling import resampled_value
from forecasting_core.specs.weather import WeatherGenerationSpec, WeatherNativeFeatureSpec, WeatherVariableSpec
from test_weather_spec import weather_options


class WeatherDirectionTest(unittest.TestCase):
    def test_rolling_mean_difference_and_ambiguous_direction(self):
        times = pd.date_range('2026-01-01', periods=2, freq='1h', tz='UTC')
        frame = pd.DataFrame({'time': times, 'variable': 'direction', 'value': [350., 10.], 'available_at': times})
        meta = {'direction': {'name':'direction','column':'direction','unit':'degree','semantics':'point','native_freq':'1h','label':'left'}}
        features = [WeatherNativeFeatureSpec('mean', ('direction',), 'rolling_mean', '2h'), WeatherNativeFeatureSpec('change', ('direction',), 'difference', '1h')]
        result, desc = derive_native(frame, meta, features)
        self.assertAlmostEqual(result.loc[result.variable.eq('mean'), 'value'].iloc[0], 0.)
        self.assertEqual(result.loc[result.variable.eq('change'), 'value'].tolist(), [20.])
        self.assertEqual(desc['change']['unit'], 'delta_degree')
        self.assertEqual(result.loc[result.variable.eq('mean'), 'available_at'].tolist(), [times[-1]])
        with self.assertRaises(ValueError):
            derive_native(frame.assign(value=[0., 180.]), meta, features)

    def test_daily_circular_mean_rejects_scalar_extrema_and_zero_resultant(self):
        times = pd.date_range('2026-01-01', periods=24, freq='1h', tz='UTC')
        frame = pd.DataFrame({'time': times, 'value': [350., 10.] * 12, 'available_at': times})
        meta = {'unit':'degree','semantics':'point','native_freq':'1h','label':'left'}
        temporal = replace(WeatherGenerationSpec.from_mapping(weather_options()).temporal, freq='1D', timezone='UTC')
        variable = WeatherVariableSpec('direction', 'direction', 'degree', 'mean')
        value, available = resampled_value(frame, times[0], meta, temporal, variable)
        self.assertAlmostEqual(value, 0.)
        self.assertEqual(available, times[-1])
        for aggregation in ('min', 'max'):
            with self.assertRaises(ValueError):
                resampled_value(frame, times[0], meta, temporal, replace(variable, aggregation=aggregation))
        with self.assertRaises(ValueError):
            resampled_value(frame.assign(value=[0., 180.] * 12), times[0], meta, temporal, variable)
