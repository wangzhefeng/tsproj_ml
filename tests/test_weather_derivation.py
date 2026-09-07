"""小时原生特征与单位计算的独立数值参照。"""
import unittest
import pandas as pd

from forecasting_core.specs.weather import WeatherNativeFeatureSpec


class WeatherDerivationTest(unittest.TestCase):
    def test_temperature_difference_uses_delta_units_without_offset(self):
        from data_loading.weather_generator.derivation import derive_native
        from data_loading.weather_generator.resampling import convert_unit
        times = pd.date_range('2026-01-01',periods=2,freq='1h',tz='UTC')
        frame = pd.DataFrame({'time':times,'variable':'temp','value':[273.15,276.15],'available_at':times})
        metadata = {'temp':{'name':'temp','column':'temp','unit':'K','semantics':'point','native_freq':'1h','label':'left'}}
        result, desc = derive_native(frame,metadata,[WeatherNativeFeatureSpec('diff',('temp',),'difference','1h')])
        self.assertEqual(desc['diff']['unit'],'delta_K')
        value = result.loc[result.variable.eq('diff'),'value'].iloc[0]
        self.assertAlmostEqual(convert_unit(value,desc['diff']['unit'],'delta_degC'),3.)
        with self.assertRaises(ValueError):
            convert_unit(value,desc['diff']['unit'],'degC')

    def test_hourly_rolling_and_difference_preserve_full_warmup(self):
        from data_loading.weather_generator.derivation import derive_native
        times = pd.date_range('2026-01-01', periods=4, freq='1h', tz='UTC')
        frame = pd.DataFrame({'time': times, 'variable': 'temperature', 'value': [0., 3., 6., 9.], 'available_at': times})
        metadata = {'temperature': {'name':'temperature','column':'temperature','unit':'degC','semantics':'point','native_freq':'1h','label':'left'}}
        features = [WeatherNativeFeatureSpec('mean3', ('temperature',), 'rolling_mean', '3h'), WeatherNativeFeatureSpec('diff1', ('temperature',), 'difference', '1h')]
        result, desc = derive_native(frame, metadata, features)
        self.assertEqual(result.loc[result.variable.eq('mean3'), 'value'].tolist(), [3., 6.])
        self.assertEqual(result.loc[result.variable.eq('diff1'), 'value'].tolist(), [3., 3., 3.])
        self.assertEqual(desc['mean3']['native_freq'], '1h')
        self.assertEqual(result.loc[result.variable.eq('mean3'), 'available_at'].tolist(), list(times[2:]))

    def test_relative_humidity_kelvin_and_impossible_dewpoint(self):
        from data_loading.weather_generator.derivation import derive_native
        time = pd.Timestamp('2026-01-01', tz='UTC')
        frame = pd.DataFrame({'time': [time,time], 'variable':['temp','dew'], 'value':[293.15,293.15], 'available_at':[time,time]})
        desc = {name:{'name':name,'column':name,'unit':'K','semantics':'point','native_freq':'1h','label':'left'} for name in ['temp','dew']}
        feature = WeatherNativeFeatureSpec('rh', ('temp','dew'), 'relative_humidity', None)
        result, _ = derive_native(frame, desc, [feature])
        self.assertAlmostEqual(result.loc[result.variable.eq('rh'), 'value'].iloc[0], 100.)
        frame.loc[1,'value'] += 2
        with self.assertRaises(ValueError):
            derive_native(frame, desc, [feature])


if __name__ == '__main__':
    unittest.main()
