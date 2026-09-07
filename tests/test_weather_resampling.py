"""点/区间物理量与完整自然日/月窗口，合成独立数值测试。"""
import unittest
import pandas as pd
from forecasting_core.specs.weather import WeatherTemporalSpec, WeatherVariableSpec


class WeatherResamplingTest(unittest.TestCase):
    def test_wind_direction_interpolates_across_north_not_south(self):
        from data_loading.weather_generator.resampling import resampled_value
        times = pd.date_range('2026-01-01',periods=2,freq='1h',tz='UTC')
        part = pd.DataFrame({'time':times,'value':[350.,10.],'available_at':times[0]})
        meta = {'unit':'degree','semantics':'point','native_freq':'1h','label':'left'}
        temporal = WeatherTemporalSpec('UTC','15min','left','left','linear','1h')
        variable = WeatherVariableSpec('wind','wind','degree','point')
        value, _ = resampled_value(part,times[0]+pd.Timedelta('30min'),meta,temporal,variable)
        self.assertAlmostEqual(value,0.)

    def test_dst_calendar_day_uses_23_actual_hours(self):
        from data_loading.weather_generator.resampling import resampled_value
        start = pd.Timestamp('2026-03-08',tz='America/New_York')
        end = start + pd.DateOffset(days=1)
        times = pd.date_range(start,end,inclusive='right',freq='1h').tz_convert('UTC')
        self.assertEqual(len(times),23)
        part = pd.DataFrame({'time':times,'value':100.,'available_at':pd.Timestamp('2026-03-07',tz='UTC')})
        meta = {'unit':'W/m2','semantics':'interval_mean','native_freq':'1h','label':'right'}
        temporal = WeatherTemporalSpec('America/New_York','1D','left','left','exact',None)
        variable = WeatherVariableSpec('solar','solar','MJ/m2','integral')
        value, _ = resampled_value(part,start.tz_convert('UTC'),meta,temporal,variable)
        self.assertAlmostEqual(value,8.28)

    def test_radiation_integral_requires_complete_day(self):
        from data_loading.weather_generator.resampling import resampled_value
        times = pd.date_range('2026-01-01T01:00Z', periods=24, freq='1h')
        part = pd.DataFrame({'time': times, 'value': 100., 'available_at': pd.Timestamp('2025-12-31', tz='UTC')})
        meta = {'unit':'W/m2','semantics':'interval_mean','native_freq':'1h','label':'right'}
        temporal = WeatherTemporalSpec('UTC','1D','left','left','exact',None)
        variable = WeatherVariableSpec('solar','shortwave_radiation','MJ/m2','integral')
        value, available = resampled_value(part, pd.Timestamp('2026-01-01', tz='UTC'), meta, temporal, variable)
        self.assertAlmostEqual(value, 8.64)
        self.assertEqual(available, pd.Timestamp('2025-12-31', tz='UTC'))
        with self.assertRaises(ValueError):
            resampled_value(part.iloc[:-1], pd.Timestamp('2026-01-01', tz='UTC'), meta, temporal, variable)

    def test_month_end_label_means_full_month_not_one_midnight(self):
        from data_loading.weather_generator.resampling import resampled_value
        times = pd.date_range('2026-02-01', '2026-03-01', inclusive='left', freq='1h', tz='UTC')
        part = pd.DataFrame({'time':times,'value':0.,'available_at':pd.Timestamp('2026-01-01',tz='UTC')})
        meta = {'unit':'degC','semantics':'point','native_freq':'1h','label':'left'}
        variable = WeatherVariableSpec('temp','temp','degC','mean')
        temporal = WeatherTemporalSpec('UTC','1ME','right','left','exact',None)
        value, _ = resampled_value(part, pd.Timestamp('2026-02-28',tz='UTC'), meta, temporal, variable)
        self.assertEqual(value, 0.)
        with self.assertRaises(ValueError):
            resampled_value(part.iloc[:1], pd.Timestamp('2026-02-28',tz='UTC'), meta, temporal, variable)

    def test_interval_precipitation_is_distributed_not_held_as_total(self):
        from data_loading.weather_generator.resampling import resampled_value
        part = pd.DataFrame({'time':[pd.Timestamp('2026-01-01T01:00Z')],'value':[4.],'available_at':[pd.Timestamp('2026-01-01T00:00Z')]})
        meta = {'unit':'mm','semantics':'interval_sum','native_freq':'1h','label':'right'}
        temporal = WeatherTemporalSpec('UTC','15min','left','left','hold','1h')
        variable = WeatherVariableSpec('rain','rain','mm','sum')
        value, _ = resampled_value(part,pd.Timestamp('2026-01-01T00:15Z'),meta,temporal,variable)
        self.assertEqual(value,1.)


if __name__ == '__main__':
    unittest.main()
