"""供应商与 Open-Meteo 本地快照适配；测试响应是合成 fixture。"""
from pathlib import Path
import tempfile
import copy
import unittest

import pandas as pd

from test_weather_assets import asset_fixture


class WeatherAdaptersTest(unittest.TestCase):
    def test_historical_json_requires_exact_explicit_row_availability(self):
        from data_loading.weather_generator.adapters import open_meteo_frame
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = asset_fixture(Path(directory), kind='reanalysis')
            meta = manifest['snapshots'][0]
            meta['evidence_class'] = 'historical_release_contract'
            response = {'latitude': 30., 'longitude': 120., 'utc_offset_seconds': 0, 'hourly_units': {'time': 'iso8601', 'temperature_2m': '°C'}, 'hourly': {'time': ['2026-01-01T01:00'], 'temperature_2m': [10.]}}
            release = pd.DataFrame({'time': ['2026-01-01T01:00Z'], 'available_at': ['2026-01-03T01:00Z']})
            result = open_meteo_frame(response, meta, availability=release)
            self.assertEqual(result.available_at.tolist(), [pd.Timestamp('2026-01-03T01:00Z')])
            for bad in (release.iloc[:0], pd.concat([release, release]), release.assign(time='2026-01-02T01:00Z'), release.assign(available_at='2025-12-31T01:00Z')):
                with self.assertRaises(ValueError):
                    open_meteo_frame(response, meta, availability=bad)

    def test_open_meteo_rejects_wrong_grid_coordinates_and_time_encoding(self):
        from data_loading.weather_generator.adapters import open_meteo_frame
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = asset_fixture(Path(directory))
            response = {'latitude': 30., 'longitude': 120., 'utc_offset_seconds': 0, 'hourly_units': {'time': 'iso8601', 'temperature_2m': '°C'}, 'hourly': {'time': ['2026-01-01T01:00'], 'temperature_2m': [10.]}}
            for field in ('latitude', 'longitude', 'time'):
                bad = copy.deepcopy(response)
                if field == 'time':
                    bad['hourly_units']['time'] = 'unixtime'
                else:
                    bad[field] += 1.
                with self.subTest(field=field), self.assertRaises(ValueError):
                    open_meteo_frame(bad, manifest['snapshots'][0])

    def test_vendor_csv_uses_explicit_columns_and_received_evidence(self):
        from data_loading.weather_generator.adapters import vendor_frame
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = asset_fixture(Path(directory))
            meta = manifest['snapshots'][0]
            meta['variables'][0]['column'] = 'pred_tt2'
            source = pd.DataFrame({'ts': ['2026-01-01T01:00:00Z'], 'pred_tt2': [10.]})
            result = vendor_frame(source, meta, time_col='ts', available_at_col=None)
            self.assertEqual(result['temperature_2m'].tolist(), [10.])
            self.assertEqual(str(result.available_at.iloc[0]), '2026-01-01 00:00:00+00:00')
            pd.testing.assert_frame_equal(source, pd.DataFrame({'ts': ['2026-01-01T01:00:00Z'], 'pred_tt2': [10.]}))

    def test_open_meteo_units_and_interval_semantics_are_not_guessed(self):
        from data_loading.weather_generator.adapters import open_meteo_frame
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = asset_fixture(Path(directory))
            meta = manifest['snapshots'][0]
            response = {'latitude': 30., 'longitude': 120., 'utc_offset_seconds': 0, 'hourly_units': {'time': 'iso8601', 'temperature_2m': '°C'}, 'hourly': {'time': ['2026-01-01T01:00'], 'temperature_2m': [10.]}}
            result = open_meteo_frame(response, meta)
            self.assertEqual(result['temperature_2m'].tolist(), [10.])
            response['hourly_units']['temperature_2m'] = 'unknown'
            with self.assertRaises(ValueError):
                open_meteo_frame(response, meta)

    def test_observation_without_row_release_evidence_is_rejected(self):
        from data_loading.weather_generator.adapters import vendor_frame
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = asset_fixture(Path(directory), kind='observation')
            with self.assertRaises(ValueError):
                vendor_frame(pd.DataFrame({'ts': ['2026-01-01T01:00:00Z'], 'temperature_2m': [10.]}), manifest['snapshots'][0], time_col='ts', available_at_col=None)


if __name__ == '__main__':
    unittest.main()
