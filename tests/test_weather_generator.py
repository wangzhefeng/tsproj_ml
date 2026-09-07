"""严格气象生成：合成合法来源，不作为真实取数/模型验收。"""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from test_weather_assets import asset_fixture
from test_weather_spec import weather_options
from forecasting_core.specs.weather import WeatherGenerationSpec


class WeatherPipelineTest(unittest.TestCase):
    def test_forecast_hold_uses_fixed_snapshot_and_preserves_availability(self):
        from data_loading.weather_generator.pipeline import generate_weather
        with tempfile.TemporaryDirectory() as directory:
            ref, _ = asset_fixture(Path(directory))
            options = weather_options()
            options['inputs'] = [ref]
            options['temporal']['timezone'] = 'UTC'
            times = pd.date_range('2026-01-01T01:00Z', periods=4, freq='15min')
            frame, proof = generate_weather(WeatherGenerationSpec.from_mapping(options), directory, pd.Timestamp('2026-01-01T00:00Z'), times, ())
            self.assertEqual(frame.temperature.tolist(), [10.] * 4)
            self.assertTrue((frame.available_at == pd.Timestamp('2026-01-01T00:00Z')).all())
            self.assertEqual(proof['scenario'], 'forecast')

    def test_late_release_and_missing_coverage_raise(self):
        from data_loading.weather_generator.pipeline import generate_weather
        with tempfile.TemporaryDirectory() as directory:
            ref, _ = asset_fixture(Path(directory))
            options = weather_options()
            options['inputs'] = [ref]
            spec = WeatherGenerationSpec.from_mapping(options)
            for origin, target in [('2025-12-31T23:00Z','2026-01-01T01:00Z'), ('2026-01-01T00:00Z','2026-01-01T04:00Z')]:
                with self.subTest(origin=origin), self.assertRaises(ValueError):
                    generate_weather(spec, directory, pd.Timestamp(origin), pd.DatetimeIndex([target]), ())


if __name__ == '__main__':
    unittest.main()
