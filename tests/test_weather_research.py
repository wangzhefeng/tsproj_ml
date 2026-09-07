"""研究回放不改变真实发布事实，不为严格运行补出证据。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from test_weather_assets import asset_fixture
from test_weather_spec import weather_options
from forecasting_core.specs.weather import WeatherGenerationSpec
from data_loading.weather_generator.assets import WeatherAssetStore
from data_loading.weather_generator.pipeline import generate_weather


class WeatherResearchTest(unittest.TestCase):
    def test_hindcast_requires_explicit_assumption_and_preserves_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, _ = asset_fixture(root, kind='hindcast')
            options = weather_options()
            options['inputs'] = [ref]
            options['temporal']['timezone'] = 'UTC'
            origin = pd.Timestamp('2025-12-31T23:00Z')
            times = pd.DatetimeIndex(['2026-01-01T01:00Z'])
            with self.assertRaises(ValueError):
                generate_weather(WeatherGenerationSpec.from_mapping(options), root, origin, times, ())
            options['research'] = {'release_delay': '5h', 'rationale': 'SYNTHETIC assumed release, not evidence'}
            options['semantics_version'] = 'weather_research_v1'
            spec = WeatherGenerationSpec.from_mapping(options)
            store = WeatherAssetStore(root)
            before = store.load(ref)[0].frame.copy(deep=True)
            frame, proof = generate_weather(spec, root, origin, times, (), store=store)
            self.assertEqual(frame.temperature.tolist(), [10.])
            self.assertFalse(proof['production_eligible'])
            self.assertFalse(proof['strict_asof_verified'])
            self.assertEqual(proof['data_kind'], 'hindcast')
            self.assertEqual(proof['research'], options['research'])
            self.assertEqual(proof['actual_max_available_at'], '2026-01-01T00:00:00+00:00')
            assert_frame_equal(before, store.load(ref)[0].frame)
            with self.assertRaises(ValueError):
                generate_weather(spec, root, origin - pd.Timedelta('1h'), times, ())
            options['semantics_version'] = 'weather_v1'
            with self.assertRaises(ValueError):
                WeatherGenerationSpec.from_mapping(options)


if __name__ == '__main__':
    unittest.main()
