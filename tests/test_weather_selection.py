"""合成快照的完整 vintage 选择；不代表真实预报发布证据。"""
import copy
import cProfile
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from data_loading.weather_generator.pipeline import generate_weather
from forecasting_core.specs.weather import WeatherGenerationSpec
from test_weather_assets import asset_fixture
from test_weather_spec import weather_options


class WeatherSelectionTest(unittest.TestCase):
    def test_latest_complete_snapshot_does_not_mix_partial_or_late_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = asset_fixture(root)
            original = copy.deepcopy(manifest)
            times = pd.DatetimeIndex(['2026-01-01T01:00Z', '2026-01-01T02:00Z'])
            for case in ('complete', 'partial', 'late', 'tie', 'hindcast'):
                with self.subTest(case=case):
                    manifest = copy.deepcopy(original)
                    newer = copy.deepcopy(manifest['snapshots'][0])
                    newer.update(snapshot_id='run-2', init_time='2025-12-31T23:00Z')
                    frame = pd.read_csv(root / 'normalized.csv')
                    frame['temperature_2m'] += 100.
                    if case == 'partial':
                        frame = frame.iloc[:1].copy()
                    elif case == 'late':
                        newer['received_at'] = '2026-01-01T00:30Z'
                        frame['available_at'] = newer['received_at']
                    elif case == 'tie':
                        newer['init_time'] = manifest['snapshots'][0]['init_time']
                    elif case == 'hindcast':
                        newer['data_kind'] = 'hindcast'
                    path = root / 'newer.csv'
                    frame.to_csv(path, index=False)
                    newer['normalized'] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    manifest['snapshots'].append(newer)
                    path = root / 'manifest.json'
                    path.write_text(json.dumps(manifest))
                    options = weather_options()
                    options['inputs'] = [{'manifest': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
                    spec = WeatherGenerationSpec.from_mapping(options)
                    if case == 'tie':
                        with self.assertRaisesRegex(ValueError, 'conflicting'):
                            generate_weather(spec, root, pd.Timestamp('2026-01-01T00:00Z'), times, ())
                        continue
                    profiler = cProfile.Profile()
                    profiler.enable()
                    result, proof = generate_weather(spec, root, pd.Timestamp('2026-01-01T00:00Z'), times, ())
                    profiler.disable()
                    if case == 'complete':
                        calls = sum(item.callcount for item in profiler.getstats()
                                    if getattr(item.code, 'co_name', None) == 'resampled_value')
                        self.assertEqual(calls, len(times), 'older ranks must not be materialized after a complete newest run')
                    expected = [110., 112.] if case == 'complete' else [10., 12.]
                    self.assertEqual(result.temperature.tolist(), expected)
                    self.assertEqual(proof['snapshot_id'], 'run-2' if case == 'complete' else 'run-1')
                    repeated, repeated_proof = generate_weather(spec, root, pd.Timestamp('2026-01-01T00:00Z'), times, ())
                    pd.testing.assert_frame_equal(result, repeated)
                    self.assertEqual(proof, repeated_proof)
                    self.assertEqual(json.loads(path.read_text()), manifest)
