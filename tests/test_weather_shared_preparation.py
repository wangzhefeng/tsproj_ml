"""共享源治理与场景入口隔离；不训练模型。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import build_scenario_weather as weather

ROOT = Path(__file__).resolve().parents[1]


class SharedWeatherPreparationTest(unittest.TestCase):
    def test_shared_tool_does_not_own_scenarios(self):
        source = (ROOT / 'scripts/build_scenario_weather.py').read_text()
        for token in ('aidc_hvac', 'aidc_load_15min', 'aidc_power_month', 'aidc_ess', 'HISTORY_END'):
            self.assertNotIn(token, source)

    def test_shared_round_trip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / 'actual'
            source_dir.mkdir()
            times = pd.date_range('2026-09-16 15:00', '2026-09-16 23:00', freq='1h')
            frame = pd.DataFrame({'ts': times, 'rt_tt2': 280.1234567890123, 'rt_dt': 278.,
                                  'rt_ssr': 0., 'rt_ws10': 2., 'rt_rain': 0., 'rt_ps': 100000.,
                                  'pred_tt2': 281., 'pred_rh': 80., 'pred_ssrd': 0.,
                                  'pred_ws10': 3., 'pred_rain': 0., 'pred_ps': 100001.})
            frame.loc[1:7, ['rt_tt2', 'rt_dt']] = np.nan
            frame.loc[1:5, ['rt_ws10', 'rt_rain']] = np.nan
            shard = source_dir / 'weather_in_20260916_20260916.csv'
            frame.to_csv(shard, index=False)
            before = shard.read_bytes()
            repair = shard.with_suffix('.six_features_repair.json')
            repair.write_text(json.dumps({'source_sha256_after': hashlib.sha256(before).hexdigest(),
                                          'semantic_status': 'audited offline repairs'}))
            era5 = root / 'era5.csv'
            pd.DataFrame({'ts': []}).to_csv(era5, index=False)
            output = root / 'processed/weather_hourly.csv'
            command = [sys.executable, str(ROOT / 'scripts/build_scenario_weather.py'),
                       '--source-dir', str(source_dir), '--era5-path', str(era5), '--output', str(output)]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            hourly, metadata = weather.read_processed(output)
            self.assertEqual(len(hourly), 9)
            self.assertTrue(np.isfinite(hourly[list(weather.SIX_MAPPING) + list(weather.SIX_MAPPING.values())]).all(axis=None))
            self.assertEqual(len(metadata['offline_interpolation']), 31)
            self.assertEqual(metadata['source_repairs'][0]['sha256'], hashlib.sha256(repair.read_bytes()).hexdigest())
            self.assertEqual(shard.read_bytes(), before)
            first = output.read_bytes()
            subprocess.run(command, cwd=root, check=True, capture_output=True)
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(metadata['sha256_file'], hashlib.sha256(first).hexdigest())
            output.write_bytes(first + b'\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                weather.read_processed(output)

    def test_raw_destination_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'overwrite raw'):
                weather.build_shared(root / 'actual', root / 'era5.csv', root / 'actual/output.csv')
            self.assertFalse((root / 'actual').exists())

    def test_missing_anchor_and_unapproved_gaps_do_not_get_filled(self):
        times = pd.date_range('2026-09-16 15:00', '2026-09-16 23:00', freq='1h')
        frame = pd.DataFrame({'rt_tt2': np.nan}, index=times)
        with self.assertRaisesRegex(ValueError, '双端锚点'):
            weather.repair_source_intervals(frame)
        outside = pd.DataFrame({'rt_tt2': [280., np.nan, 282.]},
                               index=pd.date_range('2026-09-15', periods=3, freq='1h'))
        actual, audit = weather.repair_source_intervals(outside)
        pd.testing.assert_frame_equal(actual, outside)
        self.assertEqual(audit, [])


if __name__ == '__main__':
    unittest.main()
