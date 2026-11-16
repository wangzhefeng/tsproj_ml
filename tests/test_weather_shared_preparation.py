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

from scripts.weather_data import build_scenario_weather as weather

ROOT = Path(__file__).resolve().parents[1]


class SharedWeatherPreparationTest(unittest.TestCase):
    def test_shared_tool_does_not_own_scenarios(self):
        source = (ROOT / 'scripts/weather_data/build_scenario_weather.py').read_text()
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
            command = [sys.executable, str(ROOT / 'scripts/weather_data/build_scenario_weather.py'),
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
            self.assertFalse((root / 'processed/weather_15min.csv').exists())

    def test_resampled_shared_assets_hold_hourly_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / 'actual'
            source_dir.mkdir()
            times = pd.date_range('2026-09-16 15:00', '2026-09-16 23:00', freq='1h')
            frame = pd.DataFrame({'ts': times, 'rt_tt2': 280.5, 'rt_dt': 278., 'rt_ssr': 0.,
                                  'rt_ws10': 2., 'rt_rain': 0., 'rt_ps': 100000.,
                                  'pred_tt2': 281., 'pred_rh': 80., 'pred_ssrd': 0.,
                                  'pred_ws10': 3., 'pred_rain': 0., 'pred_ps': 100001.})
            frame.to_csv(source_dir / 'weather_in_20260916_20260916.csv', index=False)
            era5 = root / 'era5.csv'
            pd.DataFrame({'ts': []}).to_csv(era5, index=False)
            output = root / 'processed/weather_hourly.csv'
            command = [sys.executable, str(ROOT / 'scripts/weather_data/build_scenario_weather.py'),
                       '--source-dir', str(source_dir), '--era5-path', str(era5),
                       '--output', str(output), '--resample', '15min,5min']
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            hourly, hourly_meta = weather.read_processed(output)
            fine15, meta15 = weather.read_processed(root / 'processed/weather_15min.csv')
            fine5, meta5 = weather.read_processed(root / 'processed/weather_5min.csv')
            self.assertEqual((len(fine15), len(fine5)), (36, 108))
            self.assertEqual([str(fine15.index[0]), str(fine15.index[-1])],
                             ['2026-09-16 15:00:00', '2026-09-16 23:45:00'])
            self.assertEqual([str(fine5.index[0]), str(fine5.index[-1])],
                             ['2026-09-16 15:00:00', '2026-09-16 23:55:00'])
            self.assertEqual((meta15['freq'], meta5['freq']), ('15min', '5min'))
            self.assertEqual(meta15['role'], 'shared_processed')
            self.assertEqual(meta15['upsampled_from']['sha256'], hourly_meta['sha256_file'])
            six = list(weather.SIX_MAPPING) + list(weather.SIX_MAPPING.values())
            for fine in (fine15, fine5):
                held = hourly.reindex(fine.index.floor('1h'))
                for col in six:
                    np.testing.assert_array_equal(fine[col].to_numpy(), held[col].to_numpy())

    def test_aggregated_shared_assets_publish_complete_intervals_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / 'actual'
            source_dir.mkdir()
            # 两个月零两天：2026-07 完整、2026-08 完整、2026-09 仅 2 天（部分月必须被丢弃）。
            times = pd.date_range('2026-07-01', '2026-09-02 05:00', freq='1h')
            frame = pd.DataFrame({'ts': times, 'rt_tt2': 280., 'rt_dt': 278., 'rt_ssr': 2.,
                                  'rt_ws10': 4., 'rt_rain': 1., 'rt_ps': 100000.,
                                  'pred_tt2': 281., 'pred_rh': 80., 'pred_ssrd': 3.,
                                  'pred_ws10': 3., 'pred_rain': 0.5, 'pred_ps': 100001.})
            frame.to_csv(source_dir / 'weather_in_20260701_20260902.csv', index=False)
            era5 = root / 'era5.csv'
            pd.DataFrame({'ts': []}).to_csv(era5, index=False)
            output = root / 'processed/weather_hourly.csv'
            command = [sys.executable, str(ROOT / 'scripts/weather_data/build_scenario_weather.py'),
                       '--source-dir', str(source_dir), '--era5-path', str(era5),
                       '--output', str(output), '--aggregate', '1D,1ME']
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            hourly, hourly_meta = weather.read_processed(output)
            daily, meta_day = weather.read_processed(root / 'processed/weather_1day.csv')
            monthly, meta_me = weather.read_processed(root / 'processed/weather_1ME.csv')
            self.assertEqual((meta_day['freq'], meta_me['freq']), ('1D', '1ME'))
            self.assertEqual(meta_day['role'], 'shared_processed')
            self.assertEqual(meta_day['aggregated_from']['sha256'], hourly_meta['sha256_file'])
            self.assertEqual(meta_day['aggregation']['rt_ssr'], 'interval_sum')
            self.assertEqual(meta_day['aggregation']['rt_tt2'], 'interval_mean')
            # 日资产：每小时值恒定 → 均值=原值、总量=×24；仅完整日发布（09-02 部分日丢弃）。
            self.assertEqual(len(daily), 63)
            self.assertEqual(str(daily.index[-1]), '2026-09-01 00:00:00')
            np.testing.assert_allclose(daily['rt_tt2'].to_numpy(), 280.)
            np.testing.assert_allclose(daily['rt_ssr'].to_numpy(), 48.)
            np.testing.assert_allclose(daily['pred_rain'].to_numpy(), 12.)
            # 月资产：部分月 2026-09 被丢弃，只保留两个完整月（月末标签）。
            self.assertEqual([str(t) for t in monthly.index], ['2026-07-31 00:00:00', '2026-08-31 00:00:00'])
            self.assertEqual(meta_me['incomplete_intervals_dropped']['either_family_dropped'], 1)
            np.testing.assert_allclose(monthly['rt_ssr'].to_numpy(), 31 * 24 * 2.)
            six = list(weather.SIX_MAPPING) + list(weather.SIX_MAPPING.values())
            for aggregated in (daily, monthly):
                self.assertTrue(np.isfinite(aggregated[six].to_numpy()).all())

    def test_aggregate_rule_validation_and_output_naming(self):
        self.assertEqual(weather.parse_aggregate_rules(['1D', '1D', '1ME', '']), ['1D', '1ME'])
        with self.assertRaisesRegex(ValueError, '非法聚合频率'):
            weather.parse_aggregate_rules(['1W'])
        self.assertEqual(weather.aggregated_output_path(Path('a/weather_hourly.csv'), '1D'),
                         Path('a/weather_1day.csv'))
        self.assertEqual(weather.aggregated_output_path(Path('a/weather_hourly.csv'), '1ME'),
                         Path('a/weather_1ME.csv'))
        source = weather.weather_source('1ME')
        self.assertEqual(source['history_path'], 'dataset/shared/weather/processed/weather_1ME.csv')
        self.assertEqual(source['inference_columns'], weather.SIX_MAPPING)
        with self.assertRaisesRegex(ValueError, '未知共享天气资产频率'):
            weather.weather_source('2min')

    def test_resample_freq_validation_and_output_naming(self):
        self.assertEqual(weather.parse_resample_freqs(['15min', '15min', '5min', '']), ['15min', '5min'])
        for bad in ('1h', '1D', '7min', '90min'):
            with self.assertRaisesRegex(ValueError, '子小时'):
                weather.parse_resample_freqs([bad])
        with self.assertRaisesRegex(ValueError, '非法上采样频率'):
            weather.parse_resample_freqs(['soon'])
        self.assertEqual(weather.resampled_output_path(Path('a/weather_hourly.csv'), '15min'),
                         Path('a/weather_15min.csv'))
        self.assertEqual(weather.resampled_output_path(Path('a/custom.csv'), '5min'),
                         Path('a/custom_5min.csv'))

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
