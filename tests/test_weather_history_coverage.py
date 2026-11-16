"""活动天气历史覆盖合同：共享 processed 资产经绑定核验覆盖各场景目标；不运行正式模型。"""
import unittest
from pathlib import Path

import pandas as pd

from scripts.weather_data import bind_weather

ROOT = Path(__file__).resolve().parents[1]


class WeatherHistoryCoverageTest(unittest.TestCase):
    def test_shared_assets_cover_all_scenario_targets(self):
        expected_targets = {'aidc_load_15min_daily': 3, 'aidc_load_15min_rolling': 3,
                            'aidc_load_15min_short': 3, 'aidc_ess_selfuse_load': 2,
                            'aidc_power_month': 4, 'aidc_hvac_load_5min_data_v1': 32,
                            'aidc_hvac_load_5min_data_v2': 32, 'aidc_hvac_load_5min_data_v3': 2}
        self.assertEqual(set(bind_weather.SCENARIOS), set(expected_targets))
        for scenario, count in expected_targets.items():
            rows = bind_weather.bind_scenario(scenario)
            self.assertEqual(len(rows), count, scenario)
            entry = bind_weather.SCENARIOS[scenario]
            manifest = pd.read_csv(ROOT / 'dataset' / entry['dataset'] / entry['manifest'])
            published = manifest.to_dict('records')
            for row in rows:
                with self.subTest(scenario=scenario, target=row['target_file']):
                    self.assertIn(row, published)
                    self.assertTrue(row['target_file'].endswith('.csv'))

    def test_history_windows_still_bound_to_available_history(self):
        for scenario, span in (
                ('aidc_load_15min_daily', ('2025-10-01 00:00:00', '2026-07-31 23:45:00')),
                ('aidc_ess_selfuse_load', ('2025-10-01 00:00:00', '2026-07-28 23:55:00'))):
            rows = bind_weather.bind_scenario(scenario)
            self.assertEqual({row['start'] for row in rows}, {span[0]}, scenario)
            self.assertEqual({row['end'] for row in rows}, {span[1]}, scenario)
        # 日目标自 2025-10-01、月目标自 2025-10-31（月末标签）；历史下界仍为 2025-01-01。
        rows = bind_weather.bind_scenario('aidc_power_month')
        self.assertEqual({row['start'] for row in rows},
                         {'2025-10-01 00:00:00', '2025-10-31 00:00:00'})
        self.assertEqual({row['end'] for row in rows}, {'2026-07-31 00:00:00'})
        for scenario in bind_weather.SCENARIOS:
            for row in bind_weather.bind_scenario(scenario):
                self.assertGreaterEqual(row['start'], '2025-01-01 00:00:00')


if __name__ == '__main__':
    unittest.main()
