"""活动天气历史覆盖合同；不运行正式模型。"""
import unittest

import pandas as pd

from config.aidc_load_15min_daily.scripts import prepare_weather as daily
from config.aidc_load_15min_rolling.scripts import prepare_weather as rolling
from config.aidc_load_15min_short.scripts import prepare_weather as short
from config.aidc_ess_selfuse_load.scripts import prepare_weather as ess
from config.aidc_power_month.scripts import prepare_weather as monthly


class WeatherHistoryCoverageTest(unittest.TestCase):
    def test_history_covers_all_available_august_data(self):
        for scenario in (daily, rolling, short, ess, monthly):
            self.assertEqual(scenario.HISTORY_START, pd.Timestamp('2025-01-01'))
            self.assertEqual(scenario.HISTORY_END, pd.Timestamp('2026-08-31 23:59:59'))


if __name__ == '__main__':
    unittest.main()
