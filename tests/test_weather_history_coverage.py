"""活动天气历史覆盖合同；不运行正式模型。"""
import unittest

import pandas as pd

from scripts import build_scenario_weather as builder


class WeatherHistoryCoverageTest(unittest.TestCase):
    def test_history_covers_all_available_august_data(self):
        self.assertEqual(builder.HISTORY_END, pd.Timestamp('2026-08-31 23:59:59'))


if __name__ == '__main__':
    unittest.main()
