"""跨年新能源窗口、数据及非递归冷启动回归。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'config/hongtaiyang_cesuan'))
from annual_backtest import schedule, forecast_window, assemble_year
from annual_reporting import write_annual_results
from prepare import validate_frame
from prepare_xinnengyuan_2026 import PERIOD
from config.config_loader import load_yaml_config


class CrossYearTests(unittest.TestCase):
    def config(self, freq):
        return load_yaml_config(ROOT / f'config/hongtaiyang_cesuan/xinnengyuan_2026/demand_load/freq_{freq}/lgbm_direct-pointwise.yaml')

    def test_schedule(self):
        daily = schedule('1D', period=PERIOD)
        fast = schedule('15min', period=PERIOD)
        self.assertEqual((len(daily), len(fast)), (11, 335))
        for freq in ('1D', '15min'):
            for start, origin, times in schedule(freq, period=PERIOD):
                self.assertEqual(start, max(pd.Timestamp('2025-09-01'), times[0] - pd.DateOffset(months=3)))
                self.assertLess(origin, times[0])
        self.assertEqual(daily[3][0], pd.Timestamp('2025-10-01'))
        self.assertEqual(daily[-1][2][-1], pd.Timestamp('2026-08-31'))
        self.assertEqual(fast[-1][2][-1], pd.Timestamp('2026-08-31 23:45'))

    def test_real_data_and_configs(self):
        base = ROOT / 'dataset/hongtaiyang_cesuan/xinnengyuan_2026'
        raw = validate_frame(pd.read_csv(base / 'demand_load.csv'), **PERIOD)
        daily = validate_frame(pd.read_csv(base / 'freq_1day/demand_load.csv'), '1D', **PERIOD)
        np.testing.assert_allclose(daily.value, raw.set_index('time').value.resample('1D').mean())
        for freq in ('1day', '15min'):
            self.assertEqual(self.config(freq).result_method()['method_label'], 'direct-pointwise')
        with self.assertRaises(ValueError):
            validate_frame(raw.iloc[1:], **PERIOD)

    def test_daily_first_and_rolling_window_asof(self):
        actual = validate_frame(pd.read_csv(ROOT / 'dataset/hongtaiyang_cesuan/xinnengyuan_2026/freq_1day/demand_load.csv'), '1D', **PERIOD)
        cfg = self.config('1day')
        for index in (0, 3):
            window = schedule('1D', period=PERIOD)[index]
            predicted, audit = forecast_window(cfg, actual, window, short_pointwise=index == 0)
            changed = actual.copy()
            changed.loc[(changed.time < window[0]) | (changed.time > window[1]), 'value'] = 1e9
            other, _ = forecast_window(cfg, changed, window, short_pointwise=index == 0)
            np.testing.assert_array_equal(predicted.y_pred, other.y_pred)
            self.assertEqual(audit['model_count'], 1)
            self.assertEqual(audit['training_horizon'], 1 if index == 0 else len(window[2]))
            self.assertLessEqual(pd.Timestamp(audit['training_label_end_max']), window[1])

    def test_assembly_and_reporting(self):
        times = pd.date_range(PERIOD['start'], PERIOD['end'], freq='1D', inclusive='left')
        actual = pd.DataFrame({'time': times, 'value': 10.})
        windows = [pd.DataFrame({'time': w[2], 'y_pred': 12.}) for w in schedule('1D', period=PERIOD)]
        annual = assemble_year(actual, windows, '1D', period=PERIOD)
        self.assertTrue(annual.loc[annual.time < '2025-10-01', 'y_pred'].eq(10).all())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            scores = write_annual_results(out, annual, '1D', {'period': PERIOD})
            self.assertEqual(scores['rows'], 365)
            self.assertEqual(len(list(out.rglob('*.png'))), 13)
            diagnostics = pd.read_csv(out / 'diagnostic_scores_df.csv')
            row = diagnostics[diagnostics.group.eq('model_period')].iloc[0]
            self.assertEqual(row.n_points, 335)
            self.assertEqual(row.MAE, 2.)


if __name__ == '__main__':
    unittest.main()
