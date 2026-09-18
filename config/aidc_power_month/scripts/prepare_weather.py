"""场景天气适配：只消费共享处理后资产，不修补原始天气。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.build_scenario_weather import DEFAULT_PROCESSED, read_processed, publish, aggregate, require_finite

HISTORY_START = pd.Timestamp('2025-01-01 00:00:00')
HISTORY_END = pd.Timestamp('2026-08-31 23:59:59')
AGG_PAIRS = [('rt_tt2', 'rt_tt2', 'mean'), ('rt_tt2_max', 'rt_tt2', 'max'), ('rt_tt2_min', 'rt_tt2', 'min'),
             ('cal_rh', 'cal_rh', 'mean'), ('rt_ssr', 'rt_ssr', 'sum'),
             ('rt_ws10', 'rt_ws10', 'mean'), ('rt_dt', 'rt_dt', 'mean')]
PRED_AGG_PAIRS = [('pred_tt2', 'pred_tt2', 'mean'), ('pred_tt2_max', 'pred_tt2', 'max'), ('pred_tt2_min', 'pred_tt2', 'min'),
                  ('pred_rh', 'pred_rh', 'mean'), ('pred_ssrd', 'pred_ssrd', 'sum'),
                  ('pred_ws10', 'pred_ws10', 'mean'), ('pred_dt', 'pred_dt', 'mean')]




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    hourly, metadata = read_processed(args.processed)
    hist_slice = hourly.loc[HISTORY_START:HISTORY_END]
    reports = []
    for freq_dir, rule, hist_name in (
        ('freq_1day', '1D', 'weather_history_1day_20250101_20260831.csv'),
        ('freq_1month', '1ME', 'weather_history_1month_20250131_20260831.csv'),
    ):
        actual, _ = aggregate(hist_slice[list(dict.fromkeys(c for _, c, _ in AGG_PAIRS))], rule, AGG_PAIRS)
        forecast, incomplete = aggregate(hist_slice[list(dict.fromkeys(c for _, c, _ in PRED_AGG_PAIRS))], rule, PRED_AGG_PAIRS)
        require_finite(actual, [c for c in actual.columns if c != 'ts'])
        combined = actual.merge(forecast, on='ts')
        reports.append(publish(combined, ROOT / f'dataset/aidc_power_month/{freq_dir}/{hist_name}',
                               {'role': 'history', 'freq': rule, 'incomplete_rows_nan_pred': incomplete,
                                'builder': str(Path(__file__).relative_to(ROOT)),
                                'processed_asset': metadata['processed_asset']}, metadata['sources']))
    print(json.dumps(reports, ensure_ascii=False))


if __name__ == '__main__':
    main()
