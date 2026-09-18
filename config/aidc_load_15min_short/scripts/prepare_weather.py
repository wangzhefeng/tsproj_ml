"""场景天气适配：只消费共享处理后资产，不修补原始天气。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.build_scenario_weather import DEFAULT_PROCESSED, read_processed, publish, resample_hold, require_finite

HISTORY_START = pd.Timestamp('2025-01-01 00:00:00')
HISTORY_END = pd.Timestamp('2026-08-31 23:59:59')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    hourly, metadata = read_processed(args.processed)
    hist = resample_hold(hourly, '15min', HISTORY_START, HISTORY_END)
    required = ['rt_tt2', 'cal_rh', 'rt_ssr', 'rt_ws10', 'rt_ps', 'rt_rain',
                'pred_tt2', 'pred_rh', 'pred_ssrd', 'pred_ws10', 'pred_ps', 'pred_rain']
    require_finite(hist, required)
    dest = ROOT / 'dataset/aidc_load_15min_short/weather_history_15min_20250101_20260831.csv'
    result = publish(hist, dest, {'role': 'history', 'freq': '15min',
                     'builder': str(Path(__file__).relative_to(ROOT)),
                     'processed_asset': metadata['processed_asset']}, metadata['sources'])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
