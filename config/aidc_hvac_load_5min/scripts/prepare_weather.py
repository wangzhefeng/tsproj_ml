"""场景天气适配：只消费共享处理后资产，不修补原始天气。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib
import numpy as np
import yaml
import pandas as pd
from scripts.build_scenario_weather import DEFAULT_PROCESSED, read_processed, publish, resample_hold

def build_hvac_weather(hourly, metadata):
    """逐目标读取实际 5min 网格，仅对共享处理结果做适配。"""
    scenario_dir = ROOT / 'dataset/aidc_hvac_load_5min'
    target_dir = scenario_dir / 'forecast_data'
    mapping = {'rt_tt2': 'pred_tt2', 'cal_rh': 'pred_rh', 'rt_ssr': 'pred_ssrd',
               'rt_ws10': 'pred_ws10', 'rt_ps': 'pred_ps', 'rt_rain': 'pred_rain'}
    paths = sorted(target_dir.glob('*/*/*.csv'))
    if not paths:
        raise FileNotFoundError(f'{target_dir} 下无预测 CSV')
    sources = metadata['sources']
    interpolation_audit = metadata['offline_interpolation']
    outputs = []
    for path in paths:
        times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path, usecols=['time'])['time']))
        if (times.empty or times.hasnans or times.has_duplicates
                or not times.is_monotonic_increasing
                or not times.equals(times.floor('5min'))
                or not times.equals(pd.date_range(times[0], times[-1], freq='5min'))):
            raise ValueError(f'{path}: 目标时间必须为非空、唯一、连续的 5min 网格')
        weather = resample_hold(hourly, '5min', times[0], times[-1])
        columns = list(mapping) + list(mapping.values())
        if not np.isfinite(weather[columns].to_numpy(dtype=float)).all():
            raise ValueError(f'{path}: 六项天气实测/预报缺失或非有限，不能发布')
        relative = path.relative_to(target_dir)
        dest_dir = scenario_dir / 'weather_data' / relative.parent / path.stem
        dest = dest_dir / f'weather_history_5min_{times[0]:%Y%m%d}_{times[-1]:%Y%m%d}.csv'
        target_file = str(path.relative_to(ROOT))
        published = publish(weather, dest, {
            'role': 'history', 'freq': '5min', 'target_file': target_file,
            'builder': 'config/aidc_hvac_load_5min/scripts/prepare_weather.py',
            'processed_asset': metadata['processed_asset'],
            'target_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'start': str(times[0]), 'end': str(times[-1]),
            'offline_interpolation': [row for row in interpolation_audit
                                      if times[0].floor('1h') <= pd.Timestamp(row['ts']) <= times[-1].floor('1h')],
            'interpolation_semantics': 'offline two-sided interpolation; right anchor may follow forecast origin; not online observation',
            'availability_assumption': 'forecast_origin; no supplier vintage evidence; not verified ex-ante',
        }, sources)
        source = {
            'name': 'weather', 'source_type': 'file', 'history_path': published['file'],
            'time_col': 'ts', 'series_id_cols': [], 'availability': 'forecast_origin',
            'columns': [{'name': col, 'role': role, 'categorical': False}
                        for role, names in (('known_future', mapping), ('ignored', mapping.values()))
                        for col in names],
            'inference_columns': mapping,
        }
        source_yaml = dest_dir / 'weather.source.yaml'
        source_yaml.write_text(
            '# data.sources 片段，不是可独立运行的模型配置；路径相对仓库根目录。\n'
            '# forecast_origin 为可得性假设，不是供应商发布时间证据。\n'
            + yaml.safe_dump({'data': {'sources': [source]}}, sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )
        outputs.append({'scenario': 'aidc_hvac_load_5min', 'target_file': target_file,
                        'start': str(times[0]), 'end': str(times[-1]),
                        'source_yaml': str(source_yaml.relative_to(ROOT)), **published})
    pd.DataFrame(outputs).to_csv(scenario_dir / 'weather_data/manifest.csv', index=False)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    hourly, metadata = read_processed(args.processed)
    outputs = build_hvac_weather(hourly, metadata)
    print(json.dumps({'count': len(outputs), 'outputs': outputs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
