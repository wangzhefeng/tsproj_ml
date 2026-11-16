"""场景天气绑定核验：共享 processed 5min 资产逐目标覆盖检查，只发布绑定清单。

不复制天气数值、不改共享资产；模型配置经 weather_source.py 直连 processed 文件，
本入口按预测目标核验时间网格覆盖与必需列有限性，把 target SHA 与天气资产血缘
写入 analysis/weather_binding/<data_version>/manifest.csv。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.build_scenario_weather import read_processed
from config.aidc_hvac_load_5min.scripts.preparation_paths import artifact_path
from config.aidc_hvac_load_5min.scripts.weather_data.weather_source import (
    PROCESSED_WEATHER_5MIN, WEATHER_INFERENCE_COLUMNS,
)

REQUIRED_COLUMNS = [*WEATHER_INFERENCE_COLUMNS, *WEATHER_INFERENCE_COLUMNS.values()]


def target_paths(scenario_dir, data_version):
    """v3 原目标与具名研究派生可同目录共存；绑定只发现原 data.csv，不自动扩展派生。"""
    target_dir = artifact_path(scenario_dir, 'forecast_data', data_version)
    pattern = '*/A1_all/data.csv' if data_version == 'data_v3' else '*/*/*.csv'
    paths = sorted(target_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'{target_dir} 下无预测 CSV')
    return paths


def target_grid(path):
    times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path, usecols=['time'])['time']))
    if (times.empty or times.hasnans or times.has_duplicates
            or not times.is_monotonic_increasing
            or not times.equals(times.floor('5min'))
            or not times.equals(pd.date_range(times[0], times[-1], freq='5min'))):
        raise ValueError(f'{path}: 目标时间必须为非空、唯一、连续的 5min 网格')
    return times


def bind_weather(scenario_dir, weather_path, *, data_version):
    """逐目标核验共享资产覆盖；返回 manifest 行。"""
    frame, metadata = read_processed(ROOT / weather_path)
    asset_times = pd.DatetimeIndex(frame.index)
    rows = []
    for path in target_paths(scenario_dir, data_version):
        times = target_grid(path)
        weather = frame.loc[times[0]:times[-1]]
        if (len(weather) != len(times)
                or not pd.DatetimeIndex(weather.index).equals(times)):
            raise ValueError(f'{path}: 共享天气未完整覆盖目标窗口 {times[0]}~{times[-1]}')
        if not weather[REQUIRED_COLUMNS].notna().all().all():
            raise ValueError(f'{path}: 六项天气实测/预报缺失或非有限，不能绑定')
        rows.append({
            'scenario': 'aidc_hvac_load_5min', 'data_version': data_version,
            'target_file': str(path.relative_to(ROOT)),
            'target_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'start': str(times[0]), 'end': str(times[-1]), 'rows': len(times),
            'weather_file': metadata['processed_asset']['file'],
            'weather_sha256': metadata['processed_asset']['sha256'],
            'weather_meta': metadata['processed_asset']['metadata'],
            'weather_meta_sha256': metadata['processed_asset']['metadata_sha256'],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=Path(PROCESSED_WEATHER_5MIN))
    parser.add_argument('--data-version', required=True, choices=('data_v1', 'data_v2', 'data_v3'))
    parser.add_argument('--replace', action='store_true', help='覆盖既有绑定清单；默认拒绝')
    args = parser.parse_args()
    scenario_dir = ROOT / 'dataset/aidc_hvac_load_5min'
    rows = bind_weather(scenario_dir, args.processed, data_version=args.data_version)
    dest_dir = artifact_path(scenario_dir, 'analysis', None) / 'weather_binding' / args.data_version
    dest = dest_dir / 'manifest.csv'
    if dest.exists() and not args.replace:
        raise FileExistsError(f'{dest} 已存在；确认后使用 --replace 重建')
    dest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(json.dumps({'data_version': args.data_version, 'count': len(rows),
                      'manifest': str(dest.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
