"""多场景天气绑定核验：共享 processed 资产逐目标覆盖检查，只发布绑定清单。

场景配置经 build_scenario_weather.weather_source 直连共享资产；本入口按
场景注册表枚举预测目标，核验时间网格、资产窗口覆盖与六项实测/预报有限性，
把 target SHA 与资产血缘写入 dataset/<场景>/analysis/weather_binding/ 下的
绑定清单（HVAC 按数据版本各一份；v3 只发现原 data.csv，不自动扩展具名派生）。
不复制天气数值、不改共享资产。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.weather_data.build_scenario_weather import read_processed, SIX_MAPPING

REQUIRED_COLUMNS = [*SIX_MAPPING, *SIX_MAPPING.values()]
PROCESSED_DIR = 'dataset/shared/weather/processed'


def _entry(dataset, asset, grid, globs, manifest=None, version=None):
    return {'dataset': dataset, 'asset': asset, 'grid': grid, 'globs': globs,
            'manifest': manifest or 'analysis/weather_binding/manifest.csv', 'version': version}


# 场景注册表：globs 相对 dataset/<dataset>/；manifest 相对 dataset/<dataset>/。
SCENARIOS = {
    'aidc_load_15min_daily': _entry('aidc_load_15min_daily', '15min', '15min',
                                    ['*_Loads_15min_*.csv', 'forecasting_data/*_Loads_15min_*.csv']),
    'aidc_load_15min_rolling': _entry('aidc_load_15min_rolling', '15min', '15min',
                                      ['*_Loads_15min_*.csv', 'forecasting_data/*_Loads_15min_*.csv']),
    'aidc_load_15min_short': _entry('aidc_load_15min_short', '15min', '15min',
                                    ['*_Loads_15min_*.csv', 'forecasting_data/*_Loads_15min_*.csv']),
    'aidc_ess_selfuse_load': _entry('aidc_ess_selfuse_load', '5min', '5min',
                                    ['forecasting_data/*_ESS_PCS_merged_5min_*.csv']),
    'aidc_power_month': _entry('aidc_power_month', None, None, []),
    'aidc_hvac_load_5min_data_v1': _entry(
        'aidc_hvac_load_5min', '5min', '5min', ['forecast_data/data_v1/*/*/*.csv'],
        manifest='analysis/weather_binding/data_v1/manifest.csv', version='data_v1'),
    'aidc_hvac_load_5min_data_v2': _entry(
        'aidc_hvac_load_5min', '5min', '5min', ['forecast_data/data_v2/*/*/*.csv'],
        manifest='analysis/weather_binding/data_v2/manifest.csv', version='data_v2'),
    'aidc_hvac_load_5min_data_v3': _entry(
        'aidc_hvac_load_5min', '5min', '5min', ['forecast_data/data_v3/*/A1_all/data.csv'],
        manifest='analysis/weather_binding/data_v3/manifest.csv', version='data_v3'),
}
# power_month 两个频率组各绑各的资产。
SCENARIOS['aidc_power_month']['groups'] = [
    {'asset': '1day', 'grid': '1D', 'globs': ['freq_1day/*_Loads_1day_*.csv']},
    {'asset': '1ME', 'grid': '1ME', 'globs': ['freq_1month/*_Loads_1month_*.csv']},
]


def target_grid(path, grid):
    times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path, usecols=['time'])['time']))
    aligned = times.equals(times.floor(grid)) if grid != '1ME' else True
    if (times.empty or times.hasnans or times.has_duplicates
            or not times.is_monotonic_increasing
            or not aligned
            or not times.equals(pd.date_range(times[0], times[-1], freq=grid))):
        raise ValueError(f'{path}: 目标时间必须为非空、唯一、连续的 {grid} 网格')
    return times


def _groups(entry):
    if entry.get('groups'):
        return entry['groups']
    return [{'asset': entry['asset'], 'grid': entry['grid'], 'globs': entry['globs']}]


def bind_scenario(scenario):
    """返回 manifest 行；不写盘。"""
    if scenario not in SCENARIOS:
        raise ValueError(f'未知场景: {scenario}（可选 {sorted(SCENARIOS)}）')
    entry = SCENARIOS[scenario]
    dataset_dir = ROOT / 'dataset' / entry['dataset']
    rows = []
    for group in _groups(entry):
        asset_path = ROOT / PROCESSED_DIR / f"weather_{group['asset']}.csv"
        frame, metadata = read_processed(asset_path)
        paths = sorted({path for pattern in group['globs'] for path in dataset_dir.glob(pattern)})
        if not paths:
            raise FileNotFoundError(f'{scenario}: {group["globs"]} 未匹配到目标')
        for path in paths:
            times = target_grid(path, group['grid'])
            weather = frame.loc[times[0]:times[-1]]
            if (len(weather) != len(times)
                    or not pd.DatetimeIndex(weather.index).equals(times)):
                raise ValueError(f'{path}: 共享天气未完整覆盖目标窗口 {times[0]}~{times[-1]}')
            if not weather[REQUIRED_COLUMNS].notna().all().all():
                raise ValueError(f'{path}: 六项天气实测/预报缺失或非有限，不能绑定')
            row = {
                'scenario': entry['dataset'],
                'target_file': str(path.relative_to(ROOT)),
                'target_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'start': str(times[0]), 'end': str(times[-1]), 'rows': len(times),
                'weather_file': metadata['processed_asset']['file'],
                'weather_sha256': metadata['processed_asset']['sha256'],
                'weather_meta': metadata['processed_asset']['metadata'],
                'weather_meta_sha256': metadata['processed_asset']['metadata_sha256'],
            }
            if entry['version']:
                row['data_version'] = entry['version']
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True, choices=sorted(SCENARIOS))
    parser.add_argument('--replace', action='store_true', help='覆盖既有绑定清单；默认拒绝')
    args = parser.parse_args()
    entry = SCENARIOS[args.scenario]
    rows = bind_scenario(args.scenario)
    dest = ROOT / 'dataset' / entry['dataset'] / entry['manifest']
    if dest.exists() and not args.replace:
        raise FileExistsError(f'{dest} 已存在；确认后使用 --replace 重建')
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(json.dumps({'scenario': args.scenario, 'count': len(rows),
                      'manifest': str(dest.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
