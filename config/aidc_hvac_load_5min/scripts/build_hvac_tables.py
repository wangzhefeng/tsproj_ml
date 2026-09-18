# -*- coding: utf-8 -*-
"""暖通负荷(大设备)数据提取与处理脚本。

依据 all_ids.xlsx「暖通负荷(大设备)」sheet 提取 A1/A2/A3 楼暖通大设备点位数据:

- spot_id:   从 dataset/aidc_load_5min/A1_A2_A3_points/<data_type>/ 提取点位 CSV
- 备注:      设备版本区分 —— hvac_all_devices(全部设备) / hvac_remove_devices(剔除冷冻水二次泵)
- route:     A / B 两路分别成表
- data_type: A1 / A2 / A3 楼区分

时间网格(缺失不填补, 保留 NaN):
- A1: 2025-10-01 00:00:00 ~ 2026-09-16 23:55:00, 5min
- A3: 2025-10-01 00:00:00 ~ 2026-09-16 23:55:00, 5min
- A2: 2026-07-24 10:05:00 ~ 2026-09-16 23:55:00, 5min
- 数据源: 基础目录 <data_type>/ (~2026-07-31) 拼接增量目录 <data_type>_20260801-20260916/
  (与基础数据无重叠时间戳, 重复时间戳直接 RAISE)
- data.csv 采用 A1 的完整网格, 其他楼列在各自有效范围之外为 NaN

输出: dataset/aidc_hvac_load_5min/{hvac_all_devices,hvac_remove_devices}/{route_A,route_B}/
  - A1_data.csv / A2_data.csv / A3_data.csv: time + 各 spot_id 列 + total_load(该楼该路点位求和)
  - data.csv: time + 三楼全部点位列(列名加 A1_/A2_/A3_ 楼前缀) + total_load(三楼该路总负荷)
求和规则: 按行对非空值求和, 全部缺失时 total_load 为 NaN(min_count=1)。
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- 路径
REPO = Path(__file__).resolve().parents[3]                    # 仓库根目录
SRC = REPO / 'dataset' / 'aidc_load_5min' / 'A1_A2_A3_points'
OUT = REPO / 'dataset' / 'aidc_hvac_load_5min'
XLSX = SRC / 'all_ids.xlsx'
SHEET = '暖通负荷(大设备)'

# ---------------------------------------------------------------- 常量
BUILDINGS = ['A1', 'A2', 'A3']
DATA_TYPE_OF = {b: f'{b}楼暖通电力负荷' for b in BUILDINGS}
REMOVE_REMARK = '冷冻水二次泵'
TOTAL_COL = 'total_load'

# 各楼 5min 时间网格(按核实后的数据范围; 增量 20260801-20260916 已于 2026-09-17 补全到 09-16)
GRID_OF = {
    'A1': pd.date_range('2025-10-01 00:00:00', '2026-09-16 23:55:00', freq='5min'),
    'A2': pd.date_range('2026-07-24 10:05:00', '2026-09-16 23:55:00', freq='5min'),
    'A3': pd.date_range('2025-10-01 00:00:00', '2026-09-16 23:55:00', freq='5min'),
}
GRID_FULL = GRID_OF['A1']                                     # data.csv 并集网格

# 数据版本: 目录名 -> 点位过滤函数(按 sheet 行)
VERSIONS = {
    'hvac_all_devices': lambda df: df,
    'hvac_remove_devices': lambda df: df[df['备注'] != REMOVE_REMARK],
}

# ---------------------------------------------------------------- 读 sheet
meta = pd.read_excel(XLSX, sheet_name=SHEET)
need_cols = {'data_type', 'spot_id', '备注', 'route'}
assert need_cols <= set(meta.columns), f'sheet 缺少必要列: {need_cols - set(meta.columns)}'
assert meta['spot_id'].is_unique, 'sheet 内 spot_id 存在重复'
assert set(meta['data_type']) == set(DATA_TYPE_OF.values()), \
    f'data_type 取值异常: {sorted(meta["data_type"].unique())}'
assert set(meta['route']) <= {'A', 'B'}, f'route 取值异常: {sorted(meta["route"].unique())}'


def load_point_series(data_type: str, spot_id: str) -> pd.Series:
    """读取单点位 CSV(基础目录 + 增量目录拼接), 返回以 time 为索引的 float Series。"""
    parts = []
    for suffix in ['', '_20260801-20260916']:
        f = SRC / f'{data_type}{suffix}' / f'{spot_id}.csv'
        if f.exists():
            parts.append(pd.read_csv(f, encoding='utf-8-sig'))
    if not parts:
        raise FileNotFoundError(f'缺少点位源文件: {data_type}/{spot_id}.csv(含增量目录)')
    raw = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(raw['time'])
    assert not t.duplicated().any(), f'{spot_id} 存在重复时间戳(含基础/增量重叠)'
    # 值列 astype 失败即抛错: 源数据应为纯数值, 不静默吞脏值
    v = raw['value'].astype('float64')
    s = pd.Series(v.to_numpy(), index=t)
    return s.sort_index()


def build_building_table(building: str, sub: pd.DataFrame) -> pd.DataFrame:
    """按楼内点位清单组装宽表: time + spot_id 列 + total_load。"""
    idx = GRID_OF[building]
    series_map: dict[str, pd.Series] = {}
    for sid in sub['spot_id']:
        s = load_point_series(DATA_TYPE_OF[building], sid)
        # reindex 到楼网格; 超出网格的源数据(提前/延后)直接丢弃
        series_map[sid] = s.reindex(idx)
    df = pd.DataFrame(series_map, index=idx)
    df[TOTAL_COL] = df.sum(axis=1, min_count=1)
    df.index.name = 'time'
    return df


# ---------------------------------------------------------------- 主流程
for version, keep in VERSIONS.items():
    for route in ['A', 'B']:
        out_dir = OUT / version / f'route_{route}'
        out_dir.mkdir(parents=True, exist_ok=True)

        building_dfs: dict[str, pd.DataFrame] = {}
        for b in BUILDINGS:
            sub = meta[(meta['data_type'] == DATA_TYPE_OF[b]) & (meta['route'] == route)]
            sub = keep(sub)
            assert sub['spot_id'].is_unique, f'{b} route_{route} spot_id 重复'
            df_b = build_building_table(b, sub)
            building_dfs[b] = df_b
            df_b.to_csv(out_dir / f'{b}_data.csv', encoding='utf-8-sig')
            n_valid = int(df_b[TOTAL_COL].notna().sum())
            print(f'{version}/route_{route}/{b}_data.csv: '
                  f'{len(df_b)} 行 x {df_b.shape[1]} 列({df_b.shape[1] - 1} 点位), '
                  f'total_load 非空 {n_valid} 行')

        # data.csv: 三楼点位并列(列名加楼前缀区分, 各楼列 reindex 到完整网格) + 三楼总负荷
        combined = pd.DataFrame(index=GRID_FULL)
        for b in BUILDINGS:
            pts = building_dfs[b].drop(columns=[TOTAL_COL]).reindex(GRID_FULL)
            pts.columns = [f'{b}_{c}' for c in pts.columns]
            combined = pd.concat([combined, pts], axis=1)
        combined[TOTAL_COL] = combined.sum(axis=1, min_count=1)
        combined.index.name = 'time'
        combined.to_csv(out_dir / 'data.csv', encoding='utf-8-sig')
        n_valid = int(combined[TOTAL_COL].notna().sum())
        print(f'{version}/route_{route}/data.csv: '
              f'{len(combined)} 行 x {combined.shape[1]} 列({combined.shape[1] - 1} 点位), '
              f'total_load 非空 {n_valid} 行')

print('\n完成。输出根目录:', OUT)
