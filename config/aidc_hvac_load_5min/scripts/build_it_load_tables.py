# -*- coding: utf-8 -*-
"""列头柜负荷(IT 负荷)数据提取与处理脚本。

依据 all_ids.xlsx「列头柜负荷」sheet 提取 A1/A2/A3 楼列头柜点位数据:

- data_type: A1 / A2 / A3 楼区分(对应源目录 <data_type>/)
- spot_id:   从 dataset/aidc_load_5min/A1_A2_A3_points/<data_type>/ 提取点位 CSV
- 不分 route、不做版本过滤, 仅按楼汇总

时间网格(缺失不填补, 保留 NaN):
- A1/A2/A3 统一: 2025-10-01 00:00:00 ~ 2026-09-16 23:55:00, 5min
- 数据源: 基础目录 <data_type>/ (~2026-07-31) 拼接增量目录 <data_type>_20260801-20260916/
  (与基础数据无重叠时间戳, 重复时间戳直接 RAISE)

点位文件缺失处理(sheet 声明但无源文件, 与 build_hvac_tables.py 的 RAISE 不同):
- sheet 中 43 个 spot_id 在 A1/A3 两楼的 CSV 均不存在:
  * 39 个 0_10xx 段 id 被同时登记在 A1(五楼 IT501/IT504) 与 A3(四楼 IT401/IT404),
    实际两个楼的目录都没有对应文件;
  * 4 个 50_0_10x 段 id (A3 二楼 YL202-RBB) 同样无文件。
- 这些点位在输出中保留为全 NaN 列(不改变 total_load 语义), 并在运行时打印警告清单。

输出: dataset/aidc_hvac_load_5min/raw_data/IT_load/ (原始归档存在时禁止覆盖)
  - A1_data.csv / A2_data.csv / A3_data.csv: time + 各 spot_id 列 + total_load(该楼点位求和)
  - data.csv: time + 三楼全部点位列(列名加 A1_/A2_/A3_ 楼前缀) + total_load(三楼总负荷)
求和规则: 按行对非空值求和, 全部缺失时 total_load 为 NaN(min_count=1)。
"""

from pathlib import Path

import pandas as pd

from migrate_hvac_data import require_new_files

# ---------------------------------------------------------------- 路径
REPO = Path(__file__).resolve().parents[3]                    # 仓库根目录
SRC = REPO / 'dataset' / 'aidc_load_5min' / 'A1_A2_A3_points'
OUT = REPO / 'dataset' / 'aidc_hvac_load_5min' / 'raw_data' / 'IT_load'
XLSX = SRC / 'all_ids.xlsx'
SHEET = '列头柜负荷'

# ---------------------------------------------------------------- 常量
BUILDINGS = ['A1', 'A2', 'A3']
DATA_TYPE_OF = {b: f'{b}楼列头柜负荷' for b in BUILDINGS}
TOTAL_COL = 'total_load'

# 三楼统一的 5min 时间网格(已核实: 三楼基础目录均从 2025-10-01 起, 增量目录均至 2026-09-16 23:55)
GRID = pd.date_range('2025-10-01 00:00:00', '2026-09-16 23:55:00', freq='5min')

# ---------------------------------------------------------------- 读 sheet
require_new_files([OUT / name for name in ['A1_data.csv', 'A2_data.csv', 'A3_data.csv', 'data.csv']])
meta = pd.read_excel(XLSX, sheet_name=SHEET)
need_cols = {'data_type', 'spot_id'}
assert need_cols <= set(meta.columns), f'sheet 缺少必要列: {need_cols - set(meta.columns)}'
assert not meta.duplicated(subset=['data_type', 'spot_id']).any(), \
    'sheet 内 (data_type, spot_id) 存在重复'
assert set(meta['data_type']) == set(DATA_TYPE_OF.values()), \
    f'data_type 取值异常: {sorted(meta["data_type"].unique())}'


def load_point_series(data_type: str, spot_id: str) -> pd.Series | None:
    """读取单点位 CSV(基础目录 + 增量目录拼接), 返回以 time 为索引的 float Series。

    两个目录都无文件时返回 None(该点位 sheet 有登记但无源数据, 输出全 NaN 列)。
    """
    parts = []
    for suffix in ['', '_20260801-20260916']:
        f = SRC / f'{data_type}{suffix}' / f'{spot_id}.csv'
        if f.exists():
            parts.append(pd.read_csv(f, encoding='utf-8-sig'))
    if not parts:
        return None
    raw = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(raw['time'])
    assert not t.duplicated().any(), f'{spot_id} 存在重复时间戳(含基础/增量重叠)'
    # 值列 astype 失败即抛错: 源数据应为纯数值, 不静默吞脏值
    v = raw['value'].astype('float64')
    s = pd.Series(v.to_numpy(), index=t)
    return s.sort_index()


def build_building_table(building: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """按楼内点位清单组装宽表: time + spot_id 列 + total_load。返回 (表, 无源文件的 spot_id 清单)。"""
    series_map: dict[str, pd.Series] = {}
    missing: list[str] = []
    for sid in sub['spot_id']:
        s = load_point_series(DATA_TYPE_OF[building], sid)
        if s is None:
            missing.append(sid)
            series_map[sid] = pd.Series(float('nan'), index=GRID)
        else:
            # reindex 到统一网格; 超出网格的源数据(提前/延后)直接丢弃
            series_map[sid] = s.reindex(GRID)
    df = pd.DataFrame(series_map, index=GRID)
    df[TOTAL_COL] = df.sum(axis=1, min_count=1)
    df.index.name = 'time'
    return df, missing


# ---------------------------------------------------------------- 主流程
OUT.mkdir(parents=True, exist_ok=True)

building_dfs: dict[str, pd.DataFrame] = {}
for b in BUILDINGS:
    sub = meta[meta['data_type'] == DATA_TYPE_OF[b]]
    df_b, missing = build_building_table(b, sub)
    building_dfs[b] = df_b
    df_b.to_csv(OUT / f'{b}_data.csv', encoding='utf-8-sig')
    n_valid = int(df_b[TOTAL_COL].notna().sum())
    print(f'{b}_data.csv: '
          f'{len(df_b)} 行 x {df_b.shape[1]} 列({df_b.shape[1] - 1} 点位), '
          f'total_load 非空 {n_valid} 行')
    if missing:
        print(f'  警告: {b} 楼 {len(missing)} 个 sheet 点位无源文件, 输出为全 NaN 列:')
        for sid in missing:
            dev = sub.loc[sub['spot_id'] == sid, 'deviceName'].iloc[0]
            print(f'    {sid} ({dev})')

# data.csv: 三楼点位并列(列名加楼前缀区分) + 三楼总负荷
combined = pd.DataFrame(index=GRID)
for b in BUILDINGS:
    pts = building_dfs[b].drop(columns=[TOTAL_COL])
    pts.columns = [f'{b}_{c}' for c in pts.columns]
    combined = pd.concat([combined, pts], axis=1)
combined[TOTAL_COL] = combined.sum(axis=1, min_count=1)
combined.index.name = 'time'
combined.to_csv(OUT / 'data.csv', encoding='utf-8-sig')
n_valid = int(combined[TOTAL_COL].notna().sum())
print(f'data.csv: '
      f'{len(combined)} 行 x {combined.shape[1]} 列({combined.shape[1] - 1} 点位), '
      f'total_load 非空 {n_valid} 行')

print('\n完成。输出根目录:', OUT)
