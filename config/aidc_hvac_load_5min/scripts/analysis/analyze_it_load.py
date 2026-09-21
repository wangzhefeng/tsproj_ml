# -*- coding: utf-8 -*-
"""IT 负荷(列头柜)窗口数据分析脚本。参考 analyze_hvac_window.py。

分析对象: dataset/aidc_hvac_load_5min/raw_data/IT_load/
  下的 A1_data.csv / A2_data.csv / A3_data.csv / data.csv 全部 4 类文件
分析窗口: 2025-10-01 00:00:00 ~ 2026-09-16 23:55:00 (IT_load 统一网格完整段)

与暖通版的差异:
- 缺失/异常段检测向量化(101088 行 x 758 列, 逐点循环不可行);
- 全 NaN 列(sheet 声明但无源文件的 82 个点位: A1 39 + A3 43)不参与
  total_load 影响分析与热力图(否则所有时间戳都被判不完整), 仅在汇总表中体现;
- data.csv 热力图 758 行不可逐行标注, 省略 y 轴标签, 以楼为序排列。

分析内容:
1. 数据缺失: 每文件每点位缺失数/缺失率/最长连续缺失段
2. 数据异常: 负值、超长恒定值段(>=12 点)、IQR 离群点
3. 缺失对 total_load 的影响: 按时间戳统计点位完整率, 线性插补估计完整总负荷,
   对比实际 total_load 找出缺失导致的低估段
4. total_load 时序可视化: data.csv 三楼总负荷 + 各楼分楼总负荷,
   标注缺失导致的不完整时间戳; 缺失热力图(data.csv 非全空点位)

输出: dataset/aidc_hvac_load_5min/analysis/data_v1/raw_diagnostics/IT_load/
  - IT_load_missing_summary.csv / IT_load_anomaly_summary.csv / IT_load_total_load_gap_summary.csv
    (均含 file 列区分 A1/A2/A3/data)
  - IT_load_total_load.png / IT_load_missing_heatmap.png
  - IT_load_building_total_load.png (A1/A2/A3 分楼三联图)
"""

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = [
    'Hiragino Sans GB', 'STHeiti', 'Songti SC', 'Arial Unicode MS',
]
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------- 路径与常量
REPO = Path(__file__).resolve().parents[4]
DATA = REPO / 'dataset' / 'aidc_hvac_load_5min' / 'raw_data' / 'IT_load'
OUT = DATA.parent.parent / 'analysis/data_v1/raw_diagnostics/IT_load'
TOTAL_COL = 'total_load'

START = pd.Timestamp('2025-10-01 00:00:00')
END = pd.Timestamp('2026-09-16 23:55:00')
GRID = pd.date_range(START, END, freq='5min')

FILES = ['A1_data.csv', 'A2_data.csv', 'A3_data.csv', 'data.csv']
CONST_RUN_MIN = 12                      # 恒定值段告警阈值: 连续 12 点(1h)取值完全相同
IQR_K = 3.0                             # IQR 离群阈值


# ---------------------------------------------------------------- 工具函数
def true_segments(mask: pd.Series):
    """向量化返回布尔序列中连续 True 段的 (起始, 结束, 长度) 列表。"""
    m = mask.to_numpy(dtype=bool)
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [len(m)]
    idx = mask.index
    return [(idx[s], idx[e - 1], int(e - s)) for s, e in zip(starts, ends)]


def analyze_column(s: pd.Series):
    """单列缺失与异常统计。"""
    miss = s.isna()
    miss_segs = true_segments(miss)
    longest_gap = max((g[2] for g in miss_segs), default=0)

    v = s.dropna()
    n_neg = int((v < 0).sum())
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    n_outlier = int(((v < q1 - IQR_K * iqr) | (v > q3 + IQR_K * iqr)).sum()) if iqr > 0 else 0

    const_segs = [g for g in true_segments(s.diff().fillna(1) == 0)
                  if g[2] >= CONST_RUN_MIN - 1]
    stats = {
        'n_missing': int(miss.sum()),
        'missing_ratio': float(miss.mean()),
        'n_missing_segments': len(miss_segs),
        'longest_gap_points': longest_gap,
        'n_negative': n_neg,
        'n_outlier_iqr': n_outlier,
        'n_const_segments_ge1h': len(const_segs),
    }
    return stats, miss_segs


def plot_total_with_gaps(impact: pd.DataFrame, n_cols: int, title: str,
                         out_png: Path, total_label: str):
    """total_load 时序 + 缺失点位数双联图。"""
    incomplete = impact[impact['n_missing_points'] > 0]
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(impact.index, impact['est_full_total'], color='tab:gray',
            lw=0.8, alpha=0.7, label='est_full_total(插补估计)')
    ax.plot(impact.index, impact['total_load'], color='tab:blue',
            lw=1.0, label=total_label)
    inc_idx = incomplete.index
    if len(inc_idx):
        ax.scatter(inc_idx, impact.loc[inc_idx, 'total_load'],
                   color='red', s=8, zorder=5, label='不完整时间戳(有点位缺失)')
    ax.set_ylabel('kW')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(impact.index, 0, impact['n_missing_points'],
                     step='mid', color='tab:red', alpha=0.7)
    ax2.set_ylabel('缺失点位数')
    ax2.set_ylim(0, max(n_cols, impact['n_missing_points'].max() + 1))
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def build_impact(df: pd.DataFrame):
    """点位完整率与插补估计总负荷。剔除全 NaN 列(无源文件点位)。"""
    pts = df.drop(columns=[TOTAL_COL])
    pts = pts.drop(columns=[c for c in pts.columns if pts[c].isna().all()])
    n_cols = pts.shape[1]
    est_full = pts.interpolate(method='time', limit_direction='both').sum(axis=1)
    impact = pd.DataFrame({
        'total_load': df[TOTAL_COL],
        'est_full_total': est_full,
        'n_missing_points': n_cols - pts.notna().sum(axis=1),
        'abs_gap': (est_full - df[TOTAL_COL]),
    })
    return pts, impact


# ---------------------------------------------------------------- 主流程
OUT.mkdir(parents=True, exist_ok=True)
missing_rows, anomaly_rows, gap_rows = [], [], []

tables = {f: pd.read_csv(DATA / f, index_col=0, parse_dates=True).reindex(GRID)
          for f in FILES}

# ---- 1) 缺失 + 2) 异常: 覆盖全部 4 类文件
for fname, df in tables.items():
    file_tag = fname.replace('_data.csv', '').replace('.csv', '')
    pts = df.drop(columns=[TOTAL_COL])
    for c in pts.columns:
        stats, miss_segs = analyze_column(pts[c])
        missing_rows.append({'file': file_tag, 'column': c, **stats})
        anomaly_rows.append({'file': file_tag, 'column': c,
                             'n_negative': stats['n_negative'],
                             'n_outlier_iqr': stats['n_outlier_iqr'],
                             'n_const_segments_ge1h': stats['n_const_segments_ge1h']})
        for st, en, ln in miss_segs:
            if ln >= 12:                # 记录 >=1h 的缺失段
                gap_rows.append({'file': file_tag, 'column': c,
                                 'gap_start': st, 'gap_end': en,
                                 'gap_points': ln})
    print(f'{fname}: {pts.shape[1]} 列缺失/异常统计完成')

# ---- 3)+4) total_load 影响与可视化: data.csv(三楼合并)
df_all = tables['data.csv']
pts_all, impact_all = build_impact(df_all)
incomplete = impact_all[impact_all['n_missing_points'] > 0]
print(f'== data.csv: {pts_all.shape[1]} 有效点位(剔除全 NaN 列), '
      f'不完整时间戳 {len(incomplete)} 个({len(incomplete) / len(GRID):.1%}), '
      f'total_load 全缺 {int(df_all[TOTAL_COL].isna().sum())} 个')
print(f'   缺失致低估最大 {impact_all["abs_gap"].max():.1f} kW '
      f'@ {impact_all["abs_gap"].idxmax()}; '
      f'不完整时间戳平均低估 {incomplete["abs_gap"].mean():.1f} kW')
plot_total_with_gaps(
    impact_all, pts_all.shape[1],
    f'IT_load data.csv total_load ({START} ~ {END})',
    OUT / 'IT_load_total_load.png',
    'total_load(三楼实际)')

# ---- 各楼 total_load 三联图(缺失影响按楼归因)
fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
for ax, b in zip(axes, ['A1', 'A2', 'A3']):
    _, imp_b = build_impact(tables[f'{b}_data.csv'])
    ax.plot(imp_b.index, imp_b['est_full_total'], color='tab:gray',
            lw=0.8, alpha=0.7, label='est_full_total(插补估计)')
    ax.plot(imp_b.index, imp_b['total_load'], color='tab:blue',
            lw=1.0, label='total_load(实际)')
    inc_b = imp_b[imp_b['n_missing_points'] > 0]
    if len(inc_b):
        ax.scatter(inc_b.index, inc_b['total_load'], color='red',
                   s=8, zorder=5, label='不完整时间戳')
    ax.set_ylabel(f'{b} kW')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_title(f'IT_load 分楼 total_load ({START} ~ {END})')
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / 'IT_load_building_total_load.png', dpi=130)
plt.close(fig)

# ---- 缺失热力图(data.csv 有效点位, 行=点位, 列=时间; 点位按 A1|A2|A3 楼序)
missing_matrix = pts_all.isna().astype(float)
n_cols = pts_all.shape[1]
fig, ax = plt.subplots(figsize=(16, 10))
ax.imshow(missing_matrix.T.values, aspect='auto', cmap='Reds',
          vmin=0, vmax=1, interpolation='nearest')
ax.set_yticks([])
b1 = sum(c.startswith('A1_') for c in missing_matrix.columns)
b2 = b1 + sum(c.startswith('A2_') for c in missing_matrix.columns)
for y, lab in [(b1 / 2, 'A1'), ((b1 + b2) / 2, 'A2'),
               ((b2 + n_cols) / 2, 'A3')]:
    ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=10)
tick_idx = np.linspace(0, len(GRID) - 1, 9).astype(int)
ax.set_xticks(tick_idx)
ax.set_xticklabels([str(GRID[i])[:16] for i in tick_idx],
                   rotation=30, ha='right', fontsize=7)
ax.set_title(f'IT_load 点位缺失热力图(红=缺失, {n_cols} 个有效点位, 剔除 82 个无源文件点位)')
fig.tight_layout()
fig.savefig(OUT / 'IT_load_missing_heatmap.png', dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- 汇总落盘
miss_df = pd.DataFrame(missing_rows)
miss_df.to_csv(OUT / 'IT_load_missing_summary.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(anomaly_rows).to_csv(OUT / 'IT_load_anomaly_summary.csv', index=False,
                                  encoding='utf-8-sig')
gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(OUT / 'IT_load_total_load_gap_summary.csv', index=False,
              encoding='utf-8-sig')

print('\n== 各文件缺失率概览 ==')
summ = miss_df.groupby('file').agg(
    n_columns=('column', 'count'),
    cols_with_missing=('n_missing', lambda s: int((s > 0).sum())),
    max_missing_ratio=('missing_ratio', 'max'),
    mean_missing_ratio=('missing_ratio', 'mean'))
print(summ.to_string())

print('\n== 异常概览(非零列数, 按文件) ==')
an_df = pd.DataFrame(anomaly_rows)
for col in ['n_negative', 'n_outlier_iqr', 'n_const_segments_ge1h']:
    nz = an_df[an_df[col] > 0]
    print(f'{col}: {len(nz)} 列')
    if len(nz):
        top = nz.nlargest(6, col)[['file', 'column', col]]
        print(top.to_string(index=False))

print('\n== >=1h 缺失段(top20) ==')
if len(gap_df):
    print(gap_df.nlargest(20, 'gap_points').to_string(index=False))
else:
    print('无')

print('\n输出目录:', OUT)
