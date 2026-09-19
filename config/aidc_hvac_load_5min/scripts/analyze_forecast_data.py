# -*- coding: utf-8 -*-
"""forecast_data 只读异常候选分析与可视化；不清洗、不替换任何源值。"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_process.outlier_process import ANOMALY_TYPE_COL, OutlierParams, detect_anomalies
from impute_hvac_data import STEP, true_runs
from migrate_hvac_data import BUILDINGS, DEFAULT_ROOT, ROUTES, VERSIONS, sha256_file
from forecast_schema import SCHEMA, file_contract, resolve_preparation_root
from select_hvac_windows import publish_prepared_directories

KINDS = ('hard_issue', 'spike', 'jump', 'low_load', 'constant')
RULES = {
    'spike': 'existing detector: +/-3 slots, robust_z>=6 or neighbors within half amplitude floor; amplitude>=max(10% observed median,1 kW)',
    'jump': '|5min difference| > max(6*1.4826*MAD(observed-to-observed differences),10% observed median,1 kW)',
    'low_load': 'value <= 10% observed median; operational-state candidate, not confirmed bad data',
    'constant': '>=12 consecutive finite samples with adjacent difference <=1e-9 kW (1h slot coverage)',
    'hard_issue': 'missing/nonfinite/negative value',
    'interpretation': 'offline descriptive screening; centered spike detector uses future neighbors, NOT a forecasting feature or automatic cleaning rule',
    'upper_bound': 'equipment rated capacities unavailable: no asserted physical upper bound',
}


def analyze_series(y, observed):
    if not y.index.equals(observed.index) or observed.isna().any():
        raise ValueError('实测掩码必须与序列逐时刻完整对齐')
    observed = observed.astype(bool)
    clean = y.where(np.isfinite(y))
    basis = clean[observed].dropna()
    if basis.empty:
        raise ValueError('没有原始实测可用于阈值估计')
    median = float(basis.median())
    floor = max(abs(median) * 0.10, 1.0)
    transitions = observed & observed.shift(1, fill_value=False)
    delta = clean.diff()
    raw_delta = delta[transitions].dropna()
    mad = float((raw_delta - raw_delta.median()).abs().median()) if len(raw_delta) else 0.0
    jump_threshold = max(6 * 1.4826 * mad, floor)
    params = OutlierParams(abs_low_threshold=-np.inf, abs_high_threshold=np.inf,
                           spike_window=3, spike_z_threshold=6, spike_abs_diff_threshold=floor,
                           spike_companion_spread_threshold=floor / 2,
                           local_robust_z_threshold=np.inf, periodic_robust_z_threshold=np.inf,
                           auto_clean_statistical=False, auto_clean_spike=False)
    detected = detect_anomalies(pd.DataFrame({'time': y.index, 'value': clean.to_numpy()}), 'time', 'value', params)
    detail = pd.DataFrame({'value': y, 'observed': observed, 'transition_observed': transitions,
                           'delta_5min': delta, 'hard_issue': ~np.isfinite(y) | y.lt(0),
                           'spike': detected[ANOMALY_TYPE_COL].str.contains('spike', regex=False).to_numpy(),
                           'jump': delta.abs().gt(jump_threshold),
                           'low_load': clean.ge(0) & clean.le(max(median, 0) * 0.1),
                           'constant': False}, index=y.index)
    finite = np.isfinite(y.to_numpy())
    same = finite[1:] & finite[:-1] & (np.abs(np.diff(y.to_numpy())) <= 1e-9)
    breaks = np.r_[0, np.flatnonzero(~same) + 1, len(y)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        if stop - start >= 12 and finite[start:stop].all():
            detail.iloc[start:stop, detail.columns.get_loc('constant')] = True
    detail['candidate'] = detail[list(KINDS)].any(axis=1)
    detail['kinds'] = [';'.join(kind for kind in KINDS if row[kind]) for row in detail[list(KINDS)].to_dict('records')]
    summary = {'rows': len(y), 'n_observed': int(observed.sum()), 'n_imputed': int((~observed).sum()),
               'min': float(clean.min()), 'median': float(clean.median()), 'max': float(clean.max()),
               'observed_median_for_threshold': median, 'spike_amplitude_floor_kw': floor,
               'jump_threshold_kw': jump_threshold, 'low_load_threshold_kw': max(median, 0) * 0.1,
               'max_abs_diff_kw': float(delta.abs().max()), 'candidate_points': int(detail.candidate.sum()),
               'candidate_observed_points': int((detail.candidate & observed).sum()),
               'candidate_imputed_points': int((detail.candidate & ~observed).sum()),
               'jump_observed_transitions': int((detail.jump & transitions).sum())}
    segments = []
    for kind in KINDS:
        summary[kind + '_points'] = int(detail[kind].sum())
        for start, stop in true_runs(detail[kind]):
            part = detail.iloc[start:stop]
            segments.append({'kind': kind, 'start': str(part.index[0]), 'end': str(part.index[-1]),
                             'points': len(part), 'slot_hours': len(part) / 12,
                             'min': float(part.value.min()), 'max': float(part.value.max()),
                             'imputed_points': int((~part.observed).sum())})
    return detail, summary, segments


def observed_mask(root, relative, column, index, cache):
    mapping, _ = file_contract(relative)
    if column not in mapping:
        raise ValueError(f'未知双路预测字段: {column}')
    members = []
    for source in mapping[column]:
        path = resolve_preparation_root(root) / 'analysis/imputation/masks' / source
        if path not in cache:
            mask = pd.read_csv(path, index_col='time', parse_dates=True, usecols=['time', 'total_observed'],
                               dtype={'total_observed': bool})
            if not mask.index.is_unique:
                raise ValueError(f'重复的掩码时间戳: {path}')
            cache[path] = mask.total_observed
        result = cache[path].reindex(index)
        if result.isna().any():
            raise ValueError(f'掩码未覆盖预测数据: {path}')
        members.append(result.astype(bool))
    return pd.concat(members, axis=1).all(axis=1)


def draw_series(frame, details, output, title):
    fig, axes = plt.subplots(len(frame.columns), 1, figsize=(16, 2.65 * len(frame.columns) + 1),
                             sharex=True, squeeze=False, layout='constrained')
    for ax, column in zip(axes[:, 0], frame.columns):
        y, d = frame[column], details[column]
        ax.plot(y.index, y, lw=0.65, color='#2463a0', label='负荷')
        filled = ~d.observed
        ax.scatter(y.index[filled], y[filled], s=6, color='#e69f00', alpha=0.7, label='含填补点位', zorder=3)
        for kind, marker, color, label in [('spike', 'x', '#d62728', '尖峰候选'),
                                          ('jump', 'o', '#8e44ad', '跳变候选'),
                                          ('low_load', 'v', '#008b8b', '低负荷候选')]:
            mask = d[kind]
            ax.scatter(y.index[mask], y[mask], s=17, marker=marker, color=color, label=label, zorder=4)
        ax.plot(y.index, y.where(d.constant), lw=2, color='#7f7f7f', label='恒值≥1h', zorder=2)
        ax.set_title(column, loc='left', fontsize=10)
        ax.set_ylabel('kW')
        ax.grid(alpha=0.18)
        ax.legend(loc='upper right', ncol=3, fontsize=7)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1, 0].xaxis.get_major_locator()))
    fig.suptitle(title + '\n离线统计候选，不代表已确认错误；本分析不改输入值', fontsize=12)
    fig.savefig(output, dpi=125)
    plt.close(fig)


def draw_heatmap(frame, output, title):
    fig, axes = plt.subplots(len(frame.columns), 1, figsize=(16, 2.9 * len(frame.columns) + 1),
                             squeeze=False, layout='constrained')
    days = pd.date_range(frame.index[0].normalize(), frame.index[-1].normalize(), freq='D')
    for ax, column in zip(axes[:, 0], frame.columns):
        daily = pd.DataFrame({'day': frame.index.normalize(),
                              'slot': frame.index.hour * 12 + frame.index.minute // 5,
                              'value': frame[column].to_numpy()})
        matrix = daily.pivot(index='day', columns='slot', values='value').reindex(index=days, columns=range(288))
        image = ax.imshow(matrix.to_numpy(), aspect='auto', interpolation='nearest', cmap='viridis',
                          extent=(0, 24, len(days), 0))
        ticks = np.unique(np.linspace(0, len(days) - 1, min(7, len(days))).astype(int))
        ax.set_yticks(ticks + 0.5, [days[i].strftime('%m-%d') for i in ticks])
        ax.set_xticks(range(0, 25, 3))
        ax.set_title(column, loc='left', fontsize=10)
        ax.set_ylabel('日期')
        fig.colorbar(image, ax=ax, label='kW', fraction=0.025, pad=0.01)
    axes[-1, 0].set_xlabel('时刻（24小时制）')
    fig.suptitle(title + '\n日内热力图：每列独立色阶，本分析未标准化或改值', fontsize=12)
    fig.savefig(output, dpi=125)
    plt.close(fig)


def analyze_file(root, relative, destination, cache=None):
    root, relative, destination = Path(root), Path(relative), Path(destination)
    cache = {} if cache is None else cache
    path = root / 'forecast_data' / relative
    before = sha256_file(path)
    frame = pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.equals(
            pd.date_range(frame.index[0], periods=len(frame), freq=STEP)):
        raise ValueError(f'非唯一升序连续5min时间轴: {path}')
    mapping, target = file_contract(relative)
    if set(frame.columns) != set(mapping) or len(frame.columns) != len(mapping):
        raise ValueError(f'预测字段不符合{SCHEMA}: {relative}')
    values = frame.to_numpy(dtype=float)
    quality = {'file': str(relative), 'rows': len(frame), 'columns': len(frame.columns),
               'start': str(frame.index[0]), 'end': str(frame.index[-1]), 'sha256': before,
               'missing_cells': int(np.isnan(values).sum()), 'infinite_cells': int(np.isinf(values).sum()),
               'negative_cells': int((values < 0).sum()), 'zero_cells': int((values == 0).sum()),
               'component_mismatch_rows': 0, 'unique_sorted_regular_5min': True,
               'schema': SCHEMA, 'target_column': target}
    sums = {'hvac_total_load_' + r: [f'{b}_hvac_total_load_{r}' for b in BUILDINGS] for r in ('A', 'B')}
    sums['hvac_total_load_AB'] = ['hvac_total_load_A', 'hvac_total_load_B']
    sums['it_total_load'] = [b + '_it_total_load' for b in BUILDINGS]
    for total, components in sums.items():
        if total in frame and set(components).issubset(frame.columns):
            mismatch = ~np.isclose(frame[total], frame[components].sum(axis=1, min_count=len(components)),
                                   rtol=1e-10, atol=1e-8, equal_nan=True)
            quality['component_mismatch_rows'] += int(mismatch.sum())
    details, summaries, segments, candidates = {}, [], [], []
    for column in frame:
        observed = observed_mask(root, relative, column, frame.index, cache)
        detail, summary, runs = analyze_series(frame[column], observed)
        details[column] = detail
        summaries.append({'file': str(relative), 'column': column, **summary})
        segments.extend({'file': str(relative), 'column': column, **row} for row in runs)
        points = detail.loc[detail.candidate].copy()
        points.insert(0, 'column', column)
        points.insert(0, 'time', points.index)
        candidates.append(points.reset_index(drop=True))
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(destination / 'series_summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(candidates, ignore_index=True).to_csv(destination / 'candidate_points.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(segments, columns=['file', 'column', 'kind', 'start', 'end', 'points', 'slot_hours',
                                    'min', 'max', 'imputed_points']).to_csv(destination / 'candidate_segments.csv', index=False, encoding='utf-8-sig')
    title = str(relative) + f'\n目标字段：{target}'
    with plt.rc_context({'font.sans-serif': ['Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'path.simplify': False}):
        draw_series(frame, details, destination / 'timeseries.png', title)
        draw_heatmap(frame, destination / 'daily_heatmap.png', title)
        relative_jump = pd.DataFrame({row['column']: details[row['column']].delta_5min.abs()
                                     / row['spike_amplitude_floor_kw'] for row in summaries})
        center = relative_jump.max(axis=1).idxmax()
        zoom = frame.loc[center - pd.Timedelta(hours=3):center + pd.Timedelta(hours=3)]
        draw_series(zoom, {c: d.loc[zoom.index] for c, d in details.items()},
                    destination / 'largest_jump_zoom.png', title + f'\n最大相对单步跳变附近：{center}')
    if sha256_file(path) != before:
        raise ValueError(f'分析过程中源文件改变: {path}')
    return summaries, quality


def run(root=DEFAULT_ROOT, *, replace=False):
    root = Path(root).resolve()
    output = root / 'analysis/forecast_data_visual'
    if output.exists() and not replace:
        raise FileExistsError(f'拒绝覆盖已有可视化: {output}')
    expected = {Path(v) / r / (name + suffix + '.csv') for v in VERSIONS for r in ROUTES
                for name in ('A1_data', 'A2_data', 'A3_data', 'data') for suffix in ('', '_with_it')}
    paths = {p.relative_to(root / 'forecast_data') for p in (root / 'forecast_data').rglob('*.csv')}
    if paths != expected:
        raise ValueError(f'要求32份预测输入，缺少={expected - paths}, 多余={paths - expected}')
    dependencies = [root / 'forecast_data' / p for p in sorted(paths)]
    prepared = resolve_preparation_root(root)
    dependencies += sorted((prepared / 'analysis/imputation/masks').rglob('*.csv'))
    if (root / 'analysis/forecast_windows/manifest.json').exists():
        dependencies.append(root / 'analysis/forecast_windows/manifest.json')
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in dependencies}
    with tempfile.TemporaryDirectory(prefix='.forecast-visual-stage-', dir=root / 'analysis') as tmp:
        stage = Path(tmp) / 'analysis/forecast_data_visual'
        cache, summaries, quality = {}, [], []
        for relative in sorted(paths):
            rows, audit = analyze_file(root, relative, stage / relative.parent / relative.stem, cache)
            summaries.extend(rows)
            quality.append(audit)
            print(f'{relative}: {len(rows)}列分析完成', flush=True)
        stats = pd.DataFrame(summaries)
        stats.to_csv(stage / 'series_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(quality).to_csv(stage / 'file_quality.csv', index=False, encoding='utf-8-sig')
        if any(sha256_file(root / name) != digest for name, digest in hashes.items()):
            raise ValueError('输入或填补掩码在分析期间改变，拒绝发布')
        plots = sorted(str(p.relative_to(stage)) for p in stage.rglob('*.png'))
        expected_series = sum(len(file_contract(relative)[0]) for relative in paths)
        if len(quality) != len(expected) or len(stats) != expected_series or len(plots) != 3 * len(expected):
            raise AssertionError('双路32文件/逐字段/每文件3图覆盖验收不符')
        manifest = {'input_files': len(quality), 'series_appearances': len(stats), 'png_files': len(plots),
                    'rules': RULES, 'inputs_sha256': hashes, 'plots': plots,
                    'preparation_root': str(prepared.relative_to(root)),
                    'code_sha256': sha256_file(Path(__file__)),
                    'schema': SCHEMA, 'schema_code_sha256': sha256_file(Path(__file__).with_name('forecast_schema.py')),
                    'note': 'counts are per-file appearances; IT is shared across route/version and must not be summed as independent signals'}
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        lines = ['# forecast_data 异常候选与可视化', '',
                 f'只读分析：{len(quality)}个CSV、{len(stats)}个文件内数值列；每文件完整时序、日内热力图、最大相对跳变±3h局部图。',
                 '图中橙色为含估计点位的汇总值。所有统计标记仅供复核，不证明传感器错误；本分析未改值、未训练模型。',
                 f'输入准备版本：`{prepared.relative_to(root)}`；是否已做上游清洗以该版本审计为准。',
                 'A/B输入在两个目标路线目录中重复出现，IT也共享；不要把文件列候选数相加当作独立物理异常数。',
                 'hvac_total_load_AB为三楼两路总负荷；目录route_A/B分别以hvac_total_load_A/B为目标，其他列仅作历史输入。', '',
                 '## 筛查规则', '',
                 '- 尖峰：复用现有±3槽检测器；幅度至少max(实测中位数10%,1kW)，且robust z≥6或左右相邻值差≤幅度门槛的一半。',
                 '- 跳变：相邻5min变化绝对值超过max(6×1.4826×实测相邻差分MAD,实测中位数10%,1kW)。',
                 '- 低负荷：不超过实测中位数10%；可能是停机/负荷转移，不等于错误。',
                 '- 恒值：相邻变化≤1e-9kW、连续≥12槽；可能是设备稳定或前值填补，不等于通信冻结。',
                 '- 居中尖峰规则使用未来邻域，仅供离线分析，不能直接作预测特征。设备额定容量未知，不做物理上限判定。',
                 '- candidate_points.csv的observed只说明当前时刻是否完整实测；jump的transition_observed同时检查前后两端。', '',
                 '## 完整性', '',
                 f"缺失单元格={sum(q['missing_cells'] for q in quality)}；无限值={sum(q['infinite_cells'] for q in quality)}；负值={sum(q['negative_cells'] for q in quality)}；汇总分量不一致行={sum(q['component_mismatch_rows'] for q in quality)}。", '',
                 '## 文件索引', '', '| 输入 | 完整时序 | 日内热力图 | 跳变局部 |', '|---|---|---|---|']
        for relative in sorted(paths):
            folder = (relative.parent / relative.stem).as_posix()
            lines.append(f'| {relative} | [查看]({folder}/timeseries.png) | [查看]({folder}/daily_heatmap.png) | [查看]({folder}/largest_jump_zoom.png) |')
        (stage / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        publish_prepared_directories(Path(tmp), root, ('analysis/forecast_data_visual',), replace=replace)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--replace', action='store_true', help='明确替换旧可视化与候选报告，不修改预测输入')
    args = parser.parse_args()
    print('输出目录:', run(args.root, replace=args.replace))


if __name__ == '__main__':
    main()
