# -*- coding: utf-8 -*-
"""点位证据约束的单槽孤立异常清洗；原始归档不变，不运行模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from impute_hvac_data import impute_table, read_table, write_csv
from migrate_hvac_data import BUILDINGS, DEFAULT_ROOT, ROUTES, VERSIONS, sha256_file

HALF_WINDOW = 3
AUDIT_COLUMNS = ['time', 'point', 'old_value', 'local_baseline', 'total_value',
                 'total_baseline', 'total_residual', 'neighbor_spread', 'amplitude_floor',
                 'point_residual', 'explained_fraction', 'detection_known_at', 'status']


def isolated_evidence(series):
    """前后各3槽必须完整且稳定；只有单槽离群，不清除短平台/阶跃。"""
    neighbors = pd.concat([series.shift(k) for k in (-3, -2, -1, 1, 2, 3)], axis=1)
    baseline = neighbors.median(axis=1)
    residual = series - baseline
    spread = neighbors.max(axis=1) - neighbors.min(axis=1)
    mad = neighbors.sub(baseline, axis=0).abs().median(axis=1)
    floor = baseline.abs().mul(0.10).clip(lower=1.0)
    accepted = (neighbors.notna().all(axis=1) & series.notna()
                & residual.abs().ge(floor) & residual.abs().ge(6 * 1.4826 * mad)
                & spread.le(residual.abs() * 0.25))
    return accepted, baseline, residual, spread, floor


def detect_isolated(points, start, end):
    if points.empty or not points.index.equals(pd.date_range(points.index[0], periods=len(points), freq='5min')):
        raise ValueError('需要非空连续5min点位表')
    values = points.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError('负荷不允许负数或无限值')
    total = points.sum(axis=1, min_count=len(points.columns))
    candidate, baseline, residual, spread, floor = isolated_evidence(total)
    candidate &= (points.index >= pd.Timestamp(start)) & (points.index <= pd.Timestamp(end))
    # 点位选择基于同一原始快照；不让已修正结果反过来触发第二轮清洗。
    point_evidence = {point: isolated_evidence(points[point]) for point in points}
    rows = []
    for time in points.index[candidate]:
        significant = [p for p, e in point_evidence.items() if bool(e[0].loc[time])]
        point = significant[0] if len(significant) == 1 else ''
        delta = float(point_evidence[point][2].loc[time]) if point else np.nan
        fraction = delta / float(residual.loc[time])
        status = 'accepted' if point and 0.80 <= fraction <= 1.20 else 'review_multiple_or_unexplained'
        rows.append({'time': str(time), 'point': point,
                     'old_value': float(points.loc[time, point]) if point else np.nan,
                     'local_baseline': float(point_evidence[point][1].loc[time]) if point else np.nan,
                     'total_value': float(total.loc[time]), 'total_baseline': float(baseline.loc[time]),
                     'total_residual': float(residual.loc[time]), 'neighbor_spread': float(spread.loc[time]),
                     'amplitude_floor': float(floor.loc[time]), 'point_residual': delta,
                     'explained_fraction': fraction, 'detection_known_at': str(time + pd.Timedelta(minutes=15)),
                     'status': status})
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def clean_table(source, events):
    """只屏蔽已接受的原始单点，重跑因果填补；离线检测可得时间另计。"""
    masked = source.copy()
    events = events.loc[events.status.eq('accepted')].copy()
    for row in events.itertuples():
        time = pd.Timestamp(row.time)
        if masked.loc[time, row.point] != row.old_value:
            raise ValueError('待清洗值与审计原值不匹配')
        masked.loc[time, row.point] = np.nan
    result, gaps, mask = impute_table(masked)
    decisions = pd.to_datetime(events.detection_known_at)
    event_times = pd.to_datetime(events.time)
    for gap in gaps:
        if gap['status'] != 'filled':
            continue
        start, end = pd.Timestamp(gap['gap_start']), pd.Timestamp(gap['gap_end'])
        # 包括选型时被排除的历史验证段；保守覆盖30天校准+昨日上下文。
        relevant = events.point.eq(gap['point']) & event_times.between(start - pd.Timedelta(days=31), end)
        known = max(pd.Timestamp(gap['eligibility_known_at']), decisions[relevant].max()) if relevant.any() else pd.Timestamp(gap['eligibility_known_at'])
        gap['eligibility_known_at'] = str(known)
        rows = mask.index.to_series().between(start, end)
        mask.loc[rows, 'eligibility_known_at'] = mask.loc[rows, 'eligibility_known_at'].clip(lower=known)
    changes = []
    for row in events.to_dict('records'):
        time, point = pd.Timestamp(row['time']), row['point']
        gap = next(g for g in gaps if g['point'] == point and pd.Timestamp(g['gap_start']) <= time <= pd.Timestamp(g['gap_end']))
        if gap['status'] != 'filled' or not np.isfinite(result.loc[time, point]):
            raise ValueError(f'孤立点没有足够的过去验证上下文，拒绝强填: {time}/{point}')
        row.update(new_value=float(result.loc[time, point]), method=gap['method'],
                   raw_dependency_start=gap['raw_dependency_start'], raw_dependency_end=gap['raw_dependency_end'],
                   eligibility_known_at=gap['eligibility_known_at'])
        changes.append(row)
    unchanged = masked.drop(columns='total_load').notna()
    if not source.drop(columns='total_load').where(unchanged).equals(result.drop(columns='total_load').where(unchanged)):
        raise AssertionError('非异常原始观测被修改')
    return result, gaps, mask, pd.DataFrame(changes)


def read_mask(path):
    return pd.read_csv(path, index_col='time', parse_dates=['time', 'eligibility_known_at'],
                       dtype={'total_observed': bool, 'total_imputed': bool})


def build_cleaned_dataset(root=DEFAULT_ROOT):
    """保留原始/既有填补版本；新准备根只重算受清洗影响的暖通表。"""
    root = Path(root).resolve()
    destination = root / 'outlier_remove_data/isolated_v1'
    if destination.exists():
        raise FileExistsError(f'拒绝覆盖清洗版本: {destination}')
    imputation_path = root / 'analysis/imputation/manifest.json'
    forecast_path = root / 'analysis/forecast_windows/manifest.json'
    original = json.loads(imputation_path.read_text())
    forecast = json.loads(forecast_path.read_text())
    if forecast.get('preparation_root', '.') != '.':
        raise ValueError('只能从未清洗的原始填补版本启动，不能递归清洗')
    if sha256_file(imputation_path) != forecast['imputation_manifest_sha256']:
        raise ValueError('预测清单与原填补版本不一致')
    inputs = [imputation_path, forecast_path]
    inputs += list((root / 'raw_data').rglob('*.csv'))
    inputs += list((root / 'imputed_data').rglob('*.csv'))
    inputs += list((root / 'analysis/imputation').rglob('*.csv'))
    inputs += [root / row['output'] for row in forecast['windows']]
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in inputs}
    for row in original['files'] + forecast['windows']:
        if hashes[row['output']] != row['sha256']:
            raise ValueError(f'源SHA不匹配: {row["output"]}')
    for key, digest in original['source_sha256'].items():
        if hashes['raw_data/' + key] != digest:
            raise ValueError(f'原始归档SHA不匹配: {key}')
    audits, events = [], {}
    for building in BUILDINGS:
        windows = [w for w in forecast['windows'] if w['building'] == building]
        start, end = min(w['start'] for w in windows), max(w['end'] for w in windows)
        for route in ROUTES:
            full = read_table(root / f'raw_data/hvac_all_devices/{route}/{building}_data.csv')
            reduced = read_table(root / f'raw_data/hvac_remove_devices/{route}/{building}_data.csv')
            pd.testing.assert_frame_equal(full[reduced.columns[:-1]], reduced.drop(columns='total_load'))
            accepted = []
            for version, table in zip(VERSIONS, (full, reduced)):
                audit = detect_isolated(table.drop(columns='total_load'), start, end)
                audit['version'], audit['building'], audit['route'] = version, building, route
                audits.extend(audit.to_dict('records'))
                accepted.extend(audit.loc[audit.status.eq('accepted')].to_dict('records'))
            # 同一物理设备的判定跨版本复用，避免全设备/去二次泵出现不同修正。
            events[building, route] = pd.DataFrame(accepted, columns=AUDIT_COLUMNS).drop_duplicates(['time', 'point'])
    with tempfile.TemporaryDirectory(prefix='.outlier-stage-', dir=root) as tmp:
        stage = Path(tmp)
        shutil.copytree(root / 'imputed_data', stage / 'imputed_data')
        shutil.copytree(root / 'analysis/imputation', stage / 'analysis/imputation')
        all_changes, propagation = [], []
        for version in VERSIONS:
            for route in ROUTES:
                family = f'{version}/{route}'
                changed = False
                for building in BUILDINGS:
                    key = f'{family}/{building}_data.csv'
                    source = read_table(root / 'raw_data' / key)
                    audit = events[building, route]
                    audit = audit.loc[audit.point.isin(source.columns)]
                    if audit.empty:
                        continue
                    result, gaps, mask, changes = clean_table(source, audit)
                    old = read_table(root / 'imputed_data' / key)
                    different = ~(result.eq(old) | (result.isna() & old.isna()))
                    for time, col in zip(*np.where(different.drop(columns='total_load').to_numpy())):
                        stamp, point = result.index[time], result.columns[col]
                        propagation.append({'source': key, 'time': str(stamp), 'point': point,
                                            'old_value': old.loc[stamp, point], 'new_value': result.loc[stamp, point],
                                            'was_raw_observed': bool(pd.notna(source.loc[stamp, point]))})
                    masked = source.copy()
                    for row in audit.itertuples():
                        masked.loc[pd.Timestamp(row.time), row.point] = np.nan
                    masked['total_load'] = masked.drop(columns='total_load').sum(axis=1, min_count=1)
                    write_csv(masked, stage / 'masked_raw' / key)
                    write_csv(result, stage / 'imputed_data' / key)
                    write_csv(mask, stage / 'analysis/imputation/masks' / key)
                    pd.DataFrame(gaps).to_csv(stage / 'analysis/imputation/gaps' / key, index=False, encoding='utf-8-sig')
                    changes['source'] = key
                    all_changes.extend(changes.to_dict('records'))
                    changed = True
                    print(f'{key}: corrected={len(changes)}, changed_point_cells={int(different.drop(columns="total_load").sum().sum())}', flush=True)
                if changed:
                    parts = [read_table(stage / f'imputed_data/{family}/{b}_data.csv').drop(columns='total_load').add_prefix(b + '_') for b in BUILDINGS]
                    grid = read_table(root / f'imputed_data/{family}/data.csv').index
                    combined = pd.concat(parts, axis=1).reindex(grid)
                    combined['total_load'] = combined.sum(axis=1, min_count=len(combined.columns))
                    masks = [read_mask(stage / f'analysis/imputation/masks/{family}/{b}_data.csv').reindex(grid) for b in BUILDINGS]
                    observed = pd.concat([m.total_observed.eq(True) for m in masks], axis=1).all(axis=1)
                    known = pd.concat([m.eligibility_known_at for m in masks], axis=1).max(axis=1)
                    mask = pd.DataFrame({'total_observed': observed, 'total_imputed': ~observed & combined.total_load.notna(),
                                         'eligibility_known_at': known.where(combined.total_load.notna())}, index=grid)
                    write_csv(combined, stage / f'imputed_data/{family}/data.csv')
                    write_csv(mask, stage / f'analysis/imputation/masks/{family}/data.csv')
        audit_dir = stage / 'analysis/outliers'
        audit_dir.mkdir()
        pd.DataFrame(audits, columns=[*AUDIT_COLUMNS, 'version', 'building', 'route']).to_csv(audit_dir / 'candidates.csv', index=False)
        pd.DataFrame(all_changes).to_csv(audit_dir / 'corrections.csv', index=False)
        pd.DataFrame(propagation).to_csv(audit_dir / 'changed_point_cells.csv', index=False)
        physical = {(x['source'].split('/')[1], x['source'].split('/')[2], x['time'], x['point']) for x in all_changes}
        manifest = {'recipe': 'isolated_single_slot_v1', 'original_inputs_sha256': hashes,
                    'physical_corrections': len(physical), 'versioned_corrections': len(all_changes),
                    'changed_point_cells': len(propagation), 'models_run': False,
                    'rule': 'single 5min slot; +/-3 complete raw neighbors; amplitude>=max(10% local median,1kW,6*1.4826*MAD); neighbor range<=25% excursion; exactly one isolated device explains 80%-120% total excursion',
                    'scope': 'union of current forecast windows per building; HVAC only; IT copied unchanged',
                    'causality': 'offline centered detection uses 15min future neighborhood; replacement values use past-only masked raw observations; NOT strict online replay',
                    'code_sha256': sha256_file(Path(__file__)),
                    'imputer_code_sha256': sha256_file(Path(__file__).with_name('impute_hvac_data.py'))}
        (audit_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        original['outlier_cleaning'] = {'manifest': 'analysis/outliers/manifest.json', 'sha256': sha256_file(audit_dir / 'manifest.json')}
        original['causality'] = manifest['causality']
        original['source_root'] = '../..'
        for row in original['files']:
            row['sha256'] = sha256_file(stage / row['output'])
            row.pop('filled_cells', None)
        original['masks_sha256'] = {str(p.relative_to(stage)): sha256_file(p) for p in (stage / 'analysis/imputation/masks').rglob('*.csv')}
        (stage / 'analysis/imputation/manifest.json').write_text(json.dumps(original, ensure_ascii=False, indent=2))
        if any(sha256_file(root / name) != digest for name, digest in hashes.items()):
            raise ValueError('清洗期间源数据发生变化，拒绝发布')
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print('新准备数据根:', build_cleaned_dataset(args.root))


if __name__ == '__main__':
    main()
