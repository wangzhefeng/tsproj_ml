# -*- coding: utf-8 -*-
"""人工红框范围内的点位清洗；替换和选型只用过去原始观测，保留不足证据项。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from impute_hvac_data import impute_series, read_table, write_csv
from clean_hvac_outliers import read_mask
from migrate_hvac_data import BUILDINGS, DEFAULT_ROOT, ROUTES, VERSIONS, sha256_file

RECIPE = Path(__file__).resolve().parents[1] / 'redbox_cleaning.json'


def detect_candidates(source, events, recipe):
    """事件前固定1h基线，隔离先前已识别异常；不读取后侧观测定阈值。"""
    selected = pd.Series(False, index=source.index)
    audit = []
    observed = source.copy()
    previous_end = None
    for event in sorted(events, key=lambda row: row['start']):
        start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end'])
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError('事件区间必须有序且不重叠')
        previous_end = end
        index = source.index[(source.index >= start) & (source.index <= end)]
        if index.empty:
            continue
        before = observed.loc[(observed.index < start) &
                              (observed.index >= start - pd.Timedelta(minutes=5 * recipe['baseline_points']))].dropna()
        entry = {'event_start': str(start), 'event_end': str(end), 'reference_count': len(before)}
        if len(before) < recipe['minimum_baseline_points']:
            audit.append({**entry, 'status': 'insufficient_reference', 'selected': 0})
            continue
        baseline = float(before.median())
        mad = float((before - baseline).abs().median())
        threshold = max(recipe['absolute_threshold_kw'], abs(baseline) * recipe['relative_threshold'],
                        recipe['mad_multiplier'] * 1.4826 * mad)
        bad = source.loc[index].notna() & (source.loc[index] - baseline).abs().ge(threshold)
        times = index[bad]
        selected.loc[times] = True
        observed.loc[times] = np.nan
        audit.append({**entry, 'baseline': baseline, 'threshold': threshold,
                      'reference_start': str(before.index[0]), 'reference_end': str(before.index[-1]),
                      'status': 'screened', 'selected': len(times)})
    return selected, audit


def fill_candidates(source, selected):
    """固定人工范围/异常mask后，复用过去遮蔽选型；含范围内受污染旧补值。"""
    masked = source.mask(selected)
    filled, gaps = impute_series(masked)
    result = source.copy()
    rows = []
    for gap in gaps:
        index = source.index[(source.index >= pd.Timestamp(gap['gap_start'])) &
                             (source.index <= pd.Timestamp(gap['gap_end'])) & selected]
        for time in index:
            accepted = gap['status'] == 'filled'
            if accepted:
                if pd.Timestamp(gap['raw_dependency_end']) >= pd.Timestamp(gap['gap_start']):
                    raise AssertionError('补值依赖跨越缺口起点')
                result.loc[time] = filled.loc[time]
            rows.append({**gap, 'time': str(time), 'old_raw_value': float(source.loc[time]),
                         'new_value': float(result.loc[time]), 'applied': accepted})
    return result, rows


def build(root=DEFAULT_ROOT, recipe_path=RECIPE):
    root, recipe_path = Path(root).resolve(), Path(recipe_path).resolve()
    recipe = json.loads(recipe_path.read_text())
    parent = (root / recipe['parent']).resolve()
    destination = root / 'outlier_remove_data' / recipe['version']
    if not parent.is_relative_to(root) or destination.exists():
        raise ValueError('准备根非法或新版本已经存在，拒绝覆盖')
    current = json.loads((root / 'analysis/forecast_windows/manifest.json').read_text())
    if current['preparation_root'] != recipe['parent']:
        raise ValueError('当前预测输入不是指定父版本')
    inputs = [p for p in parent.rglob('*') if p.is_file()]
    inputs += list((root / 'raw_data').rglob('*.csv'))
    inputs += list((root / 'forecast_data').rglob('*.csv'))
    inputs += [recipe_path, root / 'analysis/forecast_windows/manifest.json']
    hashes = {str(p): sha256_file(p) for p in inputs}
    inventory = json.loads((parent / 'analysis/imputation/manifest.json').read_text())
    for row in inventory['files']:
        if sha256_file(parent / row['output']) != row['sha256']:
            raise ValueError(f'父版本SHA失配: {row["output"]}')
    for name, digest in inventory['masks_sha256'].items():
        if sha256_file(parent / name) != digest:
            raise ValueError(f'父版本mask SHA失配: {name}')
    with tempfile.TemporaryDirectory(prefix='.redbox-stage-', dir=root) as tmp:
        stage = Path(tmp)
        shutil.copytree(parent / 'imputed_data', stage / 'imputed_data')
        shutil.copytree(parent / 'analysis/imputation', stage / 'analysis/imputation')
        corrections, screenings = [], []
        for building in BUILDINGS:
            for route in ROUTES:
                key = f'hvac_all_devices/{route}/{building}_data.csv'
                raw = read_table(root / 'raw_data' / key)
                inherited_masked = parent / 'masked_raw' / key
                source = read_table(inherited_masked) if inherited_masked.exists() else raw
                parent_values = read_table(parent / 'imputed_data' / key)
                events = [e for e in recipe['events'] if e['building'] == building and route[-1] in e['routes']]
                if not events:
                    continue
                point_results, point_rows = {}, {}
                for point in source.columns[:-1]:
                    selected, screening = detect_candidates(source[point], events, recipe)
                    # 原观测异常会污染后续缺失槽的旧补值；只在人工事件范围内重新估计。
                    envelope = pd.Series(False, index=source.index)
                    for event in events:
                        envelope |= source.index.to_series().between(pd.Timestamp(event['start']), pd.Timestamp(event['end']))
                    selected |= envelope & source[point].isna() & parent_values[point].notna()
                    screenings.extend({**row, 'building': building, 'route': route, 'point': point} for row in screening)
                    if selected.any():
                        point_results[point], point_rows[point] = fill_candidates(source[point], selected)
                for version in VERSIONS:
                    key = f'{version}/{route}/{building}_data.csv'
                    original = read_table(root / 'raw_data' / key)
                    pd.testing.assert_frame_equal(raw[original.columns[:-1]], original.drop(columns='total_load'))
                    old = read_table(parent / 'imputed_data' / key)
                    result = old.copy()
                    mask = read_mask(parent / 'analysis/imputation/masks' / key)
                    for point, rows in point_rows.items():
                        if point not in result.columns:
                            continue
                        for row in rows:
                            time = pd.Timestamp(row['time'])
                            new = point_results[point].loc[time] if row['applied'] else old.loc[time, point]
                            corrections.append({**row, 'source': key, 'building': building, 'route': route,
                                                'version': version, 'point': point,
                                                'old_value': old.loc[time, point], 'new_value': new})
                            if row['applied']:
                                result.loc[time, point] = new
                                mask.loc[time, 'total_observed'] = False
                                mask.loc[time, 'total_imputed'] = True
                                known = pd.Timestamp(row['eligibility_known_at'])
                                mask.loc[time, 'eligibility_known_at'] = max(mask.loc[time, 'eligibility_known_at'], known)
                    result['total_load'] = result.iloc[:, :-1].sum(axis=1, min_count=len(result.columns) - 1)
                    write_csv(result, stage / 'imputed_data' / key)
                    write_csv(mask, stage / 'analysis/imputation/masks' / key)
                    print(f'{key}: candidate_cells={sum(len(rows) for p, rows in point_rows.items() if p in result)}', flush=True)
        # 只重算暖通总量；IT表和mask均为父版本逐字节副本。
        for version in VERSIONS:
            for route in ROUTES:
                family = f'{version}/{route}'
                grid = read_table(parent / f'imputed_data/{family}/data.csv').index
                parts = [read_table(stage / f'imputed_data/{family}/{b}_data.csv').drop(columns='total_load').add_prefix(b + '_') for b in BUILDINGS]
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
        pd.DataFrame(screenings).to_csv(audit_dir / 'screening.csv', index=False)
        pd.DataFrame(corrections).to_csv(audit_dir / 'corrections.csv', index=False)
        pd.DataFrame([r for r in corrections if not r['applied']]).to_csv(audit_dir / 'pending.csv', index=False)
        manifest = {'recipe': recipe, 'inputs_sha256': hashes, 'code_sha256': sha256_file(Path(__file__)),
                    'imputer_code_sha256': sha256_file(Path(__file__).with_name('impute_hvac_data.py')),
                    'causality': 'manual red boxes are offline review, NOT online detection; replacement values and method scoring use only pre-gap raw observations; gap-length eligibility known at closure',
                    'overlay_policy': 'selected anomalous raw cells and pre-existing estimates within manual event envelopes reestimated from masked raw history; outside envelopes and prior isolated corrections preserved',
                    'models_run': False, 'candidate_versioned_cells': len(corrections),
                    'applied_versioned_cells': sum(r['applied'] for r in corrections)}
        (audit_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        inventory['outlier_cleaning'] = {'manifest': 'analysis/outliers/manifest.json', 'sha256': sha256_file(audit_dir / 'manifest.json')}
        inventory['parent_preparation_root'] = recipe['parent']
        inventory['causality'] = manifest['causality']
        inventory['overlay_policy'] = manifest['overlay_policy']
        for row in inventory['files']:
            row['sha256'] = sha256_file(stage / row['output'])
            row.pop('filled_cells', None)
        inventory['masks_sha256'] = {str(p.relative_to(stage)): sha256_file(p) for p in (stage / 'analysis/imputation/masks').rglob('*.csv')}
        (stage / 'analysis/imputation/manifest.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
        if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError('构建期间输入发生变化，拒绝发布')
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    args = parser.parse_args()
    print(build(args.root, args.recipe))
