# -*- coding: utf-8 -*-
"""从 imputed_data 选择近期最长完整日窗口，导出16个目标/32个有无IT场景。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from impute_hvac_data import RECIPE, STEP, publish_stage, read_table, true_runs, write_csv
from migrate_hvac_data import BUILDINGS, DEFAULT_ROOT, FAMILIES, FILES, ROUTES, VERSIONS, sha256_file


def select_window(valid, recent_start):
    valid = valid.loc[pd.Timestamp(recent_start):]
    daily = valid.resample('D').agg(['all', 'size'])
    complete = daily['all'] & daily['size'].eq(288)
    candidates = [(int(stop - start), daily.index[stop - 1], daily.index[start])
                  for start, stop in true_runs(complete)]
    if not candidates:
        raise ValueError('近期没有完整自然日窗口，拒绝生成假数据或空占位文件')
    days, last, start = max(candidates)
    return start, last + pd.Timedelta(days=1), days


def export_windows(root=DEFAULT_ROOT, recent_start=None):
    root = Path(root).resolve()
    if recent_start is None:
        recent_start = json.loads(RECIPE.read_text(encoding='utf-8'))['recent_start']
    if (root / 'analysis/forecast_windows').exists() or any((root / 'forecast_data').rglob('*.csv')):
        raise FileExistsError('预测数据或选窗审计已经存在，拒绝覆盖')
    manifest_path = root / 'analysis/imputation/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {f'imputed_data/{family}/{name}' for family in FAMILIES for name in FILES}
    if {row['output'] for row in manifest['files']} != expected or len(manifest['files']) != len(expected):
        raise ValueError('填补清单必须恰好包含20个唯一表')
    hashes = {row['output']: row['sha256'] for row in manifest['files']}
    totals, masks = {}, {}
    for output, digest in hashes.items():
        if sha256_file(root / output) != digest:
            raise ValueError(f'填补输入哈希不匹配: {output}')
        key = output.removeprefix('imputed_data/')
        frame = read_table(root / output)
        points = frame.drop(columns='total_load')
        if not np.allclose(points.sum(axis=1, min_count=len(points.columns)), frame.total_load, equal_nan=True):
            raise ValueError(f'不完整总量或求和错误: {key}')
        totals[key] = frame.total_load
        mask = pd.read_csv(root / 'analysis/imputation/masks' / key, index_col='time',
                           parse_dates=['time', 'eligibility_known_at'],
                           dtype={'total_observed': bool, 'total_imputed': bool})
        if not mask.index.equals(frame.index) or not (mask.total_observed | mask.total_imputed).equals(frame.total_load.notna()):
            raise ValueError(f'有效性审计与填补输入不一致: {key}')
        masks[key] = mask
    windows = {}
    for building in (*BUILDINGS, 'data'):
        name = building + '_data.csv' if building != 'data' else 'data.csv'
        selected = [totals[f'{v}/{r}/{name}'] for v in VERSIONS for r in ROUTES]
        for with_it in (False, True):
            members = selected + ([totals[f'IT_load/{name}']] if with_it else [])
            valid = pd.concat(members, axis=1).notna().all(axis=1)
            windows[building, with_it] = select_window(valid, recent_start)
    rows, fold_rows = [], []
    with tempfile.TemporaryDirectory(prefix='.forecast-stage-', dir=root) as tmp:
        stage = Path(tmp)
        for version in VERSIONS:
            for route in ROUTES:
                family = f'{version}/{route}'
                for building in (*BUILDINGS, 'data'):
                    name = building + '_data.csv' if building != 'data' else 'data.csv'
                    for with_it in (False, True):
                        start, stop, days = windows[building, with_it]
                        key = f'{family}/{name}'
                        it_key = f'IT_load/{name}'
                        columns = {'hvac_total_load': totals[key]}
                        if with_it:
                            columns['it_total_load'] = totals[it_key]
                        if building == 'data':
                            for b in BUILDINGS:
                                columns[b + '_hvac_total_load'] = totals[f'{family}/{b}_data.csv']
                            if with_it:
                                for b in BUILDINGS:
                                    columns[b + '_it_total_load'] = totals[f'IT_load/{b}_data.csv']
                        frame = pd.DataFrame(columns).loc[start:stop - STEP]
                        if frame.isna().any().any() or len(frame) != days * 288:
                            raise ValueError(f'选窗包含缺失或不完整天: {family}/{name}')
                        export_name = name.removesuffix('.csv') + ('_with_it.csv' if with_it else '.csv')
                        output = f'forecast_data/{family}/{export_name}'
                        write_csv(frame, stage / output)
                        mask = pd.DataFrame({'hvac_observed': masks[key].total_observed,
                                             'eligibility_known_at': masks[key].eligibility_known_at}).loc[frame.index]
                        if with_it:
                            mask['it_observed'] = masks[it_key].total_observed.loc[frame.index]
                            mask['eligibility_known_at'] = pd.concat([
                                mask.eligibility_known_at, masks[it_key].eligibility_known_at.loc[frame.index]], axis=1).max(axis=1)
                        write_csv(mask, stage / 'analysis/forecast_windows/masks' / family / export_name)
                        causal_folds = 0
                        for origin in pd.date_range(start + pd.Timedelta(days=14), stop - pd.Timedelta(days=1), freq='D'):
                            history = mask.loc[origin - pd.Timedelta(days=14):origin - STEP]
                            safe = bool((history.eligibility_known_at < origin).all())
                            causal_folds += int(safe)
                            test = mask.loc[origin:origin + pd.Timedelta(days=1) - STEP]
                            fold_rows.append({'output': output, 'origin': str(origin),
                                              'train_start': str(origin - pd.Timedelta(days=14)),
                                              'eligibility_known_before_origin': safe,
                                              'observed_target_test_rows': int(test.hvac_observed.sum())})
                        rows.append({'output': output, 'version': version, 'route': route,
                                     'building': building, 'with_it': with_it, 'start': str(start),
                                     'end': str(stop - STEP), 'days': days, 'rows': len(frame),
                                     'folds_14_1': max(0, days - 14), 'eligibility_safe_folds_14_1': causal_folds,
                                     'meets_30_days': days >= 30,
                                     'status': 'insufficient_14_plus_1' if days < 15 else 'ready_with_imputation_audit',
                                     'observed_target_rows': int(mask.hvac_observed.sum()),
                                     'imputed_target_rows': int((~mask.hvac_observed).sum()),
                                     'sha256': sha256_file(stage / output)})
        if len(rows) != 32 or len({row['output'] for row in rows}) != 32:
            raise AssertionError('必须输出32个唯一场景')
        if any(sha256_file(root / path) != digest for path, digest in hashes.items()):
            raise ValueError('选窗期间填补输入改变，拒绝发布')
        audit_dir = stage / 'analysis/forecast_windows'
        pd.DataFrame(rows).to_csv(audit_dir / 'windows.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(fold_rows, columns=['output', 'origin', 'train_start', 'eligibility_known_before_origin',
                                      'observed_target_test_rows']).to_csv(audit_dir / 'folds_14_1.csv', index=False, encoding='utf-8-sig')
        audit = {'recent_start': str(pd.Timestamp(recent_start)), 'selection': 'longest full-day contiguous window; ties favor latest',
                 'with_without_it': 'independent windows; metrics are not a controlled IT ablation without matched test origins',
                 'history_days_include_test': True, 'imputation_manifest_sha256': sha256_file(manifest_path),
                 'code_sha256': sha256_file(Path(__file__)), 'windows': rows,
                 'warning': 'imputed targets are estimates, not observed scoring truth; inspect masks and raw dependency provenance before modelling'}
        (audit_dir / 'manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
        publish_stage(stage, root)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recent-start', default=None)
    args = parser.parse_args()
    rows = export_windows(args.root, args.recent_start)
    print(pd.DataFrame(rows)[['building', 'with_it', 'start', 'end', 'days', 'folds_14_1']].drop_duplicates().to_string(index=False))
    print(f'导出完成: {len(rows)} 个预测场景 CSV')


if __name__ == '__main__':
    main()
