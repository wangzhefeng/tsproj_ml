# -*- coding: utf-8 -*-
"""从 imputed_data 选择近期最长完整日窗口，导出16个目标/32个有无IT场景。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from impute_hvac_data import RECIPE, STEP, read_table, true_runs, write_csv
from migrate_hvac_data import BUILDINGS, DEFAULT_ROOT, FAMILIES, FILES, ROUTES, VERSIONS, sha256_file
from forecast_schema import SCHEMA, column_sources, target_column


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


def assemble_forecast(totals, masks, version, building, with_it, index):
    """两路始终按同一时间轴对齐；每列mask来自自己的源，AB使用两路交集。"""
    columns, provenance = {}, {}
    for column, keys in column_sources(version, building, with_it).items():
        values = pd.concat([totals[key].reindex(index) for key in keys], axis=1)
        columns[column] = values.iloc[:, 0] if len(keys) == 1 else values.sum(axis=1, min_count=len(keys))
        observed = pd.concat([masks[key].total_observed.reindex(index) for key in keys], axis=1)
        available = pd.concat([masks[key].eligibility_known_at.reindex(index) for key in keys], axis=1)
        provenance[column + '__observed'] = observed.eq(True).all(axis=1)
        provenance[column + '__eligibility_known_at'] = available.max(axis=1).where(available.notna().all(axis=1))
    frame = pd.DataFrame(columns, index=index)
    audit = pd.DataFrame(provenance, index=index)
    availability = audit[[c + '__eligibility_known_at' for c in frame.columns]]
    audit['eligibility_known_at'] = availability.max(axis=1).where(availability.notna().all(axis=1))
    return frame, audit


def publish_prepared_directories(stage, root, names, *, replace=False):
    """只发布明确的派生目录。成功后丢弃旧版；发布失败回滚，不触碰raw/imputed。"""
    allowed = {'forecast_data', 'analysis/forecast_windows', 'analysis/forecast_data_visual'}
    if not names or len(set(names)) != len(names) or not set(names) <= allowed:
        raise ValueError('发布范围只允许预测数据及其选窗/可视化审计目录')
    for name in names:
        source, destination = stage / name, root / name
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f'暂存产物缺失或非法: {source}')
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError(f'目标不是普通目录: {destination}')
        if destination.exists() and not replace and any(p.is_file() for p in destination.rglob('*')):
            raise FileExistsError(f'拒绝覆盖已有派生数据: {destination}')
    backup = Path(tempfile.mkdtemp(prefix='.forecast-publish-', dir=root))
    moved_old, moved_new = [], []
    try:
        for name in names:
            destination = root / name
            if destination.exists():
                previous = backup / name
                previous.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(previous)
                moved_old.append(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / name).rename(destination)
            moved_new.append(name)
    except BaseException:
        # 回滚本身失败时保留备份目录，由异常报告其路径，禁止自动删除旧版。
        try:
            for name in reversed(moved_new):
                (root / name).rename(stage / name)
            for name in reversed(moved_old):
                (backup / name).rename(root / name)
        except BaseException as rollback_error:
            raise RuntimeError(f'发布及回滚失败，旧数据恢复目录: {backup}') from rollback_error
        shutil.rmtree(backup)
        raise
    shutil.rmtree(backup)


def export_windows(root=DEFAULT_ROOT, recent_start=None, *, replace=False):
    root = Path(root).resolve()
    if recent_start is None:
        recent_start = json.loads(RECIPE.read_text(encoding='utf-8'))['recent_start']
    if not replace and ((root / 'analysis/forecast_windows').exists() or any((root / 'forecast_data').rglob('*.csv'))):
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
                        index = pd.date_range(start, stop - STEP, freq=STEP, name='time')
                        frame, mask = assemble_forecast(totals, masks, version, building, with_it, index)
                        target = target_column(route)
                        if frame.isna().any().any() or len(frame) != days * 288:
                            raise ValueError(f'选窗包含缺失或不完整天: {family}/{name}')
                        export_name = name.removesuffix('.csv') + ('_with_it.csv' if with_it else '.csv')
                        output = f'forecast_data/{family}/{export_name}'
                        write_csv(frame, stage / output)
                        if mask.eligibility_known_at.isna().any():
                            raise ValueError(f'完整预测表存在缺失的可得性证据: {output}')
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
                                              'target_column': target,
                                              'observed_target_test_rows': int(test[target + '__observed'].sum())})
                        rows.append({'output': output, 'version': version, 'route': route,
                                     'schema': SCHEMA, 'target_column': target,
                                     'columns_json': json.dumps(list(frame.columns)),
                                     'history_only_columns_json': json.dumps([c for c in frame.columns if c != target]),
                                     'column_sources_json': json.dumps(column_sources(version, building, with_it)),
                                     'building': building, 'with_it': with_it, 'start': str(start),
                                     'end': str(stop - STEP), 'days': days, 'rows': len(frame),
                                     'folds_14_1': max(0, days - 14), 'eligibility_safe_folds_14_1': causal_folds,
                                     'meets_30_days': days >= 30,
                                     'status': 'insufficient_14_plus_1' if days < 15 else 'ready_with_imputation_audit',
                                     'observed_target_rows': int(mask[target + '__observed'].sum()),
                                     'imputed_target_rows': int((~mask[target + '__observed']).sum()),
                                     'sha256': sha256_file(stage / output)})
        if len(rows) != 32 or len({row['output'] for row in rows}) != 32:
            raise AssertionError('必须输出32个唯一场景')
        if any(sha256_file(root / path) != digest for path, digest in hashes.items()):
            raise ValueError('选窗期间填补输入改变，拒绝发布')
        audit_dir = stage / 'analysis/forecast_windows'
        pd.DataFrame(rows).to_csv(audit_dir / 'windows.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(fold_rows, columns=['output', 'origin', 'train_start', 'eligibility_known_before_origin',
                                      'target_column', 'observed_target_test_rows']).to_csv(audit_dir / 'folds_14_1.csv', index=False, encoding='utf-8-sig')
        audit = {'schema': SCHEMA, 'recent_start': str(pd.Timestamp(recent_start)), 'selection': 'longest full-day contiguous window; ties favor latest',
                 'with_without_it': 'independent windows; metrics are not a controlled IT ablation without matched test origins',
                 'history_days_include_test': True, 'imputation_manifest_sha256': sha256_file(manifest_path),
                 'code_sha256': sha256_file(Path(__file__)), 'windows': rows,
                 'schema_code_sha256': sha256_file(Path(__file__).with_name('forecast_schema.py')),
                 'routes': 'identical joint A/B inputs; route directory specifies target_column, not column availability',
                 'cross_route_semantics': 'opposite route, component and AB-sum observations are historical inputs only, never known future',
                 'warning': 'imputed targets are estimates, not observed scoring truth; inspect masks and raw dependency provenance before modelling'}
        (audit_dir / 'manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
        publish_prepared_directories(stage, root, ('forecast_data', 'analysis/forecast_windows'), replace=replace)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recent-start', default=None)
    parser.add_argument('--replace', action='store_true', help='明确覆盖已有预测表和选窗审计，不保留旧版本；源层不变')
    args = parser.parse_args()
    rows = export_windows(args.root, args.recent_start, replace=args.replace)
    print(pd.DataFrame(rows)[['building', 'with_it', 'start', 'end', 'days', 'folds_14_1']].drop_duplicates().to_string(index=False))
    print(f'导出完成: {len(rows)} 个预测场景 CSV')


if __name__ == '__main__':
    main()
