# -*- coding: utf-8 -*-
"""AIDC 点位因果短缺口填补：选型和数值只使用缺口之前的原始观测。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from migrate_hvac_data import BUILDINGS, DEFAULT_ROOT, FAMILIES, FILES, sha256_file

RECIPE = Path(__file__).resolve().parents[1] / 'preparation.json'
STEP = pd.Timedelta(minutes=5)
MAX_GAP = 72
BUCKETS = (1, 3, 12, 36, 72)
METHODS = ('locf', 'past_mean_1h', 'previous_day')
LOOKBACKS = (1, 12, 288)
MIN_VALIDATION = 3
CALIBRATION_DAYS = 30


def true_runs(mask):
    """返回半开区间 [start, stop)，不分割超长缺口。"""
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def candidate_predictions(values, starts, length):
    """每行独立遮蔽；预测只读 starts 之前，且不读历史补值。"""
    starts = np.asarray(starts, dtype=int)
    result = np.full((len(starts), len(METHODS), length), np.nan)
    for method, lookback in enumerate(LOOKBACKS):
        valid = starts >= lookback
        origins = starts[valid]
        if method == 0:
            pred = np.repeat(values[origins - 1, None], length, axis=1)
        elif method == 1:
            context = values[origins[:, None] - np.arange(1, 13)]
            pred = np.repeat(context.mean(axis=1)[:, None], length, axis=1)
        else:
            pred = values[origins[:, None] - 288 + np.arange(length)]
        result[valid, method, :] = pred
    return result


def validation_blocks(values, length):
    """6h 步长的原观测遮蔽块；最终按缺口时间截断，不使用未来评分。"""
    starts = np.arange(72, len(values) - length + 1, 72)
    truth = values[starts[:, None] + np.arange(length)]
    predictions = candidate_predictions(values, starts, length)
    errors = np.abs(predictions - truth[:, None, :]).mean(axis=2)
    return starts, errors


def impute_series(source):
    if not isinstance(source.index, pd.DatetimeIndex) or not source.index.is_unique:
        raise ValueError('需要唯一的 DatetimeIndex')
    if len(source) == 0 or not source.index.equals(pd.date_range(source.index[0], periods=len(source), freq=STEP)):
        raise ValueError('时间必须为升序连续5min网格')
    values = source.to_numpy(dtype=float, copy=True)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError('负荷不允许负数或无限值')
    output = values.copy()
    audits = []
    validations = {}
    for start, stop in true_runs(np.isnan(values)):
        length = int(stop - start)
        row = {'gap_start': str(source.index[start]), 'gap_end': str(source.index[stop - 1]),
               'gap_points': length, 'status': '', 'method': None,
               'validation_count': 0, 'validation_start': None, 'validation_end': None,
               'raw_dependency_start': None, 'raw_dependency_end': None,
               'eligibility_known_at': str(source.index[stop]) if stop < len(values) else None}
        if start == 0 or stop == len(values):
            row['status'] = 'unfilled_edge'
        elif length > MAX_GAP:
            row['status'] = 'unfilled_long'
        else:
            bucket = next(size for size in BUCKETS if size >= length)
            row['validation_bucket_points'] = bucket
            predictions = candidate_predictions(values, [start], length)[0]
            available = np.isfinite(predictions).all(axis=1)
            if bucket not in validations:
                validations[bucket] = validation_blocks(values, bucket)
            origins, errors = validations[bucket]
            valid = ((origins + bucket <= start)
                     & (origins >= start - CALIBRATION_DAYS * 288)
                     & np.isfinite(errors[:, available]).all(axis=1))
            count = int(valid.sum())
            row['validation_count'] = count
            if count < MIN_VALIDATION or not available.any():
                row['status'] = 'unfilled_insufficient_validation'
            else:
                scores = np.full(len(METHODS), np.inf)
                scores[available] = np.median(errors[valid][:, available], axis=0)
                winner = int(np.argmin(scores))
                output[start:stop] = predictions[winner]
                row.update(status='filled', method=METHODS[winner],
                           validation_start=str(source.index[origins[valid][0]]),
                           validation_end=str(source.index[origins[valid][-1] + bucket - 1]),
                           raw_dependency_start=str(source.index[max(0, min(
                               origins[valid][0] - max(np.asarray(LOOKBACKS)[available]),
                               start - LOOKBACKS[winner]))]),
                           raw_dependency_end=str(source.index[start - 1]))
                for i, method in enumerate(METHODS):
                    row[method + '_median_mae'] = float(scores[i]) if np.isfinite(scores[i]) else None
        audits.append(row)
    return pd.Series(output, index=source.index, name=source.name), audits


def impute_table(source, excluded=()):
    """排除经授权不存在的点位；严格总量与逐时刻审计分开存放。"""
    points = source.drop(columns='total_load')
    absent = set(points.columns[points.isna().all()])
    if absent != set(excluded):
        raise ValueError(f'全空点位与明确排除清单不一致: actual={absent}, expected={set(excluded)}')
    points = points.drop(columns=list(excluded))
    if points.empty:
        raise ValueError('没有有效点位')
    result = points.copy()
    audits = []
    known_at = points.index.to_numpy().copy()
    for point in points:
        filled, rows = impute_series(points[point])
        result[point] = filled
        for row in rows:
            row['point'] = point
            if row['status'] == 'filled':
                start = points.index.get_loc(pd.Timestamp(row['gap_start']))
                stop = start + row['gap_points']
                known_at[start:stop] = np.maximum(
                    known_at[start:stop], np.datetime64(row['eligibility_known_at']))
        audits.extend(rows)
    result['total_load'] = result.sum(axis=1, min_count=len(points.columns))
    observed = points.notna().all(axis=1)
    mask = pd.DataFrame({'total_observed': observed,
                         'total_imputed': ~observed & result.total_load.notna(),
                         'eligibility_known_at': known_at}, index=source.index)
    mask.loc[result.total_load.isna(), 'eligibility_known_at'] = pd.NaT
    return result, audits, mask


def read_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)) or header[0] != 'time' or 'total_load' not in header:
        raise ValueError(f'非法列名: {path}')
    frame = pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.equals(
            pd.date_range(frame.index[0], periods=len(frame), freq=STEP)):
        raise ValueError(f'非法5min时间网格: {path}')
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError(f'负荷中存在非法数值: {path}')
    return frame.astype(float)


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, encoding='utf-8-sig', index_label='time')


def publish_stage(stage, root):
    """先检查全部冲突；只发布新文件，不覆盖已有结果。"""
    paths = sorted(p for p in stage.rglob('*') if p.is_file())
    for path in paths:
        if (root / path.relative_to(stage)).exists():
            raise FileExistsError(root / path.relative_to(stage))
    for path in paths:
        destination = root / path.relative_to(stage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)


def impute_dataset(root=DEFAULT_ROOT, excluded_it_points=None):
    root = Path(root).resolve()
    if excluded_it_points is None:
        excluded_it_points = json.loads(RECIPE.read_text(encoding='utf-8'))['excluded_it_points']
    if set(excluded_it_points) - {'A1', 'A3'}:
        raise ValueError('只允许排除明确授权的 A1/A3 IT 点位')
    destinations = [root / 'imputed_data' / family / name for family in FAMILIES for name in FILES]
    audit_dir = root / 'analysis/imputation'
    if any(p.exists() for p in destinations) or audit_dir.exists():
        raise FileExistsError('填补产物或审计目录已经存在，拒绝覆盖')
    sources = {f'{family}/{name}': root / 'raw_data' / family / name for family in FAMILIES for name in FILES}
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    # 校验全部原始表与分量关系，再开始计算；不以部分求和证明完整性。
    for family in FAMILIES:
        parts = {}
        for building in BUILDINGS:
            frame = read_table(sources[f'{family}/{building}_data.csv'])
            pts = frame.drop(columns='total_load')
            excluded = excluded_it_points.get(building, []) if family == 'IT_load' else []
            if set(pts.columns[pts.isna().all()]) != set(excluded):
                raise ValueError(f'未经授权的全空点位: {family}/{building}')
            if not np.allclose(pts.sum(axis=1, min_count=1), frame.total_load, equal_nan=True):
                raise ValueError(f'原始总量不符合原生成规则: {family}/{building}')
            parts[building] = pts.add_prefix(building + '_')
        combined = read_table(sources[f'{family}/data.csv'])
        expected = pd.concat(parts.values(), axis=1).reindex(combined.index)
        pd.testing.assert_frame_equal(combined.drop(columns='total_load'), expected)
        if not np.allclose(expected.sum(axis=1, min_count=1), combined.total_load, equal_nan=True):
            raise ValueError(f'原始三楼总量不一致: {family}')
    inventory = {'files': [], 'excluded_it_points': excluded_it_points,
                 'methods': list(METHODS), 'max_gap_points': MAX_GAP,
                 'calibration_days': CALIBRATION_DAYS, 'validation_minimum': MIN_VALIDATION,
                 'validation_buckets': list(BUCKETS), 'selection_metric': 'median_mask_block_mae',
                 'causality': 'values and selection use raw observations before each gap; offline length eligibility known only at gap closure',
                 'raw_dependency_note': 'calibration and seasonal context can extend beyond a model training window',
                 'total_policy': 'all included points required; missing component makes total NaN',
                 'code_sha256': sha256_file(Path(__file__)), 'source_sha256': hashes}
    with tempfile.TemporaryDirectory(prefix='.imputation-stage-', dir=root) as tmp:
        stage = Path(tmp)
        for family in FAMILIES:
            parts, masks = {}, {}
            for building in BUILDINGS:
                key = f'{family}/{building}_data.csv'
                source = read_table(sources[key])
                excluded = excluded_it_points.get(building, []) if family == 'IT_load' else []
                result, audit, mask = impute_table(source, excluded)
                parts[building], masks[building] = result.drop(columns='total_load'), mask
                output = stage / 'imputed_data' / key
                write_csv(result, output)
                write_csv(mask, stage / 'analysis/imputation/masks' / key)
                audit_path = stage / 'analysis/imputation/gaps' / key
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                columns = ['point', 'gap_start', 'gap_end', 'gap_points', 'status', 'method',
                           'validation_bucket_points', 'validation_count', 'validation_start', 'validation_end',
                           'raw_dependency_start', 'raw_dependency_end', 'eligibility_known_at',
                           *(method + '_median_mae' for method in METHODS)]
                pd.DataFrame(audit, columns=columns).to_csv(audit_path, index=False, encoding='utf-8-sig')
                filled_count = int((result.drop(columns='total_load').notna() & source[result.columns[:-1]].isna()).sum().sum())
                inventory['files'].append({'output': f'imputed_data/{key}', 'rows': len(result),
                                           'points': len(result.columns) - 1, 'filled_cells': filled_count,
                                           'sha256': sha256_file(output)})
                print(f'{key}: filled_cells={filled_count}, complete_totals={result.total_load.notna().sum()}', flush=True)
            key = f'{family}/data.csv'
            grid = read_table(sources[key]).index
            combined = pd.concat([parts[b].add_prefix(b + '_') for b in BUILDINGS], axis=1).reindex(grid)
            combined['total_load'] = combined.sum(axis=1, min_count=len(combined.columns))
            observed = pd.concat([masks[b].total_observed.reindex(grid, fill_value=False) for b in BUILDINGS], axis=1).all(axis=1)
            known = pd.concat([masks[b].eligibility_known_at.reindex(grid) for b in BUILDINGS], axis=1).max(axis=1)
            mask = pd.DataFrame({'total_observed': observed, 'total_imputed': ~observed & combined.total_load.notna(),
                                 'eligibility_known_at': known}, index=grid)
            mask.loc[combined.total_load.isna(), 'eligibility_known_at'] = pd.NaT
            output = stage / 'imputed_data' / key
            write_csv(combined, output)
            write_csv(mask, stage / 'analysis/imputation/masks' / key)
            inventory['files'].append({'output': f'imputed_data/{key}', 'rows': len(combined),
                                       'points': len(combined.columns) - 1, 'sha256': sha256_file(output),
                                       'rebuilt_from_buildings': True})
        if {key: sha256_file(path) for key, path in sources.items()} != hashes:
            raise ValueError('处理过程中原始输入改变，拒绝发布')
        manifest = stage / 'analysis/imputation/manifest.json'
        manifest.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
        publish_stage(stage, root)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = impute_dataset(args.root)
    print(f"填补完成: {len(result['files'])} 个 CSV；raw_data 保持不变")


if __name__ == '__main__':
    main()
