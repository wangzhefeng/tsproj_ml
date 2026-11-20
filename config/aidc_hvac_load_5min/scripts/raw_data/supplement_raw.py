"""Conservative external-observation supplement; never rebuild downstream data.

Conflicting points are deferred in their entirety. No interpolation, route
remapping or overwrite of observed point values. Raw partial-sum totals are
recomputed, unlike the strict totals used by downstream prepared data.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
    if header[0] != 'time' or len(header) != len(set(header)):
        raise ValueError(f'Invalid header: {path}')
    frame = pd.read_csv(path, index_col='time', parse_dates=['time'], float_precision='round_trip')
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f'Invalid timestamps: {path}')
    if not frame.index.equals(frame.index.floor('5min')):
        raise ValueError(f'Off-grid timestamps: {path}')
    frame = frame.apply(pd.to_numeric, errors='raise')
    if np.isinf(frame.to_numpy()).any():
        raise ValueError(f'Infinite value: {path}')
    return frame


def supplement_point(old, candidates, *, nonnegative):
    """Return series and decision; entire point deferred on overlap conflict."""
    result = old.copy()
    if not candidates:
        return result, 'no_source', 0
    combined = pd.concat(candidates, axis=1).reindex(old.index)
    low, high = combined.min(axis=1), combined.max(axis=1)
    different = low.notna() & high.notna() & ~np.isclose(low, high, rtol=1e-9, atol=1e-6)
    if different.any():
        return result, 'external_conflict', int(different.sum())
    new = combined.bfill(axis=1).iloc[:, 0]
    overlap = old.notna() & new.notna()
    conflict = overlap & ~np.isclose(old, new, rtol=1e-9, atol=1e-6)
    if conflict.any():
        return result, 'overlap_conflict', int(conflict.sum())
    if nonnegative and (new < 0).any():
        return result, 'negative_power', 0
    result = old.fillna(new)
    return result, 'accepted' if (old.isna() & result.notna()).any() else 'no_new_values', 0


def dump(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def with_total(points):
    result = points.copy()
    result['total_load'] = points.sum(axis=1, min_count=1)
    return result


def build(data, source, audit, start='2025-10-01', end='2026-10-23 23:55', *, apply=False):
    data, source, audit = Path(data).resolve(), Path(source).resolve(), Path(audit).resolve()
    raw = data / 'raw_data'
    if audit.exists():
        raise FileExistsError(audit)
    if raw in audit.parents or source == audit or source in audit.parents:
        raise ValueError('Audit must be outside raw and external source')
    grid = pd.date_range(start, end, freq='5min', name='time')
    if not len(grid) or grid[0] != pd.Timestamp(start) or grid[-1] != pd.Timestamp(end):
        raise ValueError('Invalid requested grid')
    paths = sorted(raw.rglob('*.csv'))
    before = {str(p.relative_to(data)): sha(p) for p in paths}
    protected = {str(p.relative_to(data)): sha(p) for folder in ('imputed_data', 'outlier_remove_data', 'forecast_data') for p in sorted((data / folder).rglob('*')) if p.is_file()}
    external = {}
    for p in sorted(source.rglob('*.csv')):
        external.setdefault((p.parent.name, p.stem), []).append(p)
    audit.mkdir(parents=True)
    backup, stage = audit / 'before', audit / 'staged'
    for p in paths:
        target = backup / p.relative_to(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        if sha(target) != before[str(p.relative_to(data))]:
            raise ValueError('Backup mismatch')
    decisions, source_hashes, table_hashes = [], {}, {}
    primary = []
    for b in ('A1', 'A2', 'A3'):
        primary.append(('IT', b, Path('IT_load') / f'{b}_data.csv'))
        for route in ('route_A', 'route_B'):
            primary.append(('HVAC', b, Path('hvac_all_devices') / route / f'{b}_data.csv'))
    for kind in ('flow', 'temp'):
        primary.append((kind, 'A1', Path('hvac_flow_temp') / f'data_{kind}.csv'))
    parents = {}
    for kind, building, key in primary:
        original = read(backup / 'raw_data' / key)
        if original.index.min() < grid[0] or original.index.max() > grid[-1]:
            raise ValueError('Requested window would drop original timestamps')
        points = original.drop(columns='total_load', errors='ignore').reindex(grid)
        first_add = None
        added_count = 0
        for point in points:
            candidates, names = [], []
            for p in external.get((building, point), []):
                digest = sha(p)
                candidate = read(p)
                if list(candidate) != ['value'] or sha(p) != digest:
                    raise ValueError(f'Invalid or changing source: {p}')
                source_hashes[str(p)] = digest
                names.append(str(p))
                candidates.append(candidate['value'].reindex(grid))
            old = points[point].copy()
            new, status, conflicts = supplement_point(old, candidates, nonnegative=kind in ('IT', 'HVAC'))
            added = old.isna() & new.notna()
            points[point] = new
            n = int(added.sum())
            added_count += n
            times = grid[added]
            if len(times):
                first_add = times.min() if first_add is None else min(first_add, times.min())
            decisions.append(dict(kind=kind, building=building, point=point, sources=';'.join(names), status=status, conflicts=conflicts, added=n,
                                  pre=int((added & (grid < original.index.min())).sum()),
                                  internal=int((added & (grid >= original.index.min()) & (grid <= original.index.max())).sum()),
                                  post=int((added & (grid > original.index.max())).sum())))
        if added_count:
            lower = min(original.index.min(), first_add)
            points = points.loc[lower:]
        else:
            points = points.reindex(original.index)
        parents[str(key)] = points
        print('Prepared', key, added_count, flush=True)
    def emit(key, frame):
        key = Path(key)
        old = read(backup / 'raw_data' / key)
        if list(frame) != list(old) or not old.index.isin(frame.index).all():
            raise ValueError(f'Schema/time loss: {key}')
        cols = old.columns.drop('total_load', errors='ignore')
        old_values = old[cols].to_numpy()
        actual = frame.reindex(old.index)[cols].to_numpy()
        observed = ~np.isnan(old_values)
        np.testing.assert_array_equal(actual[observed], old_values[observed])
        target = stage / 'raw_data' / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if frame.equals(old):
            shutil.copy2(backup / 'raw_data' / key, target)
        else:
            frame.to_csv(target, encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M:%S')
        pd.testing.assert_frame_equal(read(target), frame, check_exact=True, check_freq=False)
        table_hashes[str(Path('raw_data') / key)] = sha(target)
    # Project from unique full-device point tables using original column membership.
    families = ['IT_load'] + [f'{d}/{r}' for d in ('hvac_all_devices', 'hvac_remove_devices') for r in ('route_A', 'route_B')]
    for family in families:
        parts = []
        for b in ('A1', 'A2', 'A3'):
            key = Path(family) / f'{b}_data.csv'
            cols = read(backup / 'raw_data' / key).columns.drop('total_load')
            parent_key = str(key).replace('hvac_remove_devices', 'hvac_all_devices')
            points = parents[parent_key][cols]
            emit(key, with_total(points))
            parts.append(points.add_prefix(b + '_'))
        total_key = Path(family) / 'data.csv'
        total_old = read(backup / 'raw_data' / total_key)
        lower = min(total_old.index.min(), *(p.index.min() for p in parts))
        upper = max(total_old.index.max(), *(p.index.max() for p in parts))
        total_grid = pd.date_range(lower, upper, freq='5min', name='time')
        combined = pd.concat([p.reindex(total_grid) for p in parts], axis=1)
        combined = combined[total_old.columns.drop('total_load')]
        emit(total_key, with_total(combined))
    for p in paths:
        key = p.relative_to(raw)
        if p.stem.count('_') == 2 and key.parts[0] in ('hvac_all_devices', 'hvac_remove_devices'):
            b = p.stem.split('_')[0]
            parent = parents[str(key.parent / f'{b}_data.csv').replace('hvac_remove_devices', 'hvac_all_devices')]
            cols = read(backup / 'raw_data' / key).columns
            emit(key, parent[cols])
    for kind in ('flow', 'temp'):
        key = f'hvac_flow_temp/data_{kind}.csv'
        emit(key, parents[key])
    if set(before) != set(table_hashes):
        raise ValueError('Incomplete raw table coverage')
    pd.DataFrame(decisions).to_csv(audit / 'points.csv', index=False)
    manifest = dict(status='prepared', start=str(grid[0]), end=str(grid[-1]), policy='defer_whole_conflicting_point; preserve_observed_points; raw_total_min_count=1',
                    before_sha256=before, after_sha256=table_hashes, protected_sha256=protected, source_sha256=source_hashes,
                    backup='before', builder_sha256=sha(__file__), downstream_status='historical snapshots; NOT rebuilt against supplemented raw',
                    added_cells=sum(x['added'] for x in decisions))
    dump(manifest, audit / 'manifest.json')
    for mapping in (before, protected):
        for key, digest in mapping.items():
            if sha(data / key) != digest:
                raise ValueError(f'Concurrent project change: {key}')
    for key, digest in source_hashes.items():
        if sha(key) != digest:
            raise ValueError(f'Concurrent source change: {key}')
    if apply:
        published = []
        try:
            for key, digest in table_hashes.items():
                if digest != before[key]:
                    published.append(key)
                    os.replace(stage / key, data / key)
            for key, digest in table_hashes.items():
                if sha(data / key) != digest:
                    raise ValueError(f'Publication mismatch: {key}')
            for key, digest in protected.items():
                if sha(data / key) != digest:
                    raise ValueError(f'Downstream changed: {key}')
        except BaseException:
            for key in published:
                shutil.copy2(backup / key, data / key)
            manifest['status'] = 'rolled_back'
            dump(manifest, audit / 'manifest.json')
            raise
        manifest['status'] = 'applied'
        dump(manifest, audit / 'manifest.json')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = build(args.data, args.source, args.audit, apply=args.apply)
    print(json.dumps({'status': result['status'], 'added_cells': result['added_cells']}))
