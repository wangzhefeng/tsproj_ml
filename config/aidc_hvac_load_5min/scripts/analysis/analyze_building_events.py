"""A2/A3需求响应候选筛查；只输出证据，绝不据曲线自动确认调度事件。"""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import read_table
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file


def candidates(totals):
    """探索阈值：双口径跌30%且至少20kW，持续3槽；6h内双口径稳定恢复6槽。"""
    baseline = totals.shift(1).rolling(12, min_periods=8).median()
    starts = ((totals <= baseline * .7) & ((baseline - totals) >= 20)).all(axis=1)
    rows, covered = [], -1
    for i in np.flatnonzero(starts.to_numpy()):
        if i <= covered or i < 12 or i + 3 >= len(totals):
            continue
        base = baseline.iloc[i].to_numpy()
        if not np.isfinite(base).all() or (base <= 0).any():
            continue
        window = totals.iloc[i:min(i + 73, len(totals))].to_numpy()
        if not (window[:3] <= .8 * base).all():
            continue
        within = ((window >= .9 * base) & (window <= 1.1 * base)).all(axis=1)
        stable = np.convolve(within.astype(int), np.ones(6, dtype=int), mode='valid')
        recovered = np.flatnonzero(stable == 6)
        end_offset = int(recovered[0]) if len(recovered) else None
        rebound = np.flatnonzero((window >= .9 * base).all(axis=1))
        covered = i + (end_offset if end_offset is not None else len(window) - 1)
        rows.append({'start': str(totals.index[i]),
                     'rebound_start': str(totals.index[i + int(rebound[0])]) if len(rebound) else None,
                     'end_exclusive': str(totals.index[i + end_offset]) if end_offset is not None else None,
                     'reference_all_kw': float(base[0]), 'reference_reduced_kw': float(base[1]),
                     'minimum_all_kw': float(np.nanmin(window[:, 0])),
                     'status': 'candidate_requires_confirmation' if end_offset is not None else 'no_recovery_within_6h',
                     'approved': False})
    return rows


def analyze(root):
    root = Path(root)
    rows, hashes = [], {}
    for building in ('A2', 'A3'):
        series = {}
        for device in ('hvac_all_devices', 'hvac_remove_devices'):
            routes = []
            for route in ('route_A', 'route_B'):
                rel = f'raw_data/{device}/{route}/{building}_data.csv'
                frame = read_table(root / rel).drop(columns='total_load')
                routes.append(frame)
                hashes[rel] = sha256_file(root / rel)
            points = pd.concat(routes, axis=1)
            series[device] = points.sum(axis=1, min_count=len(points.columns))
        rows.extend({'building': building, **r} for r in candidates(pd.DataFrame(series)))
    output = root / 'analysis/building_extension'
    output.mkdir(parents=True, exist_ok=True)
    dest = output / 'event_candidates.csv'
    if dest.exists():
        raise FileExistsError(dest)
    pd.DataFrame(rows).to_csv(dest, index=False)
    (output / 'event_candidates.meta.json').write_text(json.dumps({'source_sha256': hashes,
        'criteria': 'dual-scope drop>=30% and >=20kW for 3 slots; pre-1h median >=8 observed; recovery +/-10% for 6 slots within 6h',
        'automatic_correction': False, 'candidates': len(rows)}, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'dataset/aidc_hvac_load_5min')
    analyze(parser.parse_args().root)
