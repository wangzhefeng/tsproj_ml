"""A2/A3全时段点位准备和三楼汇总；不处理forecast_data。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import read_table, write_csv, publish_stage
from config.aidc_hvac_load_5min.scripts.imputed_data.gap_filling import fill_series
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')
FAMILIES = tuple(f'{d}/{r}' for d in DEVICES for r in ROUTES) + ('IT_load',)
DEFAULT_ROOT = ROOT / 'dataset/aidc_hvac_load_5min'
AUDIT = Path('analysis/building_extension')


def partial_fill(points):
    """保持全时间范围；无法按A1规则估计的缺口保留NaN并逐段记账。"""
    result = points.copy()
    rows = []
    for column in points:
        print('Fill', column, flush=True)
        result[column], audits = fill_series(points[column], output_start=points.index[0], on_unfillable='keep')
        rows.extend({**r, 'status': r.get('status', 'filled')} for r in audits)
    observed = points.notna()
    np.testing.assert_array_equal(result.to_numpy()[observed], points.to_numpy()[observed])
    result['total_load'] = result.sum(axis=1, min_count=len(points.columns))
    return result, rows


def aggregate(parts, grid):
    """沿用raw总表网格；缺任一点位则总量缺失，不填不存在的楼栋历史。"""
    if set(parts) != {'A1', 'A2', 'A3'}:
        raise ValueError('汇总必须包含三个楼栋')
    points = pd.concat([parts[b].drop(columns='total_load').add_prefix(b + '_').reindex(grid)
                        for b in ('A1', 'A2', 'A3')], axis=1)
    if not points.columns.is_unique:
        raise ValueError('重复点位')
    return points.assign(total_load=points.sum(axis=1, min_count=len(points.columns)))


def prepare(root):
    root = Path(root).resolve()
    expected = {f'raw_data/{f}/{b}_data.csv' for b in ('A2', 'A3') for f in FAMILIES}
    source_hashes = {p: sha256_file(root / p) for p in sorted(expected)}
    if (root / AUDIT / 'manifest.json').exists():
        raise FileExistsError('扩展处理已发布，拒绝覆盖')
    report = {'source_sha256': source_hashes, 'tables': {}, 'forecast_data_touched': False,
              'strict_online_eligible': False, 'unfillable_policy': 'keep_nan_with_audit',
              'anomaly_status': 'A2/A3 candidate events require separate approval; no event corrected',
              'it_scope': 'A1 existing fixed subset; A2/A3 all raw points including unresolved all-empty points'}
    with TemporaryDirectory(prefix='hvac-buildings-') as tmp:
        stage = Path(tmp)
        for building in ('A2', 'A3'):
            for family in ('hvac_all_devices/route_A', 'hvac_all_devices/route_B', 'IT_load'):
                key = f'{family}/{building}_data.csv'
                raw = read_table(root / 'raw_data' / key).drop(columns='total_load')
                filled, gaps = partial_fill(raw)
                gap_path = stage / AUDIT / key.replace('.csv', '.gaps.csv')
                gap_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(gaps).to_csv(gap_path, index=False)
                devices = DEVICES if family.startswith('hvac_') else ('IT_load',)
                for device in devices:
                    target_key = key.replace('hvac_all_devices', device) if family.startswith('hvac_') else key
                    source = read_table(root / 'raw_data' / target_key).drop(columns='total_load')
                    pd.testing.assert_frame_equal(raw[source.columns], source, check_freq=False)
                    table = filled[source.columns].copy()
                    table['total_load'] = table.sum(axis=1, min_count=len(source.columns))
                    # A1 semantics: imputed view is the baseline; outlier view adds only confirmed event corrections.
                    for category in ('imputed_data', 'outlier_remove_data'):
                        write_csv(table, stage / category / target_key)
                    observed = source.notna()
                    imputed = source.isna() & table[source.columns].notna()
                    unresolved = table[source.columns].isna()
                    for name, mask in [('observed', observed), ('imputed', imputed), ('unresolved', unresolved)]:
                        write_csv(mask, stage / AUDIT / target_key.replace('.csv', '.' + name + '.csv'))
                    report['tables'][target_key] = {'start': str(table.index[0]), 'end': str(table.index[-1]),
                        'rows': len(table), 'points': len(source.columns), 'filled_cells': int(imputed.sum().sum()),
                        'unresolved_cells': int(unresolved.sum().sum()), 'all_empty_points': list(source.columns[source.isna().all()])}
        a1_it = Path('outlier_remove_data/IT_load/A1_data.csv')
        if not (root / a1_it).exists():
            target = stage / a1_it
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / 'imputed_data/IT_load/A1_data.csv').read_bytes())
        for category in ('imputed_data', 'outlier_remove_data'):
            for family in FAMILIES:
                parts = {}
                for building in ('A1', 'A2', 'A3'):
                    rel = Path(category) / family / f'{building}_data.csv'
                    parts[building] = read_table(stage / rel if (stage / rel).exists() else root / rel)
                raw_total = read_table(root / 'raw_data' / family / 'data.csv')
                combined = aggregate(parts, raw_total.index)
                if not set(combined).issubset(raw_total.columns):
                    raise ValueError('汇总点位命名与原始总表不一致')
                write_csv(combined, stage / category / family / 'data.csv')
        for path, digest in source_hashes.items():
            if sha256_file(root / path) != digest:
                raise ValueError('处理期间源发生变化')
        report['output_sha256'] = {str(p.relative_to(stage)): sha256_file(p) for p in stage.rglob('*.csv')}
        target = stage / AUDIT / 'manifest.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10))
        publish_stage(stage, root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
