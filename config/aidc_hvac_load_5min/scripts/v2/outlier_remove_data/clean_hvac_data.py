"""data_v2 前置异常屏蔽：只读raw，合并孤立点和人工红框，不进行填补。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from config.aidc_hvac_load_5min.scripts.outlier_remove_data.outlier_detection import detect_isolated, detect_candidates
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import read_table, write_csv
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import BUILDINGS, ROUTES, VERSIONS, FAMILIES, FILES, DEFAULT_ROOT, sha256_file

RECIPE = Path(__file__).with_name('cleaning.json')
POINT_COLUMNS = ['time', 'point', 'old_value', 'rule', 'detection_known_at']


def mask_family(full, reduced, start, end, events, recipe):
    """同一原始快照上识别孤立点，再在屏蔽后观测上判断红框；共享点只判一次。"""
    pd.testing.assert_frame_equal(full[reduced.columns], reduced)
    candidates = pd.concat([detect_isolated(table, start, end).assign(device_version=version)
                            for table, version in zip((full, reduced), VERSIONS)], ignore_index=True)
    accepted = candidates.loc[candidates.status.eq('accepted')].drop_duplicates(['time', 'point'])
    masked = full.copy()
    rows, screening = [], []
    for row in accepted.itertuples():
        stamp = pd.Timestamp(row.time)
        rows.append({'time': row.time, 'point': row.point, 'old_value': full.loc[stamp, row.point],
                     'rule': 'isolated', 'detection_known_at': row.detection_known_at})
        masked.loc[stamp, row.point] = np.nan
    for point in masked:
        selected, audit = detect_candidates(masked[point], events, recipe)
        screening.extend({'point': point, **row} for row in audit)
        for stamp in masked.index[selected]:
            rows.append({'time': str(stamp), 'point': point, 'old_value': full.loc[stamp, point],
                         'rule': 'redbox', 'detection_known_at': None})
        masked.loc[selected, point] = np.nan
    return masked, pd.DataFrame(rows, columns=POINT_COLUMNS), candidates, screening


def build_cleaned(root=DEFAULT_ROOT, recipe_path=RECIPE, *, output_root=None):
    root = Path(root).resolve()
    output_root = Path(output_root).resolve() if output_root is not None else root
    recipe = json.loads(Path(recipe_path).read_text())
    destination = output_root / 'outlier_remove_data/data_v2'
    audit_dir = output_root / 'analysis/data_v2/outliers'
    if destination.exists() or audit_dir.exists():
        raise FileExistsError('data_v2 异常屏蔽数据或审计已存在，拒绝覆盖')
    source_hashes = {f'{f}/{n}': sha256_file(root / 'raw_data' / f / n) for f in FAMILIES for n in FILES}
    points, candidates, screening, masked_tables = [], [], [], {}
    for building in BUILDINGS:
        scope = recipe['isolated_windows'][building]
        for route in ROUTES:
            full = read_table(root / f'raw_data/hvac_all_devices/{route}/{building}_data.csv')
            reduced = read_table(root / f'raw_data/hvac_remove_devices/{route}/{building}_data.csv')
            events = [e for e in recipe['events'] if e['building'] == building and route[-1] in e['routes']]
            masked, rows, detected, audits = mask_family(full.drop(columns='total_load'), reduced.drop(columns='total_load'),
                                                       scope['start'], scope['end'], events, recipe)
            points.append(rows.assign(building=building, route=route))
            candidates.append(detected.assign(building=building, route=route))
            screening.extend({'building': building, 'route': route, **row} for row in audits)
            for version, original in zip(VERSIONS, (full, reduced)):
                result = masked[original.columns[:-1]].copy()
                result['total_load'] = result.sum(axis=1, min_count=1)
                masked_tables[f'{version}/{route}/{building}_data.csv'] = result
    # 原始分楼/三楼合同独立核对；IT不清洗，全空登记点仍留给填补层的显式排除名单。
    for family in FAMILIES:
        original_parts, parts = [], []
        for building in BUILDINGS:
            key = f'{family}/{building}_data.csv'
            raw = read_table(root / 'raw_data' / key)
            raw_points = raw.drop(columns='total_load')
            np.testing.assert_allclose(raw_points.sum(axis=1, min_count=1), raw.total_load, equal_nan=True)
            original_parts.append(raw_points.add_prefix(building + '_'))
            result = masked_tables.get(key, raw)
            parts.append(result.drop(columns='total_load').add_prefix(building + '_'))
            write_csv(result, destination / key)
        raw_combined = read_table(root / 'raw_data' / family / 'data.csv')
        expected = pd.concat(original_parts, axis=1).reindex(raw_combined.index)
        pd.testing.assert_frame_equal(expected, raw_combined.drop(columns='total_load'))
        np.testing.assert_allclose(expected.sum(axis=1, min_count=1), raw_combined.total_load, equal_nan=True)
        combined = pd.concat(parts, axis=1).reindex(raw_combined.index)
        combined['total_load'] = combined.sum(axis=1, min_count=1)
        write_csv(combined, destination / family / 'data.csv')
    audit_dir.mkdir(parents=True)
    merged = pd.concat(points, ignore_index=True)
    if merged.duplicated(['building', 'route', 'point', 'time']).any():
        raise AssertionError('异常物理点位重复')
    merged.to_csv(audit_dir / 'anomaly_points.csv', index=False)
    pd.concat(candidates, ignore_index=True).to_csv(audit_dir / 'isolated_candidates.csv', index=False)
    pd.DataFrame(screening).to_csv(audit_dir / 'redbox_screening.csv', index=False)
    if any(sha256_file(root / 'raw_data' / key) != digest for key, digest in source_hashes.items()):
        raise ValueError('原始源在处理期间变化')
    manifest = {'data_version': 'data_v2', 'recipe': recipe, 'source_sha256': source_hashes,
                'output_sha256': {str(p.relative_to(output_root)): sha256_file(p) for p in destination.rglob('*.csv')},
                'physical_anomaly_points': len(merged), 'stage': 'mask_only_before_imputation',
                'causality': 'OFFLINE: centered isolated detector uses future 15min; red boxes are manual offline decisions. Not deployable.',
                'code_sha256': sha256_file(Path(__file__)), 'recipe_sha256': sha256_file(recipe_path)}
    (audit_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'前置屏蔽完成: {len(merged)} 个物理点位时刻；未填补', flush=True)
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    args = parser.parse_args()
    build_cleaned(args.root, args.recipe)
