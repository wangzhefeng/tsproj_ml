"""已批准的套组归属迁移：暂存、逐点保真校验、备份发布；不运行模型。

build 只写独立工作目录；publish 仅发布核验过的 changed 文件，旧文件永久备份。
历史清洗/估计不重跑：逐点值及逐点缺口证据原样重分组，汇总掩码重新推导。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.raw_data.equipment_mapping import XLSX, load_mapping
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import read_table, write_csv
from config.aidc_hvac_load_5min.scripts.forecast_data.select_hvac_windows import export_windows

DEFAULT_ROOT = ROOT / 'dataset/aidc_hvac_load_5min'
DEFAULT_WORK = ROOT / '.hermes/plans/hvac-equipment-regroup'
DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')
BUILDINGS = ('A1', 'A2', 'A3')
ISO = 'outlier_remove_data/data_v1/isolated_v1'
RED = 'outlier_remove_data/data_v1/redbox_past_v2'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def regroup_pair(pair, mapping, *, strict):
    if not pair['route_A'].index.equals(pair['route_B'].index):
        raise ValueError('双路时间轴不同')
    points = pd.concat([pair[r].drop(columns='total_load') for r in ROUTES], axis=1)
    if not points.columns.is_unique or set(points) != set(mapping.spot_id):
        raise ValueError('双路点位并集与清单不一致')
    result = {}
    for route in ROUTES:
        columns = mapping.loc[mapping.group_route.eq(route[-1]), 'spot_id'].tolist()
        table = points[columns].copy()
        table['total_load'] = table.sum(axis=1, min_count=len(columns) if strict else 1)
        result[route] = table
        pd.testing.assert_frame_equal(table[columns], points[columns], check_exact=True)
    return result


def mask_from_gaps(table, gaps, corrections=None):
    """逐点缺口及已应用修正是掩码真值；不搬移旧路线总掩码。"""
    index = table.index
    columns = table.columns.drop('total_load')
    observed = np.ones((len(index), len(columns)), dtype=bool)
    known = index.to_numpy().copy()
    rows = gaps[gaps.point.isin(columns)]
    starts = index.searchsorted(pd.to_datetime(rows.gap_start))
    stops = index.searchsorted(pd.to_datetime(rows.gap_end), side='right')
    positions = columns.get_indexer(rows.point)
    eligibility = pd.to_datetime(rows.eligibility_known_at).to_numpy()
    filled = rows.status.eq('filled').to_numpy() if 'status' in rows else np.ones(len(rows), dtype=bool)
    if pd.isna(eligibility[filled]).any():
        raise ValueError('已填补缺口缺少资格时间')
    for start, stop, col, value, applied in zip(starts, stops, positions, eligibility, filled):
        observed[start:stop, col] = False
        if applied:
            known[start:stop] = np.maximum(known[start:stop], value)
    if corrections is not None and not corrections.empty:
        for row in corrections[corrections.point.isin(columns) & corrections.applied.eq(True)].to_dict('records'):
            stamp = pd.Timestamp(row['time'])
            if stamp in index:
                loc = index.get_loc(stamp)
                observed[loc, columns.get_loc(row['point'])] = False
                known[loc] = max(known[loc], np.datetime64(row['eligibility_known_at']))
    total_observed = observed.all(axis=1) & table.total_load.notna()
    mask = pd.DataFrame({'total_observed': total_observed,
                         'total_imputed': ~total_observed & table.total_load.notna(),
                         'eligibility_known_at': known}, index=index)
    mask.loc[table.total_load.isna(), 'eligibility_known_at'] = pd.NaT
    return mask


def read_mask(path):
    return pd.read_csv(path, index_col='time', parse_dates=['time', 'eligibility_known_at'])


def migrate_tables(stage, meta, table_rel, audit_rel=None, *, strict=True, v3=False, corrections_rel=None):
    print('Regroup', table_rel, flush=True)
    base = stage / table_rel
    corrections = pd.read_csv(stage / corrections_rel, float_precision='round_trip') if corrections_rel else None
    for version in DEVICES:
        selected = meta if version == DEVICES[0] else meta[meta['备注'].ne('冷冻水二次泵')]
        parts = {r: {} for r in ROUTES}
        masks = {r: {} for r in ROUTES}
        for building in BUILDINGS:
            name = f'{building}_data.csv'
            if not (base / version / ROUTES[0] / name).exists():
                if v3 and building != 'A1':
                    continue
                raise FileNotFoundError(base / version / ROUTES[0] / name)
            mapping = selected[selected.building.eq(building)]
            pair = {r: read_table(base / version / r / name) for r in ROUTES}
            gaps = None
            corr = None
            if audit_rel:
                gap_version = DEVICES[0] if v3 else version
                paths = [stage / audit_rel / (f'{gap_version}/{r}/{building}_data.gaps.csv' if v3 else f'gaps/{gap_version}/{r}/{name}') for r in ROUTES]
                gaps = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in paths], ignore_index=True)
                if corrections is not None:
                    corr = corrections[corrections.version.eq(version) & corrections.building.eq(building)]
                # 验证逐点审计足以重建迁移前掩码；失败不发布，不凭空猜来源。
                for route in ROUTES:
                    mask_path = stage / audit_rel / (f'{version}/{route}/{building}_data.mask.csv' if v3 else f'masks/{version}/{route}/{name}')
                    expected = read_mask(mask_path)
                    actual = mask_from_gaps(pair[route], gaps, corr)
                    pd.testing.assert_frame_equal(actual, expected, check_freq=False)
            outputs = regroup_pair(pair, mapping, strict=strict)
            for route, table in outputs.items():
                path = base / version / route / name
                write_csv(table, path)
                pd.testing.assert_frame_equal(read_table(path), table, check_exact=True, check_freq=False)
                parts[route][building] = table.drop(columns='total_load')
                if audit_rel:
                    mask = mask_from_gaps(table, gaps, corr)
                    masks[route][building] = mask
                    mask_path = stage / audit_rel / (f'{version}/{route}/{building}_data.mask.csv' if v3 else f'masks/{version}/{route}/{name}')
                    write_csv(mask, mask_path)
                    # v3仅全设备保留缺口明细；两个口径共享同一逐点估计。
                    if not v3 or version == DEVICES[0]:
                        gap_path = stage / audit_rel / (f'{version}/{route}/{building}_data.gaps.csv' if v3 else f'gaps/{version}/{route}/{name}')
                        gaps[gaps.point.isin(table.columns)].to_csv(gap_path, index=False, encoding='utf-8-sig')
            # 合并重分组结果后逐列精确对比原始点位；值与NaN掩码都必须保真。
            before = pd.concat([x.drop(columns='total_load') for x in pair.values()], axis=1)
            after = pd.concat([x.drop(columns='total_load') for x in outputs.values()], axis=1)
            pd.testing.assert_frame_equal(before, after[before.columns], check_exact=True)
        if not v3:
            for route in ROUTES:
                old = read_table(base / version / route / 'data.csv')
                combined = pd.concat([parts[route][b].add_prefix(b + '_') for b in BUILDINGS], axis=1).reindex(old.index)
                combined['total_load'] = combined.sum(axis=1, min_count=len(combined.columns) if strict else 1)
                write_csv(combined, base / version / route / 'data.csv')
                if audit_rel:
                    observed = pd.concat([masks[route][b].total_observed.reindex(old.index, fill_value=False) for b in BUILDINGS], axis=1).all(axis=1)
                    known = pd.concat([masks[route][b].eligibility_known_at.reindex(old.index) for b in BUILDINGS], axis=1).max(axis=1)
                    mask = pd.DataFrame({'total_observed': observed, 'total_imputed': ~observed & combined.total_load.notna(),
                                         'eligibility_known_at': known.where(combined.total_load.notna())}, index=old.index)
                    write_csv(mask, stage / audit_rel / 'masks' / version / route / 'data.csv')


def refresh_inventory(stage, owner, audit_rel):
    path = stage / owner / audit_rel / 'manifest.json'
    report = json.loads(path.read_text())
    for row in report['files']:
        target = stage / owner / row['output']
        if '/IT_load/' in row['output']:
            continue
        columns = pd.read_csv(target, nrows=0).columns
        row.update(sha256=sha256_file(target), points=len(columns) - 2)
        row.pop('filled_cells', None)
    report['masks_sha256'] = {str(p.relative_to(stage / owner)): sha256_file(p)
                              for p in (path.parent / 'masks').rglob('*.csv')}
    # 原执行代码SHA与历史输入事实不伪造为重跑证据。
    report['route_migration'] = {'mapping': 'analysis/equipment_regroup/mapping.csv',
                                 'semantics': 'group_route; frozen point estimates; no model or imputation rerun'}
    save_json(path, report)


def split_units(stage, meta):
    records = []
    for version in DEVICES:
        selected = meta if version == DEVICES[0] else meta[meta['备注'].ne('冷冻水二次泵')]
        for (building, unit), group in selected.groupby(['building', '东西南北'], sort=False):
            route = 'route_' + group.group_route.iloc[0]
            source = stage / 'raw_data' / version / route / f'{building}_data.csv'
            table = read_table(source)[group.spot_id.tolist()]
            path = source.with_name(f'{building}_{unit}_data.csv')
            if path.exists():
                raise FileExistsError(path)
            write_csv(table, path)
            reread = pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')
            pd.testing.assert_frame_equal(reread, table, check_exact=True, check_freq=False)
            records.append({'path': str(path.relative_to(stage)), 'building': building, 'unit': unit,
                            'route': route, 'points': len(table.columns), 'rows': len(table),
                            'start': str(table.index[0]), 'end': str(table.index[-1]), 'sha256': sha256_file(path)})
    if len(records) != 36:
        raise AssertionError('应生成36张分套表')
    save_json(stage / 'analysis/equipment_regroup/unit_tables.json', records)
    return records


def migrate_v3_targets(stage):
    for version in DEVICES:
        target_path = stage / f'forecast_data/data_v3/{version}/A1_all/data.csv'
        target = pd.read_csv(target_path, index_col='time', parse_dates=True, float_precision='round_trip')
        lineage_path = stage / f'analysis/data_v3/forecast_windows/{version}/A1_all/mask.csv'
        lineage = pd.read_csv(lineage_path, index_col='time', parse_dates=True)
        old_ab = target.hvac_total_load_AB.copy()
        for route in ROUTES:
            col = 'hvac_total_load_' + route[-1]
            target[col] = read_table(stage / f'imputed_data/data_v3/{version}/{route}/A1_data.csv').total_load
            mask = read_mask(stage / f'analysis/data_v3/imputation/{version}/{route}/A1_data.mask.csv')
            lineage[col + '_observed'] = mask.total_observed
            lineage[col + '_eligibility_known_at'] = mask.eligibility_known_at
        # 重分组只允许加法次序的末位浮点差异；仍严格保持AB=A+B合同。
        np.testing.assert_allclose(old_ab, target.hvac_total_load_A + target.hvac_total_load_B, rtol=1e-14, atol=1e-10)
        target['hvac_total_load_AB'] = target.hvac_total_load_A + target.hvac_total_load_B
        write_csv(target, target_path)
        write_csv(lineage, lineage_path)
        dr_path = target_path.with_name('data_dr_adjusted.csv')
        dr = pd.read_csv(dr_path, index_col='time', parse_dates=True, float_precision='round_trip')
        for route in ROUTES:
            dr['hvac_total_load_' + route[-1]] = read_table(stage / f'outlier_remove_data/data_v3/demand_response/{version}/{route}/A1_data.csv').total_load
        np.testing.assert_allclose(dr.hvac_total_load_AB, dr.hvac_total_load_A + dr.hvac_total_load_B, rtol=1e-14, atol=1e-10)
        dr['hvac_total_load_AB'] = dr.hvac_total_load_A + dr.hvac_total_load_B
        write_csv(dr, dr_path)
        dr_mask_path = stage / f'analysis/data_v3/demand_response/{version}/mask.csv'
        dr_mask = pd.read_csv(dr_mask_path, index_col='time', parse_dates=True)
        for column in lineage:
            if column.startswith(('hvac_total_load_A_', 'hvac_total_load_B_')):
                dr_mask[column] = lineage[column]
        report = json.loads((stage / 'analysis/data_v3/demand_response/manifest.json').read_text())
        for event in report['recipe']['events']:
            end = pd.Timestamp(event['end_exclusive'])
            active = (dr_mask.index >= pd.Timestamp(event['start'])) & (dr_mask.index < end)
            for field in ('hvac_total_load_A', 'hvac_total_load_B'):
                dr_mask.loc[active, field + '_observed'] = False
                known = pd.to_datetime(dr_mask.loc[active, field + '_eligibility_known_at'])
                dr_mask.loc[active, field + '_eligibility_known_at'] = known.clip(lower=end + pd.Timedelta(minutes=25)).astype(str)
        write_csv(dr_mask, dr_mask_path)
    path = stage / 'analysis/data_v3/manifest.json'
    report = json.loads(path.read_text())
    for row in report['targets']:
        row['sha256'] = sha256_file(stage / row['path'])
    for key, row in report['tables'].items():
        if key.startswith('hvac_'):
            table = read_table(stage / 'imputed_data/data_v3' / key)
            raw = read_table(stage / 'raw_data' / key).reindex(table.index)
            row.update(points=len(table.columns) - 1, missing_cells=int(raw.drop(columns='total_load').isna().sum().sum()))
            if 'gap_count' in row:
                gaps = pd.read_csv(stage / 'analysis/data_v3/imputation' / key.replace('.csv', '.gaps.csv'))
                row['gap_count'] = len(gaps)
                row['filled_by_method'] = gaps.groupby('method').gap_points.sum().astype(int).to_dict()
    report['route_migration'] = 'analysis/equipment_regroup/manifest.json; original recipe describes pre-migration provenance'
    save_json(path, report)


def build(root, work):
    if work.exists():
        raise FileExistsError(f'工作目录已存在，请使用新的显式 --work: {work}')
    work.mkdir(parents=True)
    stage = work / 'stage'
    meta = load_mapping()
    source_paths = [p for p in root.rglob('*') if p.is_file() and 'archive' not in p.relative_to(root).parts]
    baseline = {str(p.relative_to(root)): sha256_file(p) for p in source_paths}
    save_json(work / 'baseline.json', baseline)
    shutil.copy2(XLSX, work / 'all_ids.xlsx')
    for p in source_paths:
        dest = stage / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    audit = stage / 'analysis/equipment_regroup'
    audit.mkdir(parents=True)
    meta.to_csv(audit / 'mapping.csv', index=False, encoding='utf-8-sig')
    migrate_tables(stage, meta, 'raw_data', strict=False)
    # 旧同内容副本也同步，避免留下可被误读的旧口径文件。
    alias = stage / 'raw_data/hvac_all_devices/route_B/A1_data-B.csv'
    if alias.exists():
        shutil.copy2(alias.with_name('A1_data.csv'), alias)
    units = split_units(stage, meta)
    migrate_tables(stage, meta, 'outlier_remove_data/data_v2', strict=False)
    continue_build(root, work)


def continue_build(root, work):
    stage = work / 'stage'
    baseline = json.loads((work / 'baseline.json').read_text())
    meta = load_mapping()
    units = json.loads((stage / 'analysis/equipment_regroup/unit_tables.json').read_text())
    jobs = [('imputed_data/data_v1', 'analysis/data_v1/imputation', '', None),
            (ISO + '/imputed_data', ISO + '/analysis/imputation', ISO, None),
            (RED + '/imputed_data', RED + '/analysis/imputation', RED, RED + '/analysis/outliers/corrections.csv'),
            ('imputed_data/data_v2', 'analysis/data_v2/imputation', '', None)]
    for tables, audits, owner, corrections in jobs:
        shutil.copytree(root / tables, stage / tables, dirs_exist_ok=True)
        shutil.copytree(root / audits, stage / audits, dirs_exist_ok=True)
        migrate_tables(stage, meta, tables, audits, corrections_rel=corrections)
        refresh_inventory(stage, owner, str(Path(audits).relative_to(owner)) if owner else audits)
    for rel in ('imputed_data/data_v3', 'analysis/data_v3/imputation', 'outlier_remove_data/data_v3/demand_response'):
        shutil.copytree(root / rel, stage / rel, dirs_exist_ok=True)
    migrate_tables(stage, meta, 'imputed_data/data_v3', 'analysis/data_v3/imputation', v3=True)
    migrate_tables(stage, meta, 'outlier_remove_data/data_v3/demand_response', v3=True)
    migrate_v3_targets(stage)
    print('Export forecast windows (no models)', flush=True)
    export_windows(stage / RED, preparation_root='.', data_version=None, replace=True)
    export_windows(stage, preparation_root=RED, data_version='data_v1', replace=True)
    export_windows(stage, preparation_root='.', data_version='data_v2', replace=True)
    # 不改变窗口，从而保持天气数值、模型配置时间边界不变。
    for rel in baseline:
        if rel.startswith('forecast_data/') and rel.endswith('.csv'):
            old = pd.read_csv(root / rel, usecols=['time'])
            new = pd.read_csv(stage / rel, usecols=['time'])
            pd.testing.assert_frame_equal(old, new)
    for path in (stage / 'weather_data').rglob('*.json'):
        report = json.loads(path.read_text())
        if 'target_file' in report:
            target = Path(report['target_file']).relative_to('dataset/aidc_hvac_load_5min')
            report['target_sha256'] = sha256_file(stage / target)
            save_json(path, report)
    save_json(work / 'build_state.json', {'status': 'staged', 'unit_tables': len(units), 'xlsx_sha256': sha256_file(XLSX)})
    print('STAGED', stage, flush=True)


def finalize(root, work):
    """生成显式差异清单；发布前再次逐文件验证输入未漂移。"""
    stage = work / 'stage'
    verification = json.loads((work / 'verification.json').read_text())
    if verification['status'] != 'passed' or verification['unit_tables'] != 36:
        raise ValueError('独立资产核验未通过')
    if json.loads((work / 'asset_tests.json').read_text())['status'] != 'passed':
        raise ValueError('真实v3点位/需求响应/天气核验未通过')
    baseline = json.loads((work / 'baseline.json').read_text())
    changes = []
    for path in sorted(stage.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(stage))
        digest = sha256_file(path)
        if baseline.get(rel) != digest:
            changes.append({'path': rel, 'before': baseline.get(rel), 'after': digest})
    report = {'status': 'verified_stage', 'changes': changes, 'models_run': False,
              'recipe_changes': json.loads((work / 'recipe_changes.json').read_text()),
              'xlsx_sha256': sha256_file(work / 'all_ids.xlsx'), 'backup': str(work / 'backup'),
              'semantics': 'group_route, not electrical feeder; frozen point values and original historical evidence',
              'historical_model_results': 'unchanged; NOT validated against corrected data'}
    save_json(work / 'publication.json', report)
    print('READY', len(changes), 'files', flush=True)


def publish(root, work):
    if root != DEFAULT_ROOT.resolve():
        raise ValueError('含配方绑定的发布仅允许本仓库正式场景根')
    report = json.loads((work / 'publication.json').read_text())
    if report['status'] != 'verified_stage' or sha256_file(XLSX) != report['xlsx_sha256']:
        raise ValueError('未核验或Excel改变')
    if (work / 'backup').exists():
        raise FileExistsError('已有发布备份，禁止重复发布')
    for row in report['recipe_changes']:
        if sha256_file(ROOT / row['path']) != row['before']:
            raise ValueError(f'配方在核验后改变: {row["path"]}')
    for row in report['changes']:
        path = root / row['path']
        current = sha256_file(path) if path.exists() else None
        if current != row['before'] or sha256_file(work / 'stage' / row['path']) != row['after']:
            raise ValueError(f'发布前输入漂移: {path}')
    for row in report['changes']:
        path = root / row['path']
        if path.exists():
            backup = work / 'backup' / row['path']
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / 'stage' / row['path'], path)
        if sha256_file(path) != row['after']:
            raise ValueError(f'发布回读失败: {path}')
    report['status'] = 'published_verified'
    for row in report['recipe_changes']:
        source = ROOT / row['path']
        backup = work / 'backup_config' / row['path']
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        shutil.copy2(work / 'recipes' / source.name, source)
        if sha256_file(source) != row['after']:
            raise ValueError(f'配方发布核验失败: {source}')
    save_json(work / 'publication.json', report)
    save_json(root / 'analysis/equipment_regroup/manifest.json', report)
    print('PUBLISHED', len(report['changes']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'continue', 'finalize', 'publish'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    {'build': build, 'continue': continue_build, 'finalize': finalize, 'publish': publish}[args.action](args.root.resolve(), args.work.resolve())


if __name__ == '__main__':
    main()
