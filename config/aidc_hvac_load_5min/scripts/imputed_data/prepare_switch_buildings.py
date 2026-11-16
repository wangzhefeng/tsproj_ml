"""扩展hvac_switch分套、分楼和三楼点位表；保留真实事件，不构建预测特征。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from config.aidc_hvac_load_5min.scripts.forecast_data.prepare_switch import read_frame
from config.aidc_hvac_load_5min.scripts.forecast_data.switch_features import clean_power
from config.aidc_hvac_load_5min.scripts.imputed_data.gap_filling import fill_series
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import write_csv, publish_stage
from config.aidc_hvac_load_5min.scripts.imputed_data.prepare_buildings import aggregate
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')
BUILDINGS = ('A1', 'A2', 'A3')
CATEGORIES = ('outlier_remove_data', 'imputed_data')
AUDIT = Path('analysis/hvac_switch/building_extension')
DEFAULT_ROOT = ROOT / 'dataset/aidc_hvac_load_5min'


def unit_specs(mapping):
    required = {'building', '东西南北', 'group_route', 'spot_id', '备注'}
    if not required.issubset(mapping) or set(mapping.building) != set(BUILDINGS):
        raise ValueError('需要三楼完整套组映射')
    units = []
    for (building, label), group in mapping.groupby(['building', '东西南北'], sort=True):
        if not isinstance(label, str) or not re.fullmatch('[东西南北][123]', label):
            raise ValueError('非法套组名称')
        points = group.spot_id.tolist()
        core = group.loc[group['备注'].ne('冷冻水二次泵'), 'spot_id'].tolist()
        routes = set(group.group_route)
        if (len(points) != 6 or len(set(points)) != 6 or len(core) != 5
                or int(group['备注'].eq('冷水机组').sum()) != 1
                or len(routes) != 1 or not routes.issubset({'A', 'B'})):
            raise ValueError('套组组成/路线错误')
        units.append({'building': building, 'label': label, 'route': 'route_' + next(iter(routes)),
                      'points': points, 'core_points': core})
    for building in BUILDINGS:
        selected = [u for u in units if u['building'] == building]
        points = [p for u in selected for p in u['points']]
        if len(selected) != 6 or len(set(points)) != 36:
            raise ValueError('每楼须六套且点位不能重复')
    return units


def unit_relative(unit, device):
    return Path(device) / unit['route'] / f"{unit['building']}_{unit['label']}_data.csv"


def strict_table(points):
    return points.assign(total_load=points.sum(axis=1, min_count=len(points.columns)))


def build(root):
    root = Path(root).resolve()
    mapping_path = root / 'analysis/equipment_regroup/mapping.csv'
    units = unit_specs(pd.read_csv(mapping_path))
    expected = [Path(category) / 'hvac_switch' / unit_relative(unit, device)
                for category in CATEGORIES for unit in units if unit['building'] != 'A1' for device in DEVICES]
    expected += [Path(category) / 'hvac_switch' / device / route / name
                 for category in CATEGORIES for device in DEVICES for route in ROUTES
                 for name in ('A1_data.csv', 'A2_data.csv', 'A3_data.csv', 'data.csv')]
    if (root / AUDIT).exists() or any((root / p).exists() for p in expected):
        raise FileExistsError('扩展产物或审计已存在，拒绝覆盖')
    source_hashes = {str(mapping_path.relative_to(root)): sha256_file(mapping_path)}
    def source(relative):
        path = root / relative
        source_hashes[str(relative)] = sha256_file(path)
        return read_frame(path)
    # Freeze existing inputs/views and forecasts; never borrow counterfactual corrected power.
    protected = {str(p.relative_to(root)): sha256_file(p)
                 for category in (*CATEGORIES, 'forecast_data')
                 for p in (root / category).rglob('*') if p.is_file()}
    report = {'units': units, 'source_sha256': source_hashes, 'protected_sha256': protected,
              'forecast_data_touched': False, 'models_run': False,
              'outlier_policy': 'negative/infinite only; real switching and demand response retained',
              'unfillable_policy': 'keep NaN with audit', 'tables': {},
              'builder_sha256': sha256_file(Path(__file__))}
    with TemporaryDirectory(prefix='hvac-switch-buildings-') as temp:
        stage = Path(temp)
        building_tables = {}
        for building in BUILDINGS:
            for route in ROUTES:
                selected = [u for u in units if u['building'] == building and u['route'] == route]
                if not selected:
                    raise ValueError('每楼每路必须有套组')
                raw_parts = []
                for unit in selected:
                    whole = source(Path('raw_data') / unit_relative(unit, DEVICES[0]))
                    reduced = source(Path('raw_data') / unit_relative(unit, DEVICES[1]))
                    if set(whole) != set(unit['points']) or set(reduced) != set(unit['core_points']):
                        raise ValueError('分套原始列与映射不一致')
                    pd.testing.assert_frame_equal(whole[reduced.columns], reduced, check_freq=False)
                    if raw_parts and not whole.index.equals(raw_parts[0].index):
                        raise ValueError('分套时间轴不同')
                    raw_parts.append(whole)
                raw = pd.concat(raw_parts, axis=1)
                full = source(Path('raw_data') / DEVICES[0] / route / f'{building}_data.csv').drop(columns='total_load')
                if not raw.columns.is_unique or set(raw) != set(full):
                    raise ValueError('分套并集与分楼表不一致')
                raw = raw[full.columns]
                pd.testing.assert_frame_equal(raw, full, check_freq=False)
                cleaned, invalid = clean_power(raw)
                if building == 'A1':
                    views = {}
                    for category in CATEGORIES:
                        parts = [source(Path(category) / 'hvac_switch' / unit_relative(u, DEVICES[0])) for u in selected]
                        if any(not p.index.equals(parts[0].index) for p in parts):
                            raise ValueError('既有A1分套时间轴不一致')
                        views[category] = pd.concat(parts, axis=1)[raw.columns]
                    pd.testing.assert_frame_equal(views[CATEGORIES[0]], cleaned, check_freq=False)
                    filled = views[CATEGORIES[1]]
                    truth = cleaned.loc[filled.index]
                    observed = truth.notna().to_numpy()
                    np.testing.assert_array_equal(filled.to_numpy()[observed], truth.to_numpy()[observed])
                else:
                    filled = cleaned.copy()
                    eligibility = pd.DataFrame({p: cleaned.index for p in cleaned}, index=cleaned.index)
                    gaps = []
                    for point in cleaned:
                        print('Fill', building, route, point, flush=True)
                        filled[point], rows = fill_series(cleaned[point], output_start=cleaned.index[0], on_unfillable='keep')
                        eligibility.loc[cleaned[point].isna(), point] = pd.NaT
                        for row in rows:
                            status = row.get('status', 'filled')
                            gaps.append({**row, 'status': status})
                            if status == 'filled':
                                eligibility.loc[row['gap_start']:row['gap_end'], point] = pd.Timestamp(row['eligibility_known_at'])
                    observed = cleaned.notna().to_numpy()
                    np.testing.assert_array_equal(filled.to_numpy()[observed], cleaned.to_numpy()[observed])
                    folder = stage / AUDIT / building / route
                    folder.mkdir(parents=True)
                    pd.DataFrame(gaps, columns=['point', 'gap_start', 'gap_end', 'gap_points', 'status', 'reason',
                        'method', 'validation_bucket_points', 'validation_count', 'validation_coverage_min',
                        'validation_start', 'validation_end', 'raw_dependency_start', 'raw_dependency_end',
                        'eligibility_known_at', 'locf_median_mae', 'past_mean_1h_median_mae', 'past_day_repeat_median_mae']).to_csv(folder / 'gaps.csv', index=False)
                    for name, frame in [('invalid', invalid), ('observed', cleaned.notna()),
                                        ('imputed', cleaned.isna() & filled.notna()), ('unresolved', filled.isna()),
                                        ('eligibility', eligibility)]:
                        write_csv(frame, folder / (name + '.csv'))
                    for unit in selected:
                        for device, columns in zip(DEVICES, (unit['points'], unit['core_points'])):
                            relative = unit_relative(unit, device)
                            write_csv(cleaned[columns], stage / CATEGORIES[0] / 'hvac_switch' / relative)
                            write_csv(filled[columns], stage / CATEGORIES[1] / 'hvac_switch' / relative)
                report['tables'][f'{building}/{route}'] = {'clean_rows': len(cleaned), 'imputed_rows': len(filled),
                    'points': len(cleaned.columns), 'invalid_cells': int(invalid.to_numpy().sum()),
                    'filled_cells': int((cleaned.reindex(filled.index).isna() & filled.notna()).to_numpy().sum()),
                    'unresolved_cells': int(filled.isna().to_numpy().sum())}
                for device in DEVICES:
                    columns = [p for u in selected for p in (u['points'] if device == DEVICES[0] else u['core_points'])]
                    if building == 'A1':
                        for unit in selected:
                            for category, truth in zip(CATEGORIES, (cleaned, filled)):
                                existing = source(Path(category) / 'hvac_switch' / unit_relative(unit, device))
                                pd.testing.assert_frame_equal(existing, truth[existing.columns], check_freq=False)
                    for category, points in zip(CATEGORIES, (cleaned, filled)):
                        table = strict_table(points[columns])
                        building_tables[category, device, route, building] = table
                        write_csv(table, stage / category / 'hvac_switch' / device / route / f'{building}_data.csv')
        for device in DEVICES:
            for route in ROUTES:
                raw_total = source(Path('raw_data') / device / route / 'data.csv')
                for category in CATEGORIES:
                    parts = {b: building_tables[category, device, route, b] for b in BUILDINGS}
                    total = aggregate(parts, raw_total.index)
                    if set(total) != set(raw_total):
                        raise ValueError('三楼点位并集与raw总表不一致')
                    write_csv(total[raw_total.columns], stage / category / 'hvac_switch' / device / route / 'data.csv')
        for name, digest in {**protected, **source_hashes}.items():
            if sha256_file(root / name) != digest:
                raise ValueError(f'构建期间输入或既有资产发生变化: {name}')
        report['output_sha256'] = {str(p.relative_to(stage)): sha256_file(p) for p in stage.rglob('*.csv')}
        report['new_business_tables'] = len(expected)
        target = stage / AUDIT / 'manifest.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))
        publish_stage(stage, root)
    return {'new_business_tables': len(expected), 'tables': report['tables']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    print(json.dumps(build(parser.parse_args().root), ensure_ascii=False, indent=2))
