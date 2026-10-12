"""v3 水温流量：58点位过去观测填补、76基础变量及双版本无损拼接。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file as sha256
from config.aidc_hvac_load_5min.scripts.v3.imputed_data import gap_filling
from config.aidc_hvac_load_5min.scripts.v3.forecast_data.verify_flow_temp import verify_assets

DEFAULT_ROOT = ROOT / 'dataset/aidc_hvac_load_5min'
RECIPE = Path(__file__).with_name('flow_temp.json')
ANALYSIS = Path('analysis/data_v3/hvac_flow_temp')
FEATURES = Path('forecast_data/data_v3/hvac_flow_temp/features.csv')
DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
VERSIONS = {'data.csv': 'data_with_flow_temp.csv',
            'data_dr_adjusted.csv': 'data_dr_adjusted_with_flow_temp.csv'}
TABLES = ('data_temp.csv', 'data_flow.csv')


def feature_spec():
    """显式点位与公式合同；编号不是跨设备通用语义。"""
    units = (
        ('n1', '31_15_0_16', (6, 5, 14, 15, 3, 121)),
        ('n2', '31_18_0_17', (6, 5, 14, 15, 3, 121)),
        ('n3', '31_23_0_18', (6, 5, 15, 16, 3, 121)),
        ('s1', '31_6_0_2', (6, 5, 14, 15, 3, 98)),
        ('s2', '31_8_0_3', (46, 45, 54, 55, 43, 119)),
        ('s3', '31_2_0_4', (46, 45, 54, 55, 43, 119)),
    )
    fields = ('chw_flow', 'cw_flow', 'cond_in_temp', 'cond_out_temp',
              'evap_out_temp', 'tower_out_temp')
    points, formulas = [], []
    for unit, prefix, numbers in units:
        names = {}
        for field, number in zip(fields, numbers):
            name = f'hvac_ft_{unit}_{field}'
            names[field] = name
            points.append({'feature': name, 'point': f'{prefix}_1_{number}_0',
                           'table': 'data_flow.csv' if field.endswith('flow') else 'data_temp.csv'})
        for field, left, right in (
            ('cond_delta_temp', 'cond_out_temp', 'cond_in_temp'),
            ('water_lift_proxy', 'cond_in_temp', 'evap_out_temp'),
        ):
            formulas.append({'feature': f'hvac_ft_{unit}_{field}', 'operation': 'subtract',
                             'inputs': [names[left], names[right]]})
        formulas.append({'feature': f'hvac_ft_{unit}_reject_proxy', 'operation': 'multiply',
                         'inputs': [names['cw_flow'], f'hvac_ft_{unit}_cond_delta_temp']})
    for zone, prefix in (('north', '31_20_0_15'), ('south', '31_14_0_1')):
        for kind, numbers in (
            ('supply_temp', ((1, 1), (2, 2), (3, 3), (4, 4))),
            ('return_temp', ((1, 5), (2, 6), (4, 8))),
            ('supply_flow', ((1, 27), (2, 28))),
            ('return_flow', ((3, 29), (4, 30))),
        ):
            for label, number in numbers:
                points.append({'feature': f'hvac_ft_{zone}_{kind}_{label}',
                               'point': f'{prefix}_1_{number}_0',
                               'table': 'data_flow.csv' if kind.endswith('flow') else 'data_temp.csv'})
    return points, formulas


def read_table(path, grid, *, finite=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError(f'重复列: {path}')
    frame = pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip')
    if len(frame.columns) != len(set(frame.columns)) or frame.columns[0] != 'time':
        raise ValueError(f'非法列合同: {path}')
    frame['time'] = pd.to_datetime(frame['time'], errors='raise')
    frame = frame.set_index('time')
    if not frame.index.equals(grid):
        raise ValueError(f'时间网格不一致: {path}')
    frame = frame.apply(pd.to_numeric, errors='raise')
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or (finite and not np.isfinite(values).all()):
        raise ValueError(f'存在非有限值: {path}')
    return frame


def write_table(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M:%S', na_rep='')


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def exclusion_reason(point):
    if point in {f'{p}_1_{k}_0' for p in ('31_20_0_15', '31_14_0_1') for k in (19, 20, 21, 22)}:
        return '供回水5/6温度固定异常值，不填造'
    if point in ('31_20_0_15_1_7_0', '31_14_0_1_1_7_0'):
        return '回水3存在长恒值，暂不纳入'
    if point in {f'{p}_1_{k}_0' for p, ks in (
        ('31_23_0_18', (38, 39, 40, 41)), ('31_2_0_4', (58, 59, 60, 61))) for k in ks}:
        return '疑似华氏温度，单位未确认，不转换'
    if point == '31_23_0_18_1_77_0':
        return '塔盘温度量程异常，暂不纳入'
    if point in ('31_8_0_3_1_1_0', '31_2_0_4_1_20_0'):
        return '无源观测，不填造'
    return '不属于已批准主特征集；不代表故障或不存在'


def input_paths():
    return [f'raw_data/hvac_flow_temp/{name}' for name in TABLES] + [
        f'forecast_data/data_v3/{devices}/A1_all/{name}'
        for devices in DEVICES for name in VERSIONS]


def load_inputs(root, recipe):
    grid = pd.date_range(recipe['start'], recipe['end'], freq='5min', name='time')
    if grid.empty or str(grid[-1]) != str(pd.Timestamp(recipe['end'])):
        raise ValueError('配方时间边界未对齐')
    if set(recipe['input_sha256']) != set(input_paths()):
        raise ValueError('来源必须恰为两份水系统表与四份目标表')
    frames = {}
    for relative, expected in recipe['input_sha256'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'输入SHA漂移: {relative}')
        frames[relative] = read_table(root / relative, grid, finite=not relative.startswith('raw_data/'))
    return grid, frames


def selected_tables(inputs):
    points, _ = feature_spec()
    result = {}
    for table in TABLES:
        source = inputs[f'raw_data/hvac_flow_temp/{table}']
        columns = [p['point'] for p in points if p['table'] == table]
        if not set(columns).issubset(source):
            raise ValueError(f'缺少登记列: {table}')
        frame = source[columns].astype(float).copy()
        if frame.isna().all().any():
            raise ValueError('已选点位全空，不允许填造')
        result[table] = frame
    return result


def construct_features(tables, point_masks, point_eligibility):
    points, formulas = feature_spec()
    grid = tables[TABLES[0]].index
    features, masks, eligibility = {}, {}, {}
    for item in points:
        name, point = item['feature'], item['point']
        features[name] = tables[item['table']][point]
        masks[name] = point_masks[point]
        eligibility[name] = point_eligibility[point]
    for item in formulas:
        name = item['feature']
        left, right = item['inputs']
        if item['operation'] == 'subtract':
            features[name] = features[left] - features[right]
        else:
            features[name] = features[left] * features[right]
        masks[name] = masks[left] | masks[right]
        eligibility[name] = np.maximum(eligibility[left], eligibility[right])
    return (pd.DataFrame(features, index=grid), pd.DataFrame(masks, index=grid),
            pd.DataFrame(eligibility, index=grid))


def destination_roots():
    return [Path(stage) / 'data_v3/hvac_flow_temp'
            for stage in ('outlier_remove_data', 'imputed_data', 'forecast_data', 'analysis')]


def target_outputs():
    return [Path('forecast_data/data_v3') / devices / 'A1_all' / name
            for devices in DEVICES for name in VERSIONS.values()]


def build(root, recipe):
    root = Path(root).resolve()
    for relative in destination_roots() + target_outputs():
        if (root / relative).exists():
            raise FileExistsError(f'拒绝覆盖: {relative}')
    grid, inputs = load_inputs(root, recipe)
    selected = selected_tables(inputs)
    points, formulas = feature_spec()
    report = {'recipe': recipe, 'selected_points': len(points), 'feature_count': len(points) + len(formulas),
              'rows': len(grid), 'selected': points, 'formulas': formulas,
              'excluded': {name: {p: exclusion_reason(p) for p in sorted(
                  set(inputs[f'raw_data/hvac_flow_temp/{name}']) - set(selected[name]))}
                           for name in TABLES},
              'evidence': 'offline historical features; imputed values are estimates; gap eligibility uses closure',
              'outlier_actions': 'explicit column selection only; no numeric clipping or unit conversion',
              'code_sha256': {str(Path(path).relative_to(ROOT)): sha256(Path(path))
                              for path in (__file__, gap_filling.__file__,
                                           Path(__file__).with_name('verify_flow_temp.py'),
                                           ROOT / 'config/aidc_hvac_load_5min/scripts/imputed_data/impute_hvac_data.py',
                                           ROOT / 'config/aidc_hvac_load_5min/scripts/raw_data/migrate_hvac_data.py')}}
    masks = pd.concat([frame.isna() for frame in selected.values()], axis=1)
    eligibility = pd.DataFrame({c: grid.to_numpy().copy() for c in masks}, index=grid)
    filled, audits, counts = {}, [], Counter()
    with TemporaryDirectory(prefix='hvac-flow-temp-') as temp:
        stage = Path(temp)
        for name, source in selected.items():
            write_table(source, stage / 'outlier_remove_data/data_v3/hvac_flow_temp' / name)
            result = source.copy()
            for point in source:
                print(f'填补 {point}', flush=True)
                series, rows = gap_filling.fill_series(source[point], output_start=grid[0])
                result[point] = series
                for row in rows:
                    row['table'] = name
                    audits.append(row)
                    counts[row['method']] += row['gap_points']
                    eligibility.loc[row['gap_start']:row['gap_end'], point] = pd.Timestamp(row['eligibility_known_at'])
                observed = source[point].notna()
                np.testing.assert_array_equal(result.loc[observed, point], source.loc[observed, point])
            if not np.isfinite(result.to_numpy()).all():
                raise ValueError('填补后仍有非有限值')
            filled[name] = result
            write_table(result, stage / 'imputed_data/data_v3/hvac_flow_temp' / name)
        features, feature_masks, feature_eligibility = construct_features(filled, masks, eligibility)
        if features.shape[1] != 76 or not np.isfinite(features.to_numpy()).all():
            raise ValueError('特征数量或有限性不符合合同')
        write_table(features, stage / FEATURES)
        write_table(masks.astype(int), stage / ANALYSIS / 'point_imputed_mask.csv')
        write_table(eligibility, stage / ANALYSIS / 'point_eligibility.csv')
        write_table(feature_masks.astype(int), stage / ANALYSIS / 'feature_imputed_mask.csv')
        write_table(feature_eligibility, stage / ANALYSIS / 'feature_eligibility.csv')
        audit_columns = ['point', 'gap_start', 'gap_end', 'gap_points', 'validation_bucket_points',
                         'method', 'validation_count', 'validation_coverage_min', 'validation_start',
                         'validation_end', 'raw_dependency_start', 'raw_dependency_end', 'eligibility_known_at',
                         *[m + '_median_mae' for m in gap_filling.METHODS], 'table']
        pd.DataFrame(audits, columns=audit_columns).to_csv(stage / ANALYSIS / 'gaps.csv', index=False, encoding='utf-8-sig')
        for devices in DEVICES:
            directory = Path('forecast_data/data_v3') / devices / 'A1_all'
            for old, new in VERSIONS.items():
                source = inputs[str(directory / old)]
                if set(source) & set(features):
                    raise ValueError('目标与特征列名冲突')
                write_table(source.join(features, validate='one_to_one'), stage / directory / new)
        report.update(gaps=len(audits), filled_cells=int(masks.to_numpy().sum()), filled_by_method=dict(counts))
        report['output_sha256'] = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob('*.csv'))}
        write_json(report, stage / ANALYSIS / 'manifest.json')
        # 所有产物先回读核验；源文件未变才允许首次发布。
        verify(stage, recipe, source_root=root)
        load_inputs(root, recipe)
        for relative in destination_roots() + target_outputs():
            if (root / relative).exists():
                raise FileExistsError(f'发布前目标出现: {relative}')
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                destination = root / path.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as reader, destination.open('xb') as writer:
                    shutil.copyfileobj(reader, writer)
    verify(root, recipe)
    return report


def verify(root, recipe, *, source_root=None):
    """独立落盘核验入口；详细审计位于同阶段验证模块。"""
    return verify_assets(Path(root), recipe, source_root=source_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
    result = verify(args.root, recipe) if args.verify_only else build(args.root, recipe)
    print(json.dumps({key: value for key, value in result.items()
                      if key in ('status', 'rows', 'feature_count', 'selected_points', 'filled_cells', 'gaps', 'filled_by_method')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
