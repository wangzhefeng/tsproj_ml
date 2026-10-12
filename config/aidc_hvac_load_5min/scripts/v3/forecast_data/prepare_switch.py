"""A1套组清洗、填补、事件及无损特征追加；需求响应只作事件，不屏蔽。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file as sha256
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import publish_stage, write_csv
from config.aidc_hvac_load_5min.scripts.v3.imputed_data import gap_filling
from config.aidc_hvac_load_5min.scripts.v3.forecast_data import switch_features

DEFAULT_ROOT = ROOT / 'dataset/aidc_hvac_load_5min'
RECIPE = Path(__file__).with_name('switch.json')
DR_RECIPE = Path(__file__).parents[1] / 'outlier_remove_data/demand_response.json'
DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
PARENTS = ('data_with_flow_temp.csv', 'data_dr_adjusted_with_flow_temp.csv')
AUDIT = Path('analysis/data_v3/hvac_switch')
FEATURE = Path('forecast_data/data_v3/hvac_switch/features.csv')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_frame(path):
    with path.open(encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream))
    if not header or header[0] != 'time' or len(set(header)) != len(header):
        raise ValueError(f'非法列: {path}')
    frame = pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.tz is not None
            or not index.equals(pd.date_range(index[0], index[-1], freq='5min', name='time'))):
        raise ValueError(f'非法时间网格: {path}')
    return frame


def namespace_roots():
    return [Path(category) / 'data_v3/hvac_switch' for category in
            ('outlier_remove_data', 'imputed_data', 'forecast_data', 'analysis')]


def target_pairs():
    return [(Path('forecast_data/data_v3') / d / 'A1_all' / p,
             Path('forecast_data/data_v3') / d / 'A1_all' / p.replace('.csv', '_switch.csv'))
            for d in DEVICES for p in PARENTS]


def freeze_recipe(root, path):
    if path.exists():
        raise FileExistsError(path)
    mapping_path = root / 'analysis/equipment_regroup/mapping.csv'
    meta = pd.read_csv(mapping_path)
    meta = meta[meta.building.eq('A1')]
    units, inputs = [], [mapping_path]
    for label, group in meta.groupby('东西南北', sort=True):
        module = 'north' if label[0] == '北' else 'south'
        code = ('n' if module == 'north' else 's') + label[-1]
        route = 'route_' + group.group_route.iloc[0]
        points = group.spot_id.tolist()
        chillers = group.loc[group['备注'].eq('冷水机组'), 'spot_id'].tolist()
        core = group.loc[group['备注'].ne('冷冻水二次泵'), 'spot_id'].tolist()
        if len(points) != 6 or len(core) != 5 or len(chillers) != 1 or group.group_route.nunique() != 1:
            raise ValueError('套组组成错误')
        units.append({'code': code, 'label': label, 'module': module, 'route': route,
                      'points': points, 'core_points': core, 'chiller': chillers[0]})
        inputs.extend(root / f'raw_data/{d}/{route}/A1_{label}_data.csv' for d in DEVICES)
    units.sort(key=lambda row: row['code'])
    if len(units) != 6:
        raise ValueError('必须有六套设备')
    parents = [read_frame(root / p) for p, _ in target_pairs()]
    grid = parents[0].index
    if any(not p.index.equals(grid) for p in parents):
        raise ValueError('四份基础表时间轴不同')
    inputs.extend(root / p for p, _ in target_pairs())
    recipe = {'start': str(grid[0]), 'end': str(grid[-1]), 'units': units,
              'state': {'on': 20., 'off': 1., 'confirm': 3, 'max_minutes': 60},
              'demand_response_events': json.loads(DR_RECIPE.read_text())['events'],
              'demand_response_recipe_sha256': sha256(DR_RECIPE),
              'input_sha256': {str(p.relative_to(root)): sha256(p) for p in inputs}}
    save_json(path, recipe)
    return recipe


def verify_sources(root, recipe):
    for relative, digest in recipe['input_sha256'].items():
        if sha256(root / relative) != digest:
            raise ValueError(f'输入SHA漂移: {relative}')


def load_points(root, recipe):
    frames = []
    for unit in recipe['units']:
        path = Path(f'{unit["route"]}/A1_{unit["label"]}_data.csv')
        full = read_frame(root / 'raw_data' / DEVICES[0] / path)
        reduced = read_frame(root / 'raw_data' / DEVICES[1] / path)
        if list(full) != unit['points'] or list(reduced) != unit['core_points']:
            raise ValueError('列集合/顺序与冻结配方不一致')
        pd.testing.assert_frame_equal(full[reduced.columns], reduced, check_exact=True)
        if frames and not full.index.equals(frames[0].index):
            raise ValueError('套组时间轴不同')
        frames.append(full)
    raw = pd.concat(frames, axis=1)
    if not raw.columns.is_unique:
        raise ValueError('套组点位重复')
    return raw


def annotate_events(events, index, recipe):
    events = events.copy()
    events['offline_attribution'] = 'unattributed'
    events.loc[events.kind.eq('replacement'), 'offline_attribution'] = 'rotation_candidate'
    events['demand_response_ids'] = ''
    labels = pd.DataFrame({'lbl_dr_active': 0, 'lbl_dr_event_id': '', 'lbl_dr_phase': ''}, index=index)
    for event in recipe['demand_response_events']:
        start, rebound, end = map(pd.Timestamp, (event['start'], event['rebound_start'], event['end_exclusive']))
        active = (index >= start) & (index < end)
        labels.loc[active, 'lbl_dr_active'] = 1
        labels.loc[active, 'lbl_dr_event_id'] = event['event_id']
        labels.loc[active, 'lbl_dr_phase'] = np.where(index[active] < rebound, 'suppression', 'recovery')
        # 确认有延迟；按已观测证据区间与人工事件相交进行离线归因。
        first = pd.to_datetime(events.first_confirmed_at) - pd.Timedelta(minutes=5 * (recipe['state']['confirm'] - 1))
        last = pd.to_datetime(events.confirmed_at)
        related = (first < end) & (last >= start)
        events.loc[related, 'offline_attribution'] = 'demand_response_related'
        events.loc[related, 'demand_response_ids'] = event['event_id']
    return events, labels


def build(root, recipe):
    root = Path(root).resolve()
    targets = namespace_roots() + [p for _, p in target_pairs()]
    if any((root / p).exists() for p in targets):
        raise FileExistsError('切换特征产物已存在，拒绝覆盖')
    verify_sources(root, recipe)
    raw = load_points(root, recipe)
    cleaned, invalid = switch_features.clean_power(raw)
    grid = pd.date_range(recipe['start'], recipe['end'], freq='5min', name='time')
    if not grid.isin(raw.index).all():
        raise ValueError('原始点位不能覆盖预测窗口')
    # 源及旧产物全部冻结；现有文件任何改动均阻止发布。
    protected = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}
    report = {'recipe': recipe, 'models_run': False, 'outlier_policy': 'negative/infinite only; demand response and real switching retained',
              'invalid_cells': int(invalid.to_numpy().sum()), 'raw_rows': len(raw), 'rows': len(grid),
              'protected_sha256': protected, 'point_estimation': 'past-only v3 selection; offline gap-closure eligibility',
              'state_estimation': 'cleaned observed chiller only; no inference from gap estimates',
              'code_sha256': {str(Path(p).relative_to(ROOT)): sha256(Path(p)) for p in
                              (__file__, switch_features.__file__, gap_filling.__file__)}}
    filled = cleaned.loc[grid].copy()
    eligibility = pd.DataFrame({p: grid.to_numpy().copy() for p in filled}, index=grid)
    gaps = []
    with TemporaryDirectory(prefix='hvac-switch-') as temp:
        stage = Path(temp)
        for point in cleaned:
            print('Fill', point, flush=True)
            # 不继承旧估计，不将响应区间屏蔽；只填补真实缺失或非法采集值。
            series, rows = gap_filling.fill_series(cleaned[point], output_start=grid[0])
            filled[point] = series.loc[grid]
            for row in rows:
                eligibility.loc[row['gap_start']:row['gap_end'], point] = pd.Timestamp(row['eligibility_known_at'])
            gaps.extend(rows)
        if not np.isfinite(filled.to_numpy()).all():
            raise ValueError('目标窗口填补不完整')
        for unit in recipe['units']:
            for devices, columns in zip(DEVICES, (unit['points'], unit['core_points'])):
                relative = Path(f'{devices}/{unit["route"]}/A1_{unit["label"]}_data.csv')
                write_csv(cleaned[columns], stage / 'outlier_remove_data/data_v3/hvac_switch' / relative)
                write_csv(filled[columns], stage / 'imputed_data/data_v3/hvac_switch' / relative)
        write_csv(invalid.astype(int), stage / AUDIT / 'invalid_mask.csv')
        write_csv(cleaned.loc[grid].isna().astype(int), stage / AUDIT / 'point_estimated_mask.csv')
        write_csv(eligibility, stage / AUDIT / 'point_eligibility.csv')
        gap_columns = ['point', 'gap_start', 'gap_end', 'gap_points', 'validation_bucket_points',
                       'method', 'validation_count', 'validation_coverage_min', 'validation_start',
                       'validation_end', 'raw_dependency_start', 'raw_dependency_end', 'eligibility_known_at',
                       *[m + '_median_mae' for m in gap_filling.METHODS]]
        audit = pd.DataFrame(gaps, columns=gap_columns)
        audit.to_csv(stage / AUDIT / 'gaps.csv', index=False, encoding='utf-8-sig')
        print('Construct causal states and events', flush=True)
        features, feature_eligibility, events, spec = switch_features.construct_features(
            cleaned, filled, eligibility, recipe['units'], **recipe['state'])
        annotated, labels = annotate_events(events, grid, recipe)
        write_csv(features, stage / FEATURE)
        write_csv(feature_eligibility, stage / AUDIT / 'feature_eligibility.csv')
        write_csv(labels, stage / AUDIT / 'offline_demand_response_labels.csv')
        annotated.to_csv(stage / AUDIT / 'events.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(recipe['demand_response_events']).assign(
            annotation_type='offline_manual_event; not known_future').to_csv(
                stage / AUDIT / 'demand_response_events.csv', index=False, encoding='utf-8-sig')
        save_json(stage / AUDIT / 'feature_schema.json', spec)
        for source, destination in target_pairs():
            parent = read_frame(root / source)
            if not parent.index.equals(grid) or set(parent) & set(features):
                raise ValueError('父表时间轴/列冲突')
            write_csv(parent.join(features, validate='one_to_one'), stage / destination)
        report.update(feature_count=len(features.columns), filled_cells=int(cleaned.loc[grid].isna().to_numpy().sum()),
                      gaps=len(gaps), confirmed_event_count=len(events),
                      events_in_forecast_window=int(pd.to_datetime(events.confirmed_at).between(grid[0], grid[-1]).sum()),
                      event_counts=annotated.groupby(['module', 'kind', 'offline_attribution']).size().rename('count').reset_index().to_dict('records'))
        report['output_sha256'] = {str(p.relative_to(stage)): sha256(p) for p in stage.rglob('*') if p.is_file()}
        save_json(stage / AUDIT / 'manifest.json', report)
        verify(stage, source_root=root)
        for relative, digest in protected.items():
            if sha256(root / relative) != digest:
                raise ValueError(f'旧资产发生改变: {relative}')
        publish_stage(stage, root)
    verify(root)
    return report


def verify(root, *, source_root=None):
    root = Path(root)
    source_root = Path(source_root) if source_root is not None else root
    report = json.loads((root / AUDIT / 'manifest.json').read_text())
    recipe = report['recipe']
    verify_sources(source_root, recipe)
    for relative, digest in report['output_sha256'].items():
        if sha256(root / relative) != digest:
            raise ValueError(f'产物SHA漂移: {relative}')
    raw = load_points(source_root, recipe)
    grid = pd.date_range(recipe['start'], recipe['end'], freq='5min', name='time')
    filled_parts = []
    for unit in recipe['units']:
        relative = Path(f'hvac_all_devices/{unit["route"]}/A1_{unit["label"]}_data.csv')
        clean = read_frame(root / 'outlier_remove_data/data_v3/hvac_switch' / relative)
        truth = raw[unit['points']]
        expected = truth.mask((truth < 0) | np.isinf(truth))
        pd.testing.assert_frame_equal(clean, expected, check_exact=True)
        filled = read_frame(root / 'imputed_data/data_v3/hvac_switch' / relative)
        pd.testing.assert_index_equal(filled.index, grid)
        observed = expected.loc[grid].notna().to_numpy()
        np.testing.assert_array_equal(filled.to_numpy()[observed], expected.loc[grid].to_numpy()[observed])
        reduced = read_frame(root / 'imputed_data/data_v3/hvac_switch/hvac_remove_devices' /
                             unit['route'] / f'A1_{unit["label"]}_data.csv')
        pd.testing.assert_frame_equal(reduced, filled[unit['core_points']], check_exact=True)
        filled_parts.append(filled)
    filled = pd.concat(filled_parts, axis=1)
    clean = raw.mask((raw < 0) | np.isinf(raw))
    gaps = pd.read_csv(root / AUDIT / 'gaps.csv', float_precision='round_trip')
    actual_mask = read_frame(root / AUDIT / 'point_estimated_mask.csv')
    pd.testing.assert_frame_equal(actual_mask, clean.loc[grid].isna().astype(int), check_dtype=False)
    expected_known = pd.DataFrame({p: grid.to_numpy().copy() for p in filled}, index=grid)
    covered = pd.DataFrame(False, index=grid, columns=filled.columns)
    for row in gaps.to_dict('records'):
        point = row['point']
        start = clean.index.get_loc(pd.Timestamp(row['gap_start']))
        length = row['gap_points']
        times = pd.date_range(row['gap_start'], periods=length, freq='5min')
        values = clean[point].to_numpy()
        if row['method'] == 'locf':
            expected = np.repeat(values[start - 1], length)
        elif row['method'] == 'past_mean_1h':
            expected = np.repeat(values[start - np.arange(1, 13)].mean(), length)
        elif row['method'] == 'past_day_repeat':
            expected = values[start - 288 + np.arange(length) % 288]
        else:
            raise ValueError('未知填补方法')
        if covered.loc[times, point].any() or not clean.loc[times, point].isna().all():
            raise ValueError('重复填补或原值被改变')
        np.testing.assert_array_equal(filled.loc[times, point], expected)
        if not (row['validation_count'] >= 3 and pd.Timestamp(row['validation_end']) < times[0]
                and pd.Timestamp(row['raw_dependency_end']) < times[0]):
            raise ValueError('非过去校准证据')
        scores = [row[m + '_median_mae'] for m in gap_filling.METHODS]
        if gap_filling.METHODS[int(np.nanargmin(scores))] != row['method']:
            raise ValueError('补值选型与分数不符')
        covered.loc[times, point] = True
        expected_known.loc[times, point] = pd.Timestamp(row['eligibility_known_at'])
    pd.testing.assert_frame_equal(covered, clean.loc[grid].isna(), check_freq=False)
    known = read_frame(root / AUDIT / 'point_eligibility.csv').apply(pd.to_datetime)
    pd.testing.assert_frame_equal(known, expected_known, check_freq=False)
    features = read_frame(root / FEATURE)
    if features.shape[1] != report['feature_count'] or not np.isfinite(features.to_numpy()).all():
        raise ValueError('特征数量或有限性错误')
    for source, destination in target_pairs():
        parent = read_frame(source_root / source)
        output = read_frame(root / destination)
        if list(output) != list(parent) + list(features):
            raise ValueError('最终字段集合错误')
        pd.testing.assert_frame_equal(output[parent.columns], parent, check_exact=True)
        pd.testing.assert_frame_equal(output[features.columns], features, check_exact=True)
    # 按配方重算并验证未来截断前缀；离线人工DR标签完全不参与此调用。
    rebuilt, rebuilt_known, _, _ = switch_features.construct_features(clean, filled, known, recipe['units'], **recipe['state'])
    pd.testing.assert_frame_equal(features, rebuilt, check_exact=True)
    feature_known = read_frame(root / AUDIT / 'feature_eligibility.csv').apply(pd.to_datetime)
    pd.testing.assert_frame_equal(feature_known, rebuilt_known)
    for cut in (len(grid) // 3, 2 * len(grid) // 3):
        end = grid[cut]
        prefix, _, _, _ = switch_features.construct_features(clean.loc[:end], filled.loc[:end], known.loc[:end], recipe['units'], **recipe['state'])
        pd.testing.assert_frame_equal(prefix, features.loc[:end], check_exact=True)
    for relative, digest in report['protected_sha256'].items():
        if sha256(source_root / relative) != digest:
            raise ValueError(f'已有数据改变: {relative}')
    result = {'status': 'passed', 'rows': len(grid), 'features': len(features.columns), 'outputs': len(target_pairs()),
              'filled_cells': report['filled_cells'], 'invalid_cells': report['invalid_cells'],
              'existing_files_unchanged': len(report['protected_sha256']), 'future_prefix_invariant': True, 'models_run': False}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    parser.add_argument('--freeze-recipe', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.freeze_recipe:
        freeze_recipe(args.root, args.recipe)
        print('Frozen', args.recipe)
    elif args.verify_only:
        verify(args.root)
    else:
        build(args.root, json.loads(args.recipe.read_text()))


if __name__ == '__main__':
    main()
