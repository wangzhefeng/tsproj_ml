"""从源数据独立回读水温流量产物；不调用构建或填补流程。"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file
from config.aidc_hvac_load_5min.scripts.v3.imputed_data.gap_filling import validation_blocks


def verify_assets(root, recipe, *, source_root=None):
    root = Path(root)
    source_root = root if source_root is None else Path(source_root)
    analysis = root / 'analysis/data_v3/hvac_flow_temp'
    manifest = json.loads((analysis / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['recipe'] != recipe:
        raise ValueError('配方与manifest不一致')
    grid = pd.date_range(recipe['start'], recipe['end'], freq='5min', name='time')

    def read(path, *, dates=False):
        frame = pd.read_csv(path, encoding='utf-8-sig', index_col='time', parse_dates=['time'],
                            float_precision='round_trip')
        if not frame.index.equals(grid):
            raise ValueError(f'网格错误: {path}')
        return frame.apply(pd.to_datetime) if dates else frame

    expected_paths = set()
    for stage in ('outlier_remove_data', 'imputed_data'):
        expected_paths.update(f'{stage}/data_v3/hvac_flow_temp/{name}' for name in ('data_temp.csv', 'data_flow.csv'))
    expected_paths.add('forecast_data/data_v3/hvac_flow_temp/features.csv')
    expected_paths.update(f'analysis/data_v3/hvac_flow_temp/{name}.csv' for name in (
        'gaps', 'point_imputed_mask', 'point_eligibility', 'feature_imputed_mask', 'feature_eligibility'))
    expected_paths.update(f'forecast_data/data_v3/{d}/A1_all/{n}'
                          for d in ('hvac_all_devices', 'hvac_remove_devices')
                          for n in ('data_with_flow_temp.csv', 'data_dr_adjusted_with_flow_temp.csv'))
    if set(manifest['output_sha256']) != expected_paths:
        raise ValueError('发布产物集合错误')
    for relative, expected in manifest['output_sha256'].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f'输出SHA不符: {relative}')
    inputs = {}
    for relative, expected in recipe['input_sha256'].items():
        if sha256_file(source_root / relative) != expected:
            raise ValueError(f'源SHA不符: {relative}')
        inputs[relative] = read(source_root / relative)
    raw, filled = {}, {}
    for table in ('data_temp.csv', 'data_flow.csv'):
        selected = [p['point'] for p in manifest['selected'] if p['table'] == table]
        source = inputs[f'raw_data/hvac_flow_temp/{table}']
        clean = read(root / f'outlier_remove_data/data_v3/hvac_flow_temp/{table}')
        pd.testing.assert_frame_equal(clean, source[selected], check_dtype=False, check_exact=True)
        result = read(root / f'imputed_data/data_v3/hvac_flow_temp/{table}')
        if list(result) != selected or not np.isfinite(result.to_numpy()).all():
            raise ValueError('填补表合同错误')
        for point in selected:
            raw[point], filled[point] = source[point], result[point]
            observed = source[point].notna()
            np.testing.assert_array_equal(result.loc[observed, point], source.loc[observed, point])
    if len(raw) != 58:
        raise ValueError('点位数量错误')
    mask = read(analysis / 'point_imputed_mask.csv')
    eligible = read(analysis / 'point_eligibility.csv', dates=True)
    expected_order = [p['point'] for table in ('data_temp.csv', 'data_flow.csv')
                      for p in manifest['selected'] if p['table'] == table]
    if list(mask) != expected_order or list(eligible) != expected_order:
        raise ValueError('点位证据列合同错误')
    audit = pd.read_csv(analysis / 'gaps.csv', float_precision='round_trip')
    checked, by_method = 0, {}
    methods = ('locf', 'past_mean_1h', 'past_day_repeat')
    lookbacks = (1, 12, 288)
    for point, source in raw.items():
        values = source.to_numpy()
        expected_mask = source.isna().to_numpy()
        np.testing.assert_array_equal(mask[point].to_numpy(), expected_mask.astype(int))
        coverage = np.zeros(len(grid), dtype=int)
        expected_eligible = grid.to_numpy().copy()
        validations = {}
        rows = audit.loc[audit.point == point]
        for row in rows.to_dict('records'):
            start = int(grid.searchsorted(pd.Timestamp(row['gap_start'])))
            stop = int(grid.searchsorted(pd.Timestamp(row['gap_end']))) + 1
            length = stop - start
            if (length != row['gap_points'] or start == 0 or stop == len(grid)
                    or not expected_mask[start:stop].all()
                    or expected_mask[start - 1] or expected_mask[stop]):
                raise ValueError('审计缺口边界错误')
            candidates = np.array([
                np.repeat(values[start - 1], length),
                # 与冻结候选的最近→最远累加顺序一致，保留逐位浮点验收。
                np.repeat(values[start - 12:start][::-1].mean(), length) if start >= 12 else np.full(length, np.nan),
                values[start - 288 + np.arange(length) % 288] if start >= 288 else np.full(length, np.nan),
            ])
            available = np.isfinite(candidates).all(axis=1)
            bucket = int(row['validation_bucket_points'])
            expected_bucket = next(b for b in (1, 3, 12, 36, 72, 288, 864, 2016, 6912) if b >= length)
            if bucket != expected_bucket:
                raise ValueError('缺口验证桶不符')
            if bucket not in validations:
                validations[bucket] = validation_blocks(values, bucket)
            origins, errors, truth_coverage = validations[bucket]
            valid = ((origins + bucket <= start) & (origins >= start - 30 * 288)
                     & np.isfinite(errors[:, available]).all(axis=1))
            if not available.any() or valid.sum() < 3 or valid.sum() != row['validation_count']:
                raise ValueError('遮蔽验证样本不符')
            scores = np.full(3, np.inf)
            scores[available] = np.median(errors[valid][:, available], axis=0)
            winner = int(scores.argmin())
            if row['method'] != methods[winner]:
                raise ValueError('选型不是历史MAE最优候选')
            for i, method in enumerate(methods):
                score = row[method + '_median_mae']
                if available[i]:
                    np.testing.assert_allclose(score, scores[i], rtol=1e-13, atol=1e-13)
                elif not pd.isna(score):
                    raise ValueError('不可用方法不得有评分')
            expected_metadata = {
                'validation_start': grid[origins[valid][0]],
                'validation_end': grid[origins[valid][-1] + bucket - 1],
                'raw_dependency_start': grid[min(origins[valid][0] - max(np.asarray(lookbacks)[available]),
                                                 start - lookbacks[winner])],
                'raw_dependency_end': grid[start - 1],
                'eligibility_known_at': grid[stop],
            }
            for key, value in expected_metadata.items():
                if pd.Timestamp(row[key]) != value:
                    raise ValueError(f'审计时间不符: {key}')
            np.testing.assert_allclose(row['validation_coverage_min'], truth_coverage[valid].min())
            np.testing.assert_array_equal(filled[point].iloc[start:stop].to_numpy(), candidates[winner])
            coverage[start:stop] += 1
            expected_eligible[start:stop] = grid[stop].to_datetime64()
            checked += length
            by_method[methods[winner]] = by_method.get(methods[winner], 0) + length
        np.testing.assert_array_equal(coverage, expected_mask.astype(int))
        np.testing.assert_array_equal(eligible[point].to_numpy(), expected_eligible)
    if checked != manifest['filled_cells'] or by_method != manifest['filled_by_method'] or len(audit) != manifest['gaps']:
        raise ValueError('补值审计计数错误')
    features = read(root / 'forecast_data/data_v3/hvac_flow_temp/features.csv')
    fm = read(analysis / 'feature_imputed_mask.csv')
    fe = read(analysis / 'feature_eligibility.csv', dates=True)
    columns = [p['feature'] for p in manifest['selected']] + [f['feature'] for f in manifest['formulas']]
    if len(columns) != 76 or list(features) != columns or list(fm) != columns or list(fe) != columns:
        raise ValueError('特征列合同错误')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('特征非有限')
    for item in manifest['selected']:
        name, point = item['feature'], item['point']
        np.testing.assert_array_equal(features[name], filled[point])
        np.testing.assert_array_equal(fm[name], mask[point])
        np.testing.assert_array_equal(fe[name], eligible[point])
    for item in manifest['formulas']:
        name = item['feature']
        left, right = item['inputs']
        a, b = features[left].to_numpy(), features[right].to_numpy()
        if item['operation'] not in ('subtract', 'multiply'):
            raise ValueError('未知公式操作')
        expected = np.subtract(a, b) if item['operation'] == 'subtract' else np.multiply(a, b)
        np.testing.assert_array_equal(features[name], expected)
        np.testing.assert_array_equal(fm[name], np.maximum(fm[left], fm[right]))
        np.testing.assert_array_equal(fe[name], np.maximum(fe[left], fe[right]))
    for devices in ('hvac_all_devices', 'hvac_remove_devices'):
        directory = Path('forecast_data/data_v3') / devices / 'A1_all'
        for old, new in [('data.csv', 'data_with_flow_temp.csv'),
                         ('data_dr_adjusted.csv', 'data_dr_adjusted_with_flow_temp.csv')]:
            source = inputs[str(directory / old)]
            combined = read(root / directory / new)
            if list(combined) != list(source) + columns:
                raise ValueError('拼接列集合错误')
            pd.testing.assert_frame_equal(combined[source.columns], source, check_exact=True)
            pd.testing.assert_frame_equal(combined[columns], features, check_exact=True)
    return {'status': 'passed', 'rows': len(grid), 'feature_count': len(columns),
            'selected_points': len(raw), 'filled_cells': checked, 'gaps': len(audit),
            'filled_by_method': by_method}
