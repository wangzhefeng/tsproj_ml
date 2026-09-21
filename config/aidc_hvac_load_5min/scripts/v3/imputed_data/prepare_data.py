"""从 A1 原始点位构建 data_v3；固定 IT 子集，无异常处理，不覆盖既有资产。"""
from __future__ import annotations

import argparse
from collections import Counter
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

from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import (
    publish_stage, read_table, write_csv,
)
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file
from config.aidc_hvac_load_5min.scripts.v3.imputed_data.gap_filling import fill_series

RECIPE = Path(__file__).resolve().parents[1] / 'preparation.json'
DEFAULT_ROOT = ROOT / 'dataset/aidc_hvac_load_5min'
DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')


def select_it(points, recipe):
    """冻结精确列集合；晚出现点位是口径排除，不推断不存在或补零。"""
    absent = set(recipe['excluded_absent_it_points'])
    late = set(recipe['excluded_late_it_points'])
    retained = recipe['retained_it_points']
    if (len(retained) != len(set(retained)) or absent & late or (absent | late) & set(retained)
            or set(points) != absent | late | set(retained)):
        raise ValueError('IT 点位集合与冻结配方不一致')
    if set(points.columns[points.isna().all()]) != absent:
        raise ValueError('IT 全空点位清单发生漂移')
    if not points.loc[:recipe['start'], sorted(late)].isna().all().all():
        raise ValueError('晚出现 IT 点位在起点前出现观测，需重新审核固定子集')
    if points.loc[pd.Timestamp(recipe['start']), retained].isna().any():
        raise ValueError('保留 IT 子集在起点缺少原观测')
    return points[retained].copy()


def fill_points(points, start, audit_path):
    """逐点计算，流式保存缺口审计；不把聚合的部分和当真值。"""
    window = points.loc[start:]
    result = window.copy()
    eligibility = window.index.to_numpy().copy()
    counts = Counter()
    gap_count = 0
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open('x', encoding='utf-8', newline='') as stream:
        writer = None
        for point in points:
            filled, rows = fill_series(points[point], output_start=start)
            result[point] = filled.loc[start:]
            if rows and writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
            if writer is not None:
                writer.writerows(rows)
            for row in rows:
                left = window.index.searchsorted(pd.Timestamp(row['gap_start']))
                right = left + row['gap_points']
                eligibility[left:right] = np.maximum(eligibility[left:right], np.datetime64(row['eligibility_known_at']))
                counts[row['method']] += row['gap_points']
            gap_count += len(rows)
    original = window.to_numpy()
    observed = np.isfinite(original)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('填补后仍有缺失，拒绝发布')
    if not np.array_equal(result.to_numpy()[observed], original[observed]):
        raise AssertionError('未缺失原观测被改变')
    result['total_load'] = result.sum(axis=1, min_count=len(points.columns))
    mask = pd.DataFrame({'total_observed': observed.all(axis=1),
                         'total_imputed': ~observed.all(axis=1),
                         'eligibility_known_at': eligibility}, index=window.index)
    summary = {'points': len(points.columns), 'missing_cells': int((~observed).sum()),
               'gap_count': gap_count, 'filled_by_method': dict(counts)}
    return result, mask, summary


def prepare(root, recipe):
    root = Path(root).resolve()
    categories = ('imputed_data', 'forecast_data', 'analysis')
    if any((root / category / 'data_v3').exists() for category in categories):
        raise FileExistsError('data_v3 已存在，拒绝覆盖或混合发布')
    start, end = pd.Timestamp(recipe['start']), pd.Timestamp(recipe['end'])
    grid = pd.date_range(start, end, freq='5min', name='time')
    if (recipe['data_version'] != 'data_v3' or start != start.normalize()
            or len(grid) % 288 or grid[-1] != end or recipe['context_days'] != 30):
        raise ValueError('v3 必须使用完整自然日、固定版本和30天校准上下文')
    expected = {f'raw_data/{v}/{r}/A1_data.csv' for v in DEVICES for r in ROUTES}
    expected.add('raw_data/IT_load/A1_data.csv')
    if set(recipe['input_sha256']) != expected:
        raise ValueError('原始源必须恰为5份 A1 表')
    sources = {}
    for relative, expected_sha in recipe['input_sha256'].items():
        path = root / relative
        if sha256_file(path) != expected_sha:
            raise ValueError(f'原始输入 SHA 与已批准配方不一致: {relative}')
        raw = read_table(path)
        points = raw.drop(columns='total_load')
        if not np.allclose(raw.total_load, points.sum(axis=1, min_count=1), equal_nan=True):
            raise ValueError(f'原始总分关系错误: {relative}')
        if 'IT_load' in relative:
            points = select_it(points, recipe)
        context = pd.date_range(start - pd.Timedelta(days=30), end, freq='5min', name='time')
        if not context.isin(points.index).all():
            raise ValueError(f'输入缺少输出网格或校准上下文: {relative}')
        sources[relative.removeprefix('raw_data/')] = points.loc[context]
    for route in ROUTES:
        whole = sources[f'hvac_all_devices/{route}/A1_data.csv']
        subset = sources[f'hvac_remove_devices/{route}/A1_data.csv']
        if not set(subset).issubset(whole) or not whole[subset.columns].equals(subset):
            raise ValueError(f'两设备口径共享点位不一致: {route}')
    report = {'data_version': 'data_v3', 'start': str(start), 'end': str(end), 'rows': len(grid),
              'recipe': recipe, 'tables': {}, 'targets': [], 'outlier_processing': False,
              'builder_sha256': sha256_file(Path(__file__)),
              'filler_sha256': sha256_file(Path(__file__).with_name('gap_filling.py')),
              'evidence': 'offline gap eligibility and fixed subset; estimates are not observations'}
    with TemporaryDirectory(prefix='hvac-a1-v3-') as temp:
        stage = Path(temp)
        tables, masks = {}, {}
        for key in ('IT_load/A1_data.csv', *(f'hvac_all_devices/{r}/A1_data.csv' for r in ROUTES)):
            print(f'填补 {key}', flush=True)
            path = stage / 'analysis/data_v3/imputation' / key.replace('.csv', '.gaps.csv')
            table, mask, summary = fill_points(sources[key], start, path)
            tables[key], masks[key] = table, mask
            report['tables'][key] = summary
        for route in ROUTES:
            key = f'hvac_remove_devices/{route}/A1_data.csv'
            whole_key = f'hvac_all_devices/{route}/A1_data.csv'
            columns = sources[key].columns
            table = tables[whole_key][columns].copy()
            table['total_load'] = table.sum(axis=1, min_count=len(columns))
            observed = sources[key].loc[start:].notna().all(axis=1)
            # 共享点位使用同一次补值；资格时间按本口径实际涉及的点位计算。
            eligibility = grid.to_numpy().copy()
            audit_path = stage / 'analysis/data_v3/imputation' / whole_key.replace('.csv', '.gaps.csv')
            if audit_path.stat().st_size:
                audit = pd.read_csv(audit_path)
                for row in audit[audit.point.isin(columns)].to_dict('records'):
                    left = grid.searchsorted(pd.Timestamp(row['gap_start']))
                    right = left + row['gap_points']
                    eligibility[left:right] = np.maximum(eligibility[left:right], np.datetime64(row['eligibility_known_at']))
            tables[key] = table
            masks[key] = pd.DataFrame({'total_observed': observed, 'total_imputed': ~observed,
                                      'eligibility_known_at': eligibility}, index=grid)
            report['tables'][key] = {'points': len(columns), 'reused_from': whole_key,
                                    'missing_cells': int(sources[key].loc[start:].isna().sum().sum())}
        for key, table in tables.items():
            write_csv(table, stage / 'imputed_data/data_v3' / key)
            write_csv(masks[key], stage / 'analysis/data_v3/imputation' / key.replace('.csv', '.mask.csv'))
        for devices in DEVICES:
            keys = {'hvac_total_load_A': f'{devices}/route_A/A1_data.csv',
                    'hvac_total_load_B': f'{devices}/route_B/A1_data.csv',
                    'it_subset_load': 'IT_load/A1_data.csv'}
            target = pd.DataFrame({c: tables[k].total_load for c, k in keys.items()}, index=grid)
            target.insert(0, 'hvac_total_load_AB', target.hvac_total_load_A + target.hvac_total_load_B)
            lineage = pd.DataFrame(index=grid)
            for column, key in keys.items():
                lineage[column + '_observed'] = masks[key].total_observed
                lineage[column + '_eligibility_known_at'] = masks[key].eligibility_known_at
            lineage['hvac_total_load_AB_observed'] = (
                lineage.hvac_total_load_A_observed & lineage.hvac_total_load_B_observed)
            lineage['hvac_total_load_AB_eligibility_known_at'] = np.maximum(
                masks[keys['hvac_total_load_A']].eligibility_known_at.to_numpy(),
                masks[keys['hvac_total_load_B']].eligibility_known_at.to_numpy())
            relative = Path('forecast_data/data_v3') / devices / 'A1_all/data.csv'
            write_csv(target, stage / relative)
            write_csv(lineage, stage / 'analysis/data_v3/forecast_windows' / devices / 'A1_all/mask.csv')
            report['targets'].append({'path': str(relative), 'sha256': sha256_file(stage / relative),
                                      'rows': len(target), 'columns': list(target),
                                      'target': 'hvac_total_load_AB'})
        for relative, expected_sha in recipe['input_sha256'].items():
            if sha256_file(root / relative) != expected_sha:
                raise ValueError(f'准备期间源发生变化: {relative}')
        manifest = stage / 'analysis/data_v3/manifest.json'
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        publish_stage(stage, root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    args = parser.parse_args()
    report = prepare(args.root, json.loads(args.recipe.read_text(encoding='utf-8')))
    print(json.dumps({'rows': report['rows'], 'tables': report['tables'], 'targets': report['targets']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
