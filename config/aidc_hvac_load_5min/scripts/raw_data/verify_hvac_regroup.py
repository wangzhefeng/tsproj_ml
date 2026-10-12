"""套组迁移的证据/图表更新与独立真实资产核验；不训练模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.aidc_hvac_load_5min.scripts.raw_data.equipment_mapping import load_mapping
from config.aidc_hvac_load_5min.scripts.raw_data.regroup_hvac_equipment import (
    DEFAULT_ROOT, DEFAULT_WORK, DEVICES, ROUTES, ISO, RED, BUILDINGS,
    save_json, read_table, write_csv, regroup_pair, migrate_v3_targets,
)
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file
from config.aidc_hvac_load_5min.scripts.analysis.analyze_forecast_data import run as draw_forecasts
from config.aidc_hvac_load_5min.scripts.v3.outlier_remove_data.adjust_demand_response import plot_events


def refresh(root, work):
    stage = work / 'stage'
    meta = load_mapping()
    lookup = meta.set_index('spot_id').group_route
    if not lookup.index.is_unique:
        raise ValueError('证据重映射要求全局点位唯一')
    # 部分masked_raw以迁移前raw为其余点位的明确基底，生成完整双路快照。
    for version in DEVICES:
        selected = meta if version == DEVICES[0] else meta[meta['备注'].ne('冷冻水二次泵')]
        for building in BUILDINGS:
            paths = [root / ISO / f'masked_raw/{version}/{r}/{building}_data.csv' for r in ROUTES]
            if not any(p.exists() for p in paths):
                continue
            pair = {r: read_table(p if p.exists() else root / f'raw_data/{version}/{r}/{building}_data.csv') for r, p in zip(ROUTES, paths)}
            mapped = regroup_pair(pair, selected[selected.building.eq(building)], strict=False)
            for route, table in mapped.items():
                write_csv(table, stage / ISO / f'masked_raw/{version}/{route}/{building}_data.csv')
    # 将历史判定发生时的路线显式保留；route/source成为当前点位表的可定位路径。
    evidence_dirs = [stage / ISO / 'analysis/outliers', stage / RED / 'analysis/outliers',
                     stage / 'analysis/data_v2/outliers', stage / 'analysis/data_v3/demand_response']
    for directory in evidence_dirs:
        for path in directory.glob('*.csv'):
            # 旧redbox的零待办报告只写一个换行；无点位行，原样保留。
            if path.stat().st_size <= 3 and not path.read_text(encoding='utf-8-sig').strip():
                continue
            data = pd.read_csv(path, float_precision='round_trip')
            if 'point' not in data or data.empty:
                continue
            known = data.point.isin(lookup.index)
            if not known.all():
                raise ValueError(f'未知证据点位: {path}')
            route = 'route_' + data.point.map(lookup)
            if 'route' in data:
                if 'original_route' not in data:
                    data['original_route'] = data.route
                data['route'] = route
            if 'source' in data:
                if 'original_source' not in data:
                    data['original_source'] = data.source
                data['source'] = [str(Path(s).parent.parent / r / Path(s).name) for s, r in zip(data.source, route)]
            data.to_csv(path, index=False, encoding='utf-8-sig')
    migrate_v3_targets(stage)
    # 原始迁移记录保留旧SHA，同时提供当前归属版本的SHA供活动消费者校验。
    path = stage / 'analysis/data_v1/preparation/raw_migration.json'
    report = json.loads(path.read_text())
    for row in report['raw_files']:
        row.setdefault('pre_regroup_sha256', row['sha256'])
        row['sha256'] = sha256_file(stage / row['destination'])
    report['route_migration'] = 'analysis/equipment_regroup/manifest.json'
    save_json(path, report)
    # 当前v3运行配方仅刷新已授权输入身份，原配方保留在备份/manifest字段。
    recipe_paths = ['config/aidc_hvac_load_5min/scripts/v3/preparation.json',
                    'config/aidc_hvac_load_5min/scripts/v3/outlier_remove_data/demand_response.json']
    recipe_changes = []
    for recipe_rel, manifest_rel in zip(recipe_paths, ['analysis/data_v3/manifest.json', 'analysis/data_v3/demand_response/manifest.json']):
        recipe = json.loads((ROOT / recipe_rel).read_text())
        report = json.loads((stage / manifest_rel).read_text())
        report.setdefault('pre_regroup_recipe', report['recipe'])
        for rel in recipe['input_sha256']:
            recipe['input_sha256'][rel] = sha256_file(stage / rel)
        report['recipe'] = recipe
        report['route_migration'] = 'analysis/equipment_regroup/manifest.json'
        save_json(stage / manifest_rel, report)
        save_json(work / 'recipes' / Path(recipe_rel).name, recipe)
        recipe_changes.append({'path': recipe_rel, 'before': sha256_file(ROOT / recipe_rel),
                               'after': sha256_file(work / 'recipes' / Path(recipe_rel).name)})
    save_json(work / 'recipe_changes.json', recipe_changes)
    print('Regenerate current v1/v2 forecast plots', flush=True)
    for version in ('data_v1', 'data_v2'):
        draw_forecasts(stage, data_version=version, replace=True)
    before = {d: pd.read_csv(stage / f'forecast_data/data_v3/{d}/A1_all/data.csv', index_col='time', parse_dates=True, float_precision='round_trip') for d in DEVICES}
    after = {d: pd.read_csv(stage / f'forecast_data/data_v3/{d}/A1_all/data_dr_adjusted.csv', index_col='time', parse_dates=True, float_precision='round_trip') for d in DEVICES}
    path = stage / 'analysis/data_v3/demand_response/manifest.json'
    report = json.loads(path.read_text())
    plot_events(before, after, report['recipe']['events'], path.parent / 'plots')
    for rel in report['artifact_sha256']:
        report['artifact_sha256'][rel] = sha256_file(stage / rel)
    save_json(path, report)
    plot_rows = []
    for device in DEVICES:
        for filename, plotname in [('data.csv', 'timeseries.png'), ('data_dr_adjusted.csv', 'timeseries_dr_adjusted.png')]:
            source = stage / f'forecast_data/data_v3/{device}/A1_all/{filename}'
            frame = pd.read_csv(source, index_col='time', parse_dates=True, float_precision='round_trip')
            destination = stage / f'analysis/data_v3/forecast_data_visual/{device}/A1_all/{plotname}'
            with plt.rc_context({'path.simplify': False}):
                fig, axes = plt.subplots(len(frame.columns), 1, figsize=(18, 12), sharex=True, layout='constrained')
                for ax, column in zip(axes, frame):
                    line, = ax.plot(frame.index, frame[column], lw=.6)
                    np.testing.assert_array_equal(line.get_ydata(), frame[column].to_numpy())
                    ax.set_title(column, loc='left')
                    ax.set_ylabel('kW')
                    ax.grid(alpha=.2)
                fig.suptitle(f'{device} / {filename} / group-route totals')
                fig.savefig(destination, dpi=180)
                plt.close(fig)
            plot_rows.append({'source': str(source.relative_to(stage)), 'source_sha256': sha256_file(source),
                              'output': str(destination.relative_to(stage)), 'output_sha256': sha256_file(destination),
                              'rows': len(frame), 'columns': list(frame), 'downsampled': False, 'additional_smoothing': False})
    save_json(stage / 'analysis/data_v3/forecast_data_visual/data_dr_adjusted.manifest.json',
              {'plots': plot_rows, 'route_migration': 'analysis/equipment_regroup/manifest.json',
               'verification': {'all_samples_plotted': True, 'plotted_values_equal_csv': True}})
    for path in (stage / 'weather_data').rglob('*.json'):
        report = json.loads(path.read_text())
        if 'target_file' in report:
            relative = Path(report['target_file']).relative_to('dataset/aidc_hvac_load_5min')
            report['target_sha256'] = sha256_file(stage / relative)
            save_json(path, report)
    print('EVIDENCE_REFRESHED', flush=True)


def bind_lineage(root, work):
    stage = work / 'stage'
    # 活动来源身份更新；旧身份另存，不冒称重新执行过异常检测/补值。
    paths = [('analysis/data_v2/outliers/manifest.json', 'raw_data'),
             ('analysis/data_v1/imputation/manifest.json', 'raw_data'),
             (ISO + '/analysis/imputation/manifest.json', 'raw_data'),
             (RED + '/analysis/imputation/manifest.json', 'raw_data'),
             ('analysis/data_v2/imputation/manifest.json', 'outlier_remove_data/data_v2')]
    for relative, source in paths:
        path = stage / relative
        report = json.loads(path.read_text())
        report.setdefault('pre_regroup_source_sha256', report['source_sha256'].copy())
        report['source_sha256'] = {key: sha256_file(stage / source / key) for key in report['source_sha256']}
        if 'output_sha256' in report:
            report.setdefault('pre_regroup_output_sha256', report['output_sha256'].copy())
            report['output_sha256'] = {key: sha256_file(stage / key) for key in report['output_sha256']}
        if 'outlier_cleaning' in report:
            owner = path.parents[2] if relative.startswith('outlier_remove_data/data_v1/') else stage
            report['outlier_cleaning']['sha256'] = sha256_file(owner / report['outlier_cleaning']['manifest'])
        report['route_migration'] = {'manifest': 'analysis/equipment_regroup/manifest.json',
                                     'historical_algorithm_not_rerun': True,
                                     'script_sha256': sha256_file(Path(__file__))}
        save_json(path, report)
    for relative, source in [('analysis/data_v1/forecast_windows/manifest.json', RED + '/analysis/imputation/manifest.json'),
                             ('analysis/data_v2/forecast_windows/manifest.json', 'analysis/data_v2/imputation/manifest.json'),
                             (RED + '/analysis/forecast_windows/manifest.json', RED + '/analysis/imputation/manifest.json')]:
        path = stage / relative
        report = json.loads(path.read_text())
        report['imputation_manifest_sha256'] = sha256_file(stage / source)
        save_json(path, report)
    for relative in ('analysis/data_v1/preparation/preparation_verification.json', 'analysis/data_v2/preparation_verification.json'):
        path = stage / relative
        report = json.loads(path.read_text())
        report['historical_pre_regroup_evidence'] = True
        report['current_verification'] = 'analysis/equipment_regroup/verification.json'
        save_json(path, report)
    print('LINEAGE_BOUND', flush=True)


def verify(root, work):
    """独立读取旧/新文件，核对全部被迁移点位及无关资产的字节不变性。"""
    stage = work / 'stage'
    meta = load_mapping()
    specs = [('raw_data', False), ('imputed_data/data_v1', True), ('imputed_data/data_v2', True),
             ('outlier_remove_data/data_v2', False), (ISO + '/imputed_data', True),
             (RED + '/imputed_data', True), ('imputed_data/data_v3', True),
             ('outlier_remove_data/data_v3/demand_response', True)]
    count = 0
    for base, strict in specs:
        for device in DEVICES:
            mapping = meta if device == DEVICES[0] else meta[meta['备注'].ne('冷冻水二次泵')]
            for building in BUILDINGS:
                if not (root / base / device / ROUTES[0] / f'{building}_data.csv').exists():
                    continue
                old = [read_table(root / base / device / r / f'{building}_data.csv').drop(columns='total_load') for r in ROUTES]
                expected = pd.concat(old, axis=1)
                seen = []
                for route in ROUTES:
                    path = stage / base / device / route / f'{building}_data.csv'
                    actual = read_table(path)
                    columns = mapping.loc[mapping.building.eq(building) & mapping.group_route.eq(route[-1]), 'spot_id'].tolist()
                    assert list(actual) == columns + ['total_load'], path
                    pd.testing.assert_frame_equal(actual[columns], expected[columns], check_exact=True)
                    total = actual[columns].sum(axis=1, min_count=len(columns) if strict else 1)
                    np.testing.assert_array_equal(actual.total_load, total)
                    seen.extend(columns)
                    count += 1
                assert len(seen) == len(set(seen)) and set(seen) == set(expected)
    unit_paths = []
    for device in DEVICES:
        selected = meta if device == DEVICES[0] else meta[meta['备注'].ne('冷冻水二次泵')]
        for (building, unit), group in selected.groupby(['building', '东西南北']):
            route = 'route_' + group.group_route.iloc[0]
            path = stage / f'raw_data/{device}/{route}/{building}_{unit}_data.csv'
            actual = pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')
            expected = read_table(stage / f'raw_data/{device}/{route}/{building}_data.csv')[group.spot_id.tolist()]
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            unit_paths.append(str(path.relative_to(stage)))
    assert len(unit_paths) == len(set(unit_paths)) == 36
    baseline = json.loads((work / 'baseline.json').read_text())
    unchanged = 0
    for rel, digest in baseline.items():
        if '/IT_load/' in rel or (rel.startswith('weather_data/') and rel.endswith('.csv')) or rel.startswith('raw_data/hvac_flow_temp/'):
            assert sha256_file(stage / rel) == digest, rel
            unchanged += 1
    for version in ('data_v1', 'data_v2'):
        manifest = json.loads((stage / f'analysis/{version}/forecast_windows/manifest.json').read_text())
        for row in manifest['windows']:
            assert sha256_file(stage / row['output']) == row['sha256']
            actual = pd.read_csv(stage / row['output'], float_precision='round_trip')
            old = pd.read_csv(root / row['output'], float_precision='round_trip')
            pd.testing.assert_series_equal(old.time, actual.time)
            for col in [c for c in actual if 'it_' in c]:
                pd.testing.assert_series_equal(old[col], actual[col], check_exact=True)
            assert np.isfinite(actual.drop(columns='time').to_numpy()).all()
    weather_count = 0
    for path in (stage / 'weather_data').rglob('*.meta.json'):
        metadata = json.loads(path.read_text())
        target = Path(metadata['target_file']).relative_to('dataset/aidc_hvac_load_5min')
        assert sha256_file(stage / target) == metadata['target_sha256'], path
        weather_count += 1
    assert weather_count == 66
    report = {'status': 'passed', 'point_tables_verified': count, 'unit_tables': len(unit_paths),
              'protected_files_unchanged': unchanged, 'weather_target_bindings': weather_count,
              'point_values_exact': True, 'models_run': False}
    save_json(stage / 'analysis/equipment_regroup/verification.json', report)
    save_json(work / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['refresh', 'bind', 'verify'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    {'refresh': refresh, 'bind': bind_lineage, 'verify': verify}[args.action](args.root.resolve(), args.work.resolve())
