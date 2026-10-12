"""A1 需求响应常态估计；原 data.csv 不变，新增 data_dr_adjusted.csv。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import publish_stage, write_csv
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file
from config.aidc_hvac_load_5min.scripts.v3.imputed_data.gap_filling import METHODS, predictions, validation_blocks

STEP = pd.Timedelta(minutes=5)
DEVICES = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')
COLUMNS = ('hvac_total_load_AB', 'hvac_total_load_A', 'hvac_total_load_B', 'it_subset_load')
AUDIT = Path('analysis/data_v3/demand_response')
POINT_TABLES = Path('outlier_remove_data/data_v3/demand_response')
RECIPE = Path(__file__).with_name('demand_response.json')
DEFAULT_ROOT = ROOT / 'dataset/aidc_hvac_load_5min'


def read_frame(path):
    frame = pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.tz is not None
            or not index.equals(pd.date_range(index[0], periods=len(index), freq=STEP))):
        raise ValueError(f'需要唯一连续5min时间轴: {path}')
    return frame


def validate_events(index, events):
    if not events:
        raise ValueError('事件清单不得为空')
    scope = np.zeros(len(index), dtype=bool)
    identifiers = set()
    previous_end = None
    for event in events:
        if set(event) != {'event_id', 'start', 'rebound_start', 'end_exclusive'}:
            raise ValueError('事件字段必须显式且无未知字段')
        name = event['event_id']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9_]+', name) or name in identifiers:
            raise ValueError('事件标识非法或重复')
        identifiers.add(name)
        start, rebound, end = (pd.Timestamp(event[k]) for k in ('start', 'rebound_start', 'end_exclusive'))
        if (any(t not in index for t in (start, rebound, end)) or not start < rebound < end
                or (end - start) > pd.Timedelta(hours=6)
                or (previous_end is not None and start < previous_end)):
            raise ValueError('事件越界、重叠、无序或长度不合法')
        scope |= (index >= start) & (index < end)
        previous_end = end
    return scope


def correct_points(raw, parent, events):
    """仅重估指定事件；原始点位为参考，父表只提供范围外原样保留的值。"""
    if not raw.columns.equals(parent.columns) or not parent.index.isin(raw.index).all():
        raise ValueError('点位集合或父表时间轴不一致')
    for frame in (raw, parent):
        if not frame.index.equals(pd.date_range(frame.index[0], periods=len(frame), freq=STEP)):
            raise ValueError('点位表必须为连续5min网格')
        values = frame.to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError('负荷不允许负值或无限值')
    if not np.isfinite(parent.to_numpy()).all():
        raise ValueError('父版填补表必须完整')
    scope = validate_events(parent.index, events)
    clean = raw.copy()
    for event in events:
        clean.loc[(clean.index >= event['start']) & (clean.index < event['end_exclusive'])] = np.nan
    result = parent.copy()
    methods, cells, blocks = [], [], []
    for event in events:
        start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end_exclusive'])
        times = pd.date_range(start, end - STEP, freq=STEP)
        length = len(times)
        for point in raw:
            # 30天校准起点 + 最多1天候选上下文；不读取当前事件后的观测。
            history = clean.loc[start - pd.Timedelta(days=31):start - STEP, point]
            values = history.to_numpy(dtype=float)
            candidates = predictions(values, [len(values)], length)[0]
            available = np.isfinite(candidates).all(axis=1)
            origins, errors, _ = validation_blocks(values, length)
            common = np.isfinite(errors[:, available]).all(axis=1)
            if not available.any() or common.sum() < 3:
                raise ValueError(f'{event["event_id"]}/{point}: 过去原观测校准不足')
            scores = np.full(len(METHODS), np.inf)
            scores[available] = np.median(errors[common][:, available], axis=0)
            winner = int(scores.argmin())
            corrected = candidates[winner]
            result.loc[times, point] = corrected
            context = {'event_id': event['event_id'], 'point': point}
            methods.append({**context, 'start': str(start), 'end_exclusive': str(end),
                            'method': METHODS[winner], 'validation_count': int(common.sum()),
                            'raw_dependency_start': str(history.index[0]),
                            'raw_dependency_end': str(history.index[-1]),
                            **{m + '_median_mae': float(scores[i]) if available[i] else None
                               for i, m in enumerate(METHODS)}})
            for i in np.flatnonzero(common):
                for method in np.flatnonzero(available):
                    blocks.append({**context, 'method': METHODS[method],
                                   'validation_start': str(history.index[origins[i]]),
                                   'validation_end': str(history.index[origins[i] + length - 1]),
                                   'mae': float(errors[i, method])})
            for time, value in zip(times, corrected):
                original, old = raw.at[time, point], parent.at[time, point]
                cells.append({**context, 'time': str(time),
                              'phase': 'suppression' if time < pd.Timestamp(event['rebound_start']) else 'recovery',
                              'raw_value': original, 'raw_observed': bool(pd.notna(original)),
                              'parent_value': old, 'adjusted_value': value, 'changed': bool(old != value)})
    pd.testing.assert_frame_equal(result.loc[~scope], parent.loc[~scope])
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('修正后数值不完整')
    return result, pd.DataFrame(methods), pd.DataFrame(cells), pd.DataFrame(blocks)


def recovery_evidence(targets, masks, events):
    """重新核对冻结边界；曲线判定与补值的过去原观测约束分别记账。"""
    rows = []
    for event in events:
        start, rebound, end = (pd.Timestamp(event[k]) for k in ('start', 'rebound_start', 'end_exclusive'))
        joint = None
        current = []
        for devices, frame in targets.items():
            series = frame.hvac_total_load_AB
            pre = series.loc[start - pd.Timedelta(hours=1):start - STEP]
            if len(pre) != 12:
                raise ValueError('事件边界缺少1h参考')
            baseline = float(pre.median())
            after = series.loc[rebound:start + pd.Timedelta(hours=6)]
            inside = after.between(.9 * baseline, 1.1 * baseline)
            joint = inside if joint is None else joint & inside
            window = series.loc[start:end - STEP]
            current.append({**event, 'devices': devices, 'baseline': baseline,
                            'reference_estimated_rows': int((~masks[devices].loc[pre.index, 'hvac_total_load_AB_observed']).sum()),
                            'event_estimated_rows': int((~masks[devices].loc[window.index, 'hvac_total_load_AB_observed']).sum()),
                            'minimum': float(window.min()), 'maximum': float(window.max()),
                            'slots': len(window), 'boundary_evidence_through': str(end + 5 * STEP)})
        stable = joint.rolling(6).sum()
        matches = stable[stable == 6].index
        if not len(matches) or matches[0] - 5 * STEP != end:
            raise ValueError(f'{event["event_id"]}: 冻结结束边界与双口径恢复规则不一致')
        rows.extend(current)
    return pd.DataFrame(rows)


def plot_events(before, after, events, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for event in events:
        start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end_exclusive'])
        fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
        for col, devices in enumerate(DEVICES):
            for row, field in enumerate(COLUMNS[:3]):
                ax = axes[row, col]
                old = before[devices].loc[start - pd.Timedelta(hours=2):end + pd.Timedelta(hours=2), field]
                new = after[devices].loc[old.index, field]
                ax.plot(old.index, old, color='#858585', linewidth=1.3, label='Original v3')
                ax.plot(new.index, new, color='#087f8c', linewidth=1.4, label='DR-adjusted estimate')
                ax.axvspan(start, pd.Timestamp(event['rebound_start']), alpha=.12, color='#d95f02')
                ax.axvspan(pd.Timestamp(event['rebound_start']), end, alpha=.1, color='#7570b3')
                ax.set_title(f'{devices} | {field}')
                ax.grid(alpha=.2)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.legend(fontsize=8)
        fig.suptitle(f'{event["event_id"]}: suppression + recovery (offline counterfactual)')
        fig.tight_layout()
        fig.savefig(directory / f'{event["event_id"]}.png', dpi=150)
        plt.close(fig)


def prepare(root, recipe):
    root = Path(root).resolve()
    if set(recipe) != {'kind', 'purpose', 'events', 'input_sha256'} or recipe['kind'] != 'hvac_demand_response_v1':
        raise ValueError('非法需求响应配方')
    outputs = [Path('forecast_data/data_v3') / d / 'A1_all/data_dr_adjusted.csv' for d in DEVICES]
    if (root / AUDIT).exists() or (root / POINT_TABLES).exists() or any((root / p).exists() for p in outputs):
        raise FileExistsError('需求响应产物已存在，拒绝覆盖')
    expected = set()
    for devices in DEVICES:
        for route in ROUTES:
            for category in ('raw_data', 'imputed_data/data_v3'):
                expected.add(f'{category}/{devices}/{route}/A1_data.csv')
        expected.add(f'forecast_data/data_v3/{devices}/A1_all/data.csv')
        expected.add(f'analysis/data_v3/forecast_windows/{devices}/A1_all/mask.csv')
    if set(recipe['input_sha256']) != expected:
        raise ValueError('输入必须恰为两口径的原始/父版点位、目标和mask')
    for path, digest in recipe['input_sha256'].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f'输入SHA不匹配: {path}')
    frames = {path: read_frame(root / path) for path in expected}
    before = {d: frames[f'forecast_data/data_v3/{d}/A1_all/data.csv'] for d in DEVICES}
    masks = {d: frames[f'analysis/data_v3/forecast_windows/{d}/A1_all/mask.csv'] for d in DEVICES}
    index = before[DEVICES[0]].index
    scope = validate_events(index, recipe['events'])
    for devices in DEVICES:
        target, mask = before[devices], masks[devices]
        if (tuple(target.columns) != COLUMNS or not target.index.equals(index)
                or not mask.index.equals(index) or not np.isfinite(target.to_numpy()).all()):
            raise ValueError('目标字段、网格或数值不符合父版合同')
        for field in COLUMNS:
            if mask[field + '_observed'].dtype != bool:
                raise ValueError('来源mask必须为布尔值')
            pd.to_datetime(mask[field + '_eligibility_known_at'], errors='raise')
        if not np.allclose(target.hvac_total_load_AB, target.hvac_total_load_A + target.hvac_total_load_B):
            raise ValueError('父版目标总分不一致')
        for route, field in zip(ROUTES, COLUMNS[1:3]):
            key = f'{devices}/{route}/A1_data.csv'
            raw, parent = frames['raw_data/' + key], frames['imputed_data/data_v3/' + key]
            points = parent.columns.drop('total_load')
            if not parent.index.equals(index) or not points.equals(raw.columns.drop('total_load')):
                raise ValueError('父版点位时间或列集合错误')
            if not index.isin(raw.index).all():
                raise ValueError('原始点位缺少业务时间轴')
            original = raw.loc[index, points].to_numpy()
            observed = np.isfinite(original)
            if not np.array_equal(parent[points].to_numpy()[observed], original[observed]):
                raise ValueError('父版原观测与raw不同')
            if (not np.allclose(parent[points].sum(axis=1, min_count=len(points)), parent.total_load)
                    or not np.allclose(parent.total_load, target[field])):
                raise ValueError('父版点位总量不一致')
            if not np.array_equal(observed.all(axis=1), mask[field + '_observed']):
                raise ValueError('父版观测mask与raw不同')
    if not before[DEVICES[0]].it_subset_load.equals(before[DEVICES[1]].it_subset_load):
        raise ValueError('两设备口径的IT不一致')
    evidence = recovery_evidence(before, masks, recipe['events'])
    all_methods, all_cells, all_blocks = [], [], []
    corrected = {}
    for route in ROUTES:
        key = f'hvac_all_devices/{route}/A1_data.csv'
        raw = frames['raw_data/' + key].drop(columns='total_load')
        parent = frames['imputed_data/data_v3/' + key].drop(columns='total_load')
        points, methods, cells, blocks = correct_points(raw, parent, recipe['events'])
        corrected[key] = points
        all_methods.append(methods.assign(route=route))
        all_cells.append(cells.assign(route=route))
        all_blocks.append(blocks.assign(route=route))
        subset_key = f'hvac_remove_devices/{route}/A1_data.csv'
        subset = frames['imputed_data/data_v3/' + subset_key].drop(columns='total_load')
        subset_raw = frames['raw_data/' + subset_key].drop(columns='total_load')
        if not set(subset).issubset(parent) or not parent[subset.columns].equals(subset) or not raw[subset.columns].equals(subset_raw):
            raise ValueError('两设备口径共享点位不一致')
        corrected[subset_key] = points[subset.columns].copy()
    methods, cells, blocks = (pd.concat(parts, ignore_index=True) for parts in (all_methods, all_cells, all_blocks))
    after = {}
    with TemporaryDirectory(prefix='hvac-dr-') as temporary:
        stage = Path(temporary)
        audit = stage / AUDIT
        audit.mkdir(parents=True)
        for key, points in corrected.items():
            table = points.assign(total_load=points.sum(axis=1, min_count=len(points.columns)))
            write_csv(table, stage / POINT_TABLES / key)
        for devices, output in zip(DEVICES, outputs):
            target = before[devices].copy()
            mask = masks[devices].copy()
            mask['dr_event_id'] = ''
            mask['dr_phase'] = ''
            mask['dr_counterfactual'] = scope
            for route, field in zip(ROUTES, COLUMNS[1:3]):
                points = corrected[f'{devices}/{route}/A1_data.csv']
                target.loc[scope, field] = points.loc[scope].sum(axis=1, min_count=len(points.columns))
            target.loc[scope, 'hvac_total_load_AB'] = target.loc[scope, 'hvac_total_load_A'] + target.loc[scope, 'hvac_total_load_B']
            for event in recipe['events']:
                start, rebound, end = (pd.Timestamp(event[k]) for k in ('start', 'rebound_start', 'end_exclusive'))
                active = (index >= start) & (index < end)
                mask.loc[active, 'dr_event_id'] = event['event_id']
                mask.loc[active, 'dr_phase'] = np.where(index[active] < rebound, 'suppression', 'recovery')
                for field in COLUMNS[:3]:
                    mask.loc[active, field + '_observed'] = False
                    known = pd.to_datetime(mask.loc[active, field + '_eligibility_known_at'])
                    mask.loc[active, field + '_eligibility_known_at'] = known.clip(lower=end + 5 * STEP).astype(str)
            pd.testing.assert_frame_equal(target.loc[~scope], before[devices].loc[~scope])
            pd.testing.assert_series_equal(target.it_subset_load, before[devices].it_subset_load)
            if not np.isfinite(target.to_numpy()).all() or (target.to_numpy() < 0).any():
                raise ValueError('派生目标非法')
            write_csv(target, stage / output)
            # CSV往返是验收的一部分，不能只验内存表。
            pd.testing.assert_frame_equal(read_frame(stage / output), target, check_exact=True)
            write_csv(mask, audit / devices / 'mask.csv')
            after[devices] = target
        for name, frame in (('events', evidence), ('point_methods', methods), ('point_changes', cells), ('validation_blocks', blocks)):
            frame.to_csv(audit / f'{name}.csv', index=False, encoding='utf-8-sig')
        plot_events(before, after, recipe['events'], audit / 'plots')
        for path, digest in recipe['input_sha256'].items():
            if sha256_file(root / path) != digest:
                raise ValueError(f'处理期间源改变: {path}')
        report = {'kind': recipe['kind'], 'recipe': recipe, 'strict_online_eligible': False,
                  'scope_note': 'Offline reviewed boundaries; estimates are not observations; outside-event parent estimates are retained.',
                  'boundary_rule': 'First 6 consecutive slots within both pre-event 1h medians +/-10%; end-exclusive at stable run start.',
                  'rows_per_target': len(index), 'event_rows_per_target': int(scope.sum()),
                  'point_estimates': len(cells), 'point_values_changed': int(cells.changed.sum()),
                  'methods': methods.method.value_counts().to_dict(),
                  'checks': {'source_sha_unchanged': True, 'outside_event_unchanged': True, 'it_unchanged': True,
                             'shared_points_equal': True, 'finite_nonnegative': True, 'csv_roundtrip_exact': True},
                  'builder_sha256': sha256_file(Path(__file__)),
                  'candidate_code_sha256': sha256_file(Path(__file__).parents[1] / 'imputed_data/gap_filling.py'),
                  'artifact_sha256': {str(p.relative_to(stage)): sha256_file(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
        (audit / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        publish_stage(stage, root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    args = parser.parse_args()
    result = prepare(args.root, json.loads(args.recipe.read_text(encoding='utf-8')))
    print(json.dumps({k: result[k] for k in ('rows_per_target', 'event_rows_per_target', 'point_values_changed', 'methods', 'checks')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
