"""暖通状态/事件特征：只确认过去已观测的变化，不从补值推断启停。"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_CODES = {'start': 1, 'stop': 2, 'replacement': 3, 'mixed': 4}
EVENT_COLUMNS = ['first_confirmed_at', 'confirmed_at', 'kind', 'from_unit', 'to_unit',
                 'before', 'after', 'confirmation_span_minutes']


def clean_power(raw):
    """只屏蔽确定违反非负有限功率合同的值；不把幅度/停机/恒值当异常。"""
    invalid = (raw < 0) | np.isinf(raw)
    return raw.mask(invalid), invalid


def unit_states(power, *, on, off, confirm):
    if not 0 <= off < on or confirm < 1:
        raise ValueError('非法迟滞阈值或确认槽数')
    records = []
    state, candidate, count = -1, -1, 0
    since = None
    censored = True
    for stamp, value in power.items():
        start = stop = 0
        if not np.isfinite(value):
            state, candidate, count, since, censored = -1, -1, 0, None, True
            transition = 0
        else:
            wanted = 1 if value >= on else 0 if value <= off else -1
            transition = int(wanted != state or wanted == -1)
            if wanted == -1 or wanted == state:
                candidate, count = -1, 0
            else:
                count = count + 1 if wanted == candidate else 1
                candidate = wanted
                if count >= confirm:
                    prior = state
                    state = wanted
                    since = stamp - pd.Timedelta(minutes=5 * (confirm - 1))
                    censored = prior == -1
                    start, stop = int(prior == 0 and state == 1), int(prior == 1 and state == 0)
                    candidate, count, transition = -1, 0, 0
        age = (stamp - since).total_seconds() / 3600 if since is not None else 0.
        records.append((state, int(state >= 0), transition, age, int(censored), start, stop))
    return pd.DataFrame(records, index=power.index,
                        columns=['state', 'valid', 'transition', 'age_h', 'left_censored', 'start', 'stop'])


def module_events(states, valid, *, max_minutes):
    records = []
    previous = None
    pending = None
    for stamp, values, usable in zip(states.index, states.to_numpy(dtype=int), valid.to_numpy(dtype=bool)):
        if not usable:
            previous, pending = None, None
            continue
        if previous is None:
            previous = values.copy()
            continue
        if np.array_equal(previous, values):
            continue
        removed = np.flatnonzero((previous == 1) & (values == 0))
        added = np.flatnonzero((previous == 0) & (values == 1))
        kind = 'replacement' if len(removed) == len(added) == 1 else (
            'start' if not len(removed) else 'stop' if not len(added) else 'mixed')
        first = stamp
        before = previous
        if pending is not None and (stamp - pending[0]).total_seconds() <= max_minutes * 60:
            base = pending[1]
            off = np.flatnonzero((base == 1) & (values == 0))
            on = np.flatnonzero((base == 0) & (values == 1))
            if len(off) == len(on) == 1:
                kind, first, before, removed, added = 'replacement', pending[0], base, off, on
        records.append({'first_confirmed_at': str(first), 'confirmed_at': str(stamp), 'kind': kind,
                        'from_unit': int(removed[0] + 1) if len(removed) == 1 else 0,
                        'to_unit': int(added[0] + 1) if len(added) == 1 else 0,
                        'before': ''.join(map(str, before)), 'after': ''.join(map(str, values)),
                        'confirmation_span_minutes': (stamp - first).total_seconds() / 60})
        pending = None if kind == 'replacement' else (stamp, previous.copy())
        previous = values.copy()
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def construct_features(cleaned, filled, point_eligibility, units, *, on=20., off=1., confirm=3, max_minutes=60):
    """状态来自cleaned全历史；功率来自filled，依赖可得时间单列审计。"""
    index = cleaned.index
    output_index = filled.index
    features, eligibility, spec = {}, {}, []
    states = {}

    def add(name, values, description, known=None):
        name = 'hvac_sw_' + name
        series = pd.Series(values, index=index).reindex(output_index) if not isinstance(values, pd.Series) else values.reindex(output_index)
        features[name] = series
        eligibility[name] = pd.Series(output_index, index=output_index) if known is None else known.reindex(output_index)
        spec.append({'name': name, 'description': description, 'role': 'observed_past',
                     'availability': 'feature_eligibility.csv; never known_future'})

    for unit in units:
        code, point = unit['code'], unit['chiller']
        state = unit_states(cleaned[point], on=on, off=off, confirm=confirm)
        states[code] = state
        for column in ('state', 'valid', 'transition', 'age_h', 'left_censored'):
            add(f'{code}_{column}', state[column], f'{code}冷机{column}; state=-1为未知，时长为可观测下界')
        for kind in ('start', 'stop'):
            add(f'{code}_{kind}_24h', state[kind].rolling(288, min_periods=1).sum(), '过去24h已确认启停次数；无缺口跨接')
        add(f'{code}_chiller_kw', filled[point], '冷机功率，可能含过去值估计', point_eligibility[point])
        add(f'{code}_chiller_estimated', cleaned[point].reindex(output_index).isna().astype(int), '冷机功率是否估计')
        columns = unit['core_points']
        load = filled[columns].sum(axis=1, min_count=len(columns))
        known = point_eligibility[columns].max(axis=1)
        add(f'{code}_core_load_kw', load, '冷机+双塔+一次泵+冷却水泵严格总量，不含二次泵', known)
        add(f'{code}_core_load_estimated', cleaned[columns].reindex(output_index).isna().any(axis=1).astype(int), '一次侧套组功率是否含估计')

    all_events = []
    for module in ('north', 'south'):
        selected = [u for u in units if u['module'] == module]
        codes = [u['code'] for u in selected]
        matrix = pd.DataFrame({c: states[c].state for c in codes}, index=index)
        valid = matrix.ge(0).all(axis=1)
        count = matrix.clip(lower=0).sum(axis=1).where(valid, -1)
        add(f'{module}_running_count', count, '模块冷机运行台数；-1为未知')
        add(f'{module}_valid', valid.astype(int), '三台冷机状态是否同时可观测')
        # 类别编码只用于类别标识，不代表有序大小。
        combo = (matrix.clip(lower=0).to_numpy() * np.array([4, 2, 1])).sum(axis=1)
        add(f'{module}_state_code', pd.Series(combo, index=index).where(valid, -1), '0..7类别编码，按1/2/3号位；-1未知')
        add(f'{module}_transition', pd.concat([states[c].transition for c in codes], axis=1).max(axis=1), '任一设备正在确认状态变化')
        for days in (7, 30):
            slots = days * 288
            coverage = valid.astype(float).rolling(slots, min_periods=1).sum() / slots
            hours = matrix.eq(1).mul(valid, axis=0).astype(float).rolling(slots, min_periods=1).sum() / 12
            add(f'{module}_coverage_{days}d', coverage, '共同有效采样比例；统计运行小时必须结合覆盖率')
            add(f'{module}_runtime_range_{days}d_h', hours.max(axis=1) - hours.min(axis=1), '同模组可观测运行小时极差')
            ranks = hours.rank(axis=1, method='min')
            for code in codes:
                add(f'{code}_runtime_{days}d_h', hours[code], '共同可观测区间的开机小时，不是设备寿命累计表')
                add(f'{code}_runtime_delta_{days}d_h', hours[code] - hours.mean(axis=1), '相对模组平均运行小时差')
                add(f'{code}_runtime_rank_{days}d', ranks[code], '运行小时升序排名，并列取最小排名')
        events = module_events(matrix, valid, max_minutes=max_minutes)
        events['module'] = module
        all_events.append(events)
        event_flag = pd.Series(0., index=index)
        replacement_flag = pd.Series(0., index=index)
        latest = pd.DataFrame(np.nan, index=index, columns=['kind', 'from', 'to', 'time_ns'])
        for row in events.to_dict('records'):
            stamp = pd.Timestamp(row['confirmed_at'])
            event_flag.loc[stamp] = 1
            replacement_flag.loc[stamp] = int(row['kind'] == 'replacement')
            latest.loc[stamp] = [EVENT_CODES[row['kind']], row['from_unit'], row['to_unit'], stamp.value]
        latest = latest.ffill()
        seen = latest.kind.notna()
        add(f'{module}_event_history_available', seen.astype(int), '是否已有可观测历史事件；不承诺缺口期间无事件')
        age = pd.Series((index.asi8 - latest.time_ns.to_numpy()) / 3.6e12, index=index).where(seen, 0.)
        add(f'{module}_since_event_h', age, '距最近一次已确认事件，未见事件编码0并附available')
        for source, name in [('kind', 'last_event_type'), ('from', 'last_from_unit'), ('to', 'last_to_unit')]:
            add(f'{module}_{name}', latest[source].fillna(0), '最近事件编码；0无历史/无唯一设备，type:1启动2停止3接替4混合')
        add(f'{module}_events_24h', event_flag.rolling(288, min_periods=1).sum(), '过去24h状态变化事件数')
        add(f'{module}_replacements_24h', replacement_flag.rolling(288, min_periods=1).sum(), '过去24h接替候选数，含可能需求响应相关事件，不称常规轮换')
        add(f'{module}_coverage_24h', valid.astype(float).rolling(288, min_periods=1).sum() / 288, '事件计数对应的共同观测覆盖率')
        chillers = [u['chiller'] for u in selected]
        total = filled[chillers].sum(axis=1)
        known = point_eligibility[chillers].max(axis=1)
        add(f'{module}_chiller_total_kw', total, '模组三冷机功率严格和', known)
        for unit in selected:
            share = (filled[unit['chiller']] / total.where(total.ne(0))).fillna(0.)
            add(f'{unit["code"]}_power_share', share, '冷机占模组冷机功率比；分母为0编码0并附valid', known)
        add(f'{module}_power_share_valid', total.ne(0).astype(int), '功率占比是否有定义', known)
    frame = pd.DataFrame(features, index=output_index)
    if frame.columns.has_duplicates or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('特征重复、缺失或非有限')
    return frame, pd.DataFrame(eligibility, index=output_index), pd.concat(all_events, ignore_index=True), spec
