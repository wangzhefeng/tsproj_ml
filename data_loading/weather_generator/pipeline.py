"""固定资产、原点可见性与明确天气配方的本地组合。"""
import hashlib
import json

import numpy as np
import pandas as pd

from data_loading.weather_generator.assets import WeatherAssetStore
from data_loading.weather_generator.derivation import derive_native
from data_loading.weather_generator.resampling import resampled_value
from data_loading.weather_generator.scenarios import proxy_time, combine_historical_snapshots


def generate_weather(spec, base_dir, origin, times, identity, *, store=None):
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        origin = origin.tz_localize(spec.temporal.timezone, ambiguous='raise', nonexistent='raise')
    origin = origin.tz_convert('UTC')
    local_times = pd.DatetimeIndex(times)
    if local_times.tz is None:
        local_times = local_times.tz_localize(spec.temporal.timezone, ambiguous='raise', nonexistent='raise')
    utc_times = local_times.tz_convert('UTC')
    if utc_times.empty or utc_times.has_duplicates or not utc_times.is_monotonic_increasing:
        raise ValueError('weather request needs nonempty unique ordered times')
    dependency_times = utc_times if spec.scenario == 'forecast' else pd.DatetimeIndex([
        proxy_time(t, spec.temporal.timezone, spec.proxy.leap_day, spec.temporal.freq) for t in utc_times
    ])
    mapping = {m.series_id: m.location_id for m in spec.location_map}
    if tuple(identity) not in mapping:
        raise ValueError('weather location_map missing request identity')
    location = mapping[tuple(identity)]
    store = store or WeatherAssetStore(base_dir)
    snapshots = [s for ref in spec.inputs for s in store.load(ref) if s.metadata['location_id'] == location]
    if len({s.metadata['source_id'] for s in snapshots}) > 1:
        raise ValueError('weather recipes cannot implicitly mix sources at one location')
    if spec.scenario == 'prior_year_proxy':
        snapshots = combine_historical_snapshots(snapshots, spec.proxy.data_kind,
                                                 None if spec.research is not None else origin)
    candidates = []
    errors = []
    identities = [s.metadata['snapshot_id'] for s in snapshots]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate/conflicting weather snapshot across manifests')
    if spec.scenario == 'forecast':
        snapshots.sort(key=lambda s: pd.Timestamp(s.metadata['init_time'])
                       if s.metadata['init_time'] is not None else pd.Timestamp.min.tz_localize('UTC'),
                       reverse=True)
    for snapshot in snapshots:
        meta = snapshot.metadata
        expected_kind = 'forecast' if spec.scenario == 'forecast' else spec.proxy.data_kind
        allowed_kinds = {expected_kind}
        if spec.research is not None and expected_kind == 'forecast':
            allowed_kinds.add('hindcast')
        if meta['data_kind'] not in allowed_kinds:
            continue
        rank = pd.Timestamp(meta['init_time']) if spec.scenario == 'forecast' else pd.Timestamp(0, tz='UTC')
        # 完整的最新rank已找到后，只检查同rank冲突；不重复物化所有较老批次。
        if candidates and rank < candidates[0][0]:
            break
        simulated = snapshot.frame.copy(deep=True)
        if spec.research is not None:
            delay = pd.Timedelta(spec.research.release_delay)
            if spec.scenario == 'forecast':
                if meta['init_time'] is None:
                    raise ValueError('research forecast requires explicit initialization')
                simulated['available_at'] = pd.Timestamp(meta['init_time']) + delay
            else:
                simulated['available_at'] = simulated.time + delay
                for variable in meta['variables']:
                    if variable['semantics'] != 'point' and variable['label'] == 'left':
                        mask = simulated.variable.eq(variable['name'])
                        simulated.loc[mask, 'available_at'] += pd.Timedelta(variable['native_freq'])
        visible = simulated.loc[simulated.available_at <= origin].copy()
        if visible.empty:
            continue
        variables = {v['name']: v for v in meta['variables']}
        try:
            visible, variables = derive_native(visible, variables, spec.native_features)
            records = []
            for time in dependency_times:
                record = {'time': time}
                availability = []
                for v in spec.variables:
                    if v.input not in variables:
                        raise ValueError('missing weather variable')
                    part = visible.loc[visible.variable.eq(v.input)]
                    value, available = resampled_value(part, time, variables[v.input], spec.temporal, v)
                    record[v.name] = value
                    availability.append(available)
                record['available_at'] = max(availability)
                records.append(record)
            frame = pd.DataFrame(records)
            value_columns = [v.name for v in spec.variables]
            if not np.isfinite(frame[value_columns].to_numpy(dtype=float)).all():
                raise ValueError('requested weather depends on missing native values')
            candidates.append((rank, snapshot, frame))
        except ValueError as exc:
            errors.append(str(exc))
    if not candidates:
        raise ValueError(f'no complete available weather snapshot: {errors}')
    candidates.sort(key=lambda c: c[0])
    if len(candidates) > 1 and candidates[-1][0] == candidates[-2][0]:
        raise ValueError('conflicting weather snapshots with equal rank')
    _, snapshot, frame = candidates[-1]
    recipe_hash = hashlib.sha256(json.dumps(spec.canonical_payload(), sort_keys=True).encode()).hexdigest()
    proof = {'scenario': spec.scenario, 'recipe_sha256': recipe_hash, 'snapshot_id': snapshot.metadata['snapshot_id'], 'source_id': snapshot.metadata['source_id'], 'location_id': location, 'dependencies': list(snapshot.dependency_hashes), 'max_available_at': frame.available_at.max().isoformat(), 'origin': origin.isoformat()}
    proof['dependency_times'] = [t.isoformat() for t in dependency_times]
    proof['snapshot_ids'] = list(snapshot.component_ids) or [snapshot.metadata['snapshot_id']]
    if spec.research is not None:
        proof.update(research=spec.canonical_payload()['research'], production_eligible=False,
                     strict_asof_verified=False, data_kind=snapshot.metadata['data_kind'],
                     actual_max_available_at=snapshot.frame.available_at.max().isoformat(),
                     availability_basis='research_assumption_not_historical_evidence')
    if snapshot.component_ids:
        proof['snapshot_id'] = None  # 历史组合没有单一预报批次，不能伪造批次名。
    frame['time'] = pd.DatetimeIndex(times)
    if times.tz is None:
        frame['available_at'] = frame.available_at.dt.tz_convert(spec.temporal.timezone).dt.tz_localize(None)
    else:
        frame['available_at'] = frame.available_at.dt.tz_convert(times.tz)
    return frame, proof
