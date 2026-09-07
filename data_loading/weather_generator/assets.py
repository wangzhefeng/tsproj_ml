"""固定 manifest 及传递依赖校验；不联网，不选择 latest 文件。"""
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading.weather_generator.contracts import (
    WeatherSnapshot, exact_fields, utc_timestamp, validate_snapshot_metadata,
)
from forecasting_core.specs.weather import WeatherInputSpec


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate manifest key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def checked_bytes(base_dir, reference):
    exact_fields(reference, "path sha256", "asset reference")
    checked = WeatherInputSpec(reference['path'], reference['sha256'])
    path = Path(checked.manifest)
    if not path.is_absolute():
        path = Path(base_dir) / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checked.sha256:
        raise ValueError(f"weather asset hash mismatch: {path}")
    return raw


def validate_normalized(frame, meta):
    """宽表规范 CSV：time + 每变量一列 + available_at；NaN 空单元格允许（缺测不填），inf 拒绝。"""
    variables = {v['name']: v for v in meta['variables']}
    if set(frame.columns) != {'time', 'available_at', *variables} or frame.empty:
        raise ValueError("invalid normalized weather columns or empty frame")
    frame = frame.copy(deep=True)
    for column in ('time', 'available_at'):
        frame[column] = pd.DatetimeIndex([utc_timestamp(v) for v in frame[column]])
    if frame['time'].duplicated().any():
        raise ValueError("conflicting duplicate weather timestamps")
    for name in variables:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        if np.isinf(frame[name].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("nonfinite normalized weather")
    if meta['evidence_class'] == 'received_snapshot':
        if (frame.available_at < utc_timestamp(meta['received_at'])).any():
            raise ValueError("available_at precedes received snapshot evidence")
    if meta['evidence_class'] == 'documented_release':
        if (frame.available_at < utc_timestamp(meta['issued_at'])).any():
            raise ValueError("available_at precedes documented release")
    if meta['data_kind'] == 'forecast':
        if (frame.available_at < utc_timestamp(meta['init_time'])).any():
            raise ValueError("forecast availability precedes initialization")
        if meta['issued_at'] is not None and (frame.available_at < utc_timestamp(meta['issued_at'])).any():
            raise ValueError("forecast availability precedes release")
    if meta['data_kind'] in {'observation', 'reanalysis'}:
        # 宽表逐行 available_at 对全部变量成立：左标签区间变量要求区间结束后才可得。
        shift = max((pd.Timedelta(v['native_freq']) for v in variables.values()
                     if v['semantics'] != 'point' and v['label'] == 'left'), default=pd.Timedelta(0))
        if (frame.available_at < frame.time + shift).any():
            raise ValueError("actual/reanalysis availability precedes dependency time")
    return frame.sort_values('time').reset_index(drop=True)


def wide_to_long(frame, meta):
    """宽表存储 → 内部 long 计算帧；NaN 行保留，由请求级覆盖与有限性校验拒绝。"""
    names = [v['name'] for v in meta['variables']]
    long = frame.melt(id_vars=['time', 'available_at'], value_vars=names,
                      var_name='variable', value_name='value')
    return long.sort_values(['variable', 'time']).reset_index(drop=True)


class WeatherAssetStore:
    """解码缓存按内容 hash；每次读取重验字节，避免同 registry 热改命中旧输入。"""
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self._decoded = {}

    def load(self, reference):
        ref = reference if isinstance(reference, WeatherInputSpec) else WeatherInputSpec(**reference)
        raw = checked_bytes(self.base_dir, {'path': ref.manifest, 'sha256': ref.sha256})
        manifest = strict_json(raw)
        exact_fields(manifest, "schema_version snapshots", "weather manifest")
        if manifest['schema_version'] != 'weather_asset_v1' or not isinstance(manifest['snapshots'], list) or not manifest['snapshots']:
            raise ValueError("invalid weather manifest version or snapshots")
        snapshots = []
        identities = set()
        for meta in manifest['snapshots']:
            validate_snapshot_metadata(meta)
            identity = tuple(meta[k] for k in ('source_id', 'location_id', 'snapshot_id'))
            if identity in identities:
                raise ValueError("duplicate/conflicting snapshot identity")
            identities.add(identity)
            if not isinstance(meta['raw'], list) or not meta['raw']:
                raise ValueError("snapshot requires raw dependencies")
            if meta['evidence_ref'] not in {r.get('path') for r in meta['raw']}:
                raise ValueError("release evidence must be a hashed raw dependency")
            dependencies = [(ref.manifest, ref.sha256)]
            for dependency in meta['raw']:
                checked_bytes(self.base_dir, dependency)
                dependencies.append((dependency['path'], dependency['sha256']))
            normalized = checked_bytes(self.base_dir, meta['normalized'])
            dependencies.append((meta['normalized']['path'], meta['normalized']['sha256']))
            key = meta['normalized']['sha256']
            if key not in self._decoded:
                self._decoded[key] = pd.read_csv(io.BytesIO(normalized))
            wide = validate_normalized(self._decoded[key], meta)
            snapshots.append(WeatherSnapshot(copy.deepcopy(meta), wide_to_long(wide, meta), tuple(dependencies)))
        return tuple(snapshots)
