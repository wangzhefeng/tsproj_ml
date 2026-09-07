"""共享资产校验的纯合同；时间证据不由文件名推断。"""
from dataclasses import dataclass
import math
from zoneinfo import ZoneInfo

import pandas as pd

from forecasting_core.specs.weather import WEATHER_UNITS


def exact_fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"{label} requires exactly: {expected}")


def utc_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("weather timestamp must be explicit timezone-bearing string")
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("weather timestamp requires timezone")
    return result.tz_convert("UTC")


def validate_snapshot_metadata(meta):
    exact_fields(meta, "source_id product model snapshot_id location_id latitude longitude coordinate_system data_kind timezone init_time issued_at received_at evidence_class evidence_ref variables raw normalized", "snapshot")
    for field in ("source_id", "product", "model", "snapshot_id", "location_id", "evidence_ref"):
        if not isinstance(meta[field], str) or not meta[field].strip():
            raise ValueError(f"missing snapshot {field}")
    if meta['coordinate_system'] != 'WGS84':
        raise ValueError("weather coordinates require WGS84")
    for field, limit in (("latitude", 90), ("longitude", 180)):
        v = meta[field]
        if type(v) not in (int, float) or not math.isfinite(v) or abs(v) > limit:
            raise ValueError(f"invalid weather {field}")
    ZoneInfo(meta['timezone'])
    if meta['data_kind'] not in {'forecast', 'observation', 'reanalysis', 'hindcast'}:
        raise ValueError("unknown weather data_kind")
    if meta['evidence_class'] not in {'received_snapshot', 'documented_release', 'historical_release_contract'}:
        raise ValueError("unknown weather availability evidence")
    for field in ('init_time', 'issued_at', 'received_at'):
        if meta[field] is not None:
            utc_timestamp(meta[field])
    if meta['evidence_class'] == 'received_snapshot' and meta['received_at'] is None:
        raise ValueError("received_snapshot requires received_at")
    if meta['evidence_class'] == 'documented_release' and meta['issued_at'] is None:
        raise ValueError("documented_release requires issued_at")
    if meta['data_kind'] == 'forecast' and meta['init_time'] is None:
        raise ValueError("forecast requires init_time")
    if meta['init_time'] and meta['issued_at'] and utc_timestamp(meta['issued_at']) < utc_timestamp(meta['init_time']):
        raise ValueError("release precedes initialization")
    if not isinstance(meta['variables'], list) or not meta['variables']:
        raise ValueError("snapshot variables must be nonempty")
    names = []
    for v in meta['variables']:
        exact_fields(v, "name column unit semantics native_freq label", "snapshot variable")
        if not all(isinstance(v[k], str) and v[k] for k in ('name', 'column')):
            raise ValueError("variable requires name and column")
        if v['unit'] not in WEATHER_UNITS or v['semantics'] not in {'point', 'interval_mean', 'interval_sum'}:
            raise ValueError("unknown variable unit or interval semantics")
        if v['native_freq'] not in {'5min', '15min', '1h', '1D'} or v['label'] not in {'left', 'right'}:
            raise ValueError("invalid native grid")
        names.append(v['name'])
    if len(names) != len(set(names)):
        raise ValueError("duplicate snapshot variable")


@dataclass(frozen=True, slots=True)
class WeatherSnapshot:
    metadata: dict
    frame: pd.DataFrame
    dependency_hashes: tuple[tuple[str, str], ...]
    component_ids: tuple[str, ...] = ()
