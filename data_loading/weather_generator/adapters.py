"""本地供应商宽表与 Open-Meteo 原响应的显式薄适配器。"""
import pandas as pd

from data_loading.weather_generator.assets import validate_normalized
from data_loading.weather_generator.contracts import utc_timestamp, validate_snapshot_metadata


OPEN_METEO_UNITS = {'°C': 'degC', '%': '%', 'km/h': 'km/h', 'm/s': 'm/s', 'hPa': 'hPa', 'mm': 'mm', 'W/m²': 'W/m2', '°': 'degree'}
OPEN_METEO_INTERVALS = {'shortwave_radiation': 'interval_mean', 'direct_radiation': 'interval_mean', 'diffuse_radiation': 'interval_mean', 'precipitation': 'interval_sum', 'rain': 'interval_sum'}
OPEN_METEO_POINTS = {'temperature_2m', 'dew_point_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'pressure_msl', 'surface_pressure'}


def vendor_frame(source, metadata, *, time_col, available_at_col):
    """不推断单位/版本；只有有证据的 forecast 允许快照级可得时间。输出宽表规范帧。"""
    validate_snapshot_metadata(metadata)
    if available_at_col is None:
        if metadata['data_kind'] != 'forecast':
            raise ValueError("actual/reanalysis requires explicit row availability evidence")
        evidence = metadata['evidence_class']
        field = {'received_snapshot': 'received_at', 'documented_release': 'issued_at'}.get(evidence)
        if field is None:
            raise ValueError("historical contract requires explicit row availability")
        available = [utc_timestamp(metadata[field]).isoformat()] * len(source)
    else:
        if available_at_col not in source.columns:
            raise ValueError(f"vendor source missing availability column: {available_at_col}")
        available = source[available_at_col].tolist()
    if time_col not in source.columns:
        raise ValueError(f"vendor source missing time column: {time_col}")
    wide = pd.DataFrame({'time': source[time_col].tolist()})
    for variable in metadata['variables']:
        if variable['column'] not in source.columns:
            raise ValueError(f"vendor source missing declared column: {variable['column']}")
        wide[variable['name']] = source[variable['column']].tolist()
    wide['available_at'] = available
    return validate_normalized(wide, metadata)


def open_meteo_frame(response, metadata, *, availability=None):
    """首版读取 hourly 原 JSON；UTC ISO 时间，保留响应原始单位，拒绝未知物理变量。"""
    validate_snapshot_metadata(metadata)
    if not isinstance(response, dict) or response.get('utc_offset_seconds') != 0:
        raise ValueError("Open-Meteo import requires explicit UTC response")
    if not all(k in response for k in ('hourly', 'hourly_units', 'latitude', 'longitude')):
        raise ValueError("Open-Meteo response lacks hourly units or location")
    for coordinate in ('latitude', 'longitude'):
        if type(response[coordinate]) not in (int, float) or response[coordinate] != metadata[coordinate]:
            raise ValueError(f"Open-Meteo response grid coordinate mismatch: {coordinate}")
    if response['hourly_units'].get('time') != 'iso8601':
        raise ValueError("Open-Meteo import requires iso8601 time encoding")
    frame = pd.DataFrame(response['hourly'])
    # API UTC 的 iso8601 可以省略 Z；只在 utc_offset_seconds 已核对为 0 后补其声明时区。
    frame['time'] = [pd.Timestamp(v).tz_localize('UTC').isoformat() if pd.Timestamp(v).tzinfo is None else utc_timestamp(v).isoformat() for v in frame['time']]
    for v in metadata['variables']:
        name = v['column']
        if name not in OPEN_METEO_POINTS and name not in OPEN_METEO_INTERVALS:
            raise ValueError(f"unsupported Open-Meteo physical variable: {name}")
        if OPEN_METEO_UNITS.get(response['hourly_units'].get(name)) != v['unit']:
            raise ValueError(f"Open-Meteo unit mismatch: {name}")
        expected = OPEN_METEO_INTERVALS.get(name, 'point')
        if v['semantics'] != expected or v['native_freq'] != '1h':
            raise ValueError(f"Open-Meteo interval semantics mismatch: {name}")
        if expected != 'point' and v['label'] != 'right':
            raise ValueError("Open-Meteo radiation/precipitation refer to preceding hour")
    available_at_col = None
    if availability is not None:
        if set(availability.columns) != {'time', 'available_at'} or len(availability.columns) != 2:
            raise ValueError('availability CSV requires exactly time,available_at')
        released = availability.copy(deep=True)
        released['time'] = released.time.map(utc_timestamp)
        released['available_at'] = released.available_at.map(utc_timestamp)
        expected = pd.DatetimeIndex(frame.time.map(utc_timestamp))
        if released.time.duplicated().any() or len(released) != len(expected) or set(released.time) != set(expected):
            raise ValueError('availability must cover response times exactly once')
        frame['available_at'] = [value.isoformat() for value in released.set_index('time').loc[expected, 'available_at']]
        available_at_col = 'available_at'
    return vendor_frame(frame, metadata, time_col='time', available_at_col=available_at_col)
