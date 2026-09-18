"""公共天气源治理：只读原料，统一补缺和派生，发布共享小时资产。

不枚举预测场景；场景适配脚本从 config/ 调用公开的读取、重采样和发布函数。
保留现有供应商格式及离线修补语义，不把插值/再分析替代当作在线可得证据。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / 'dataset/shared/weather/extracted/actual'
SOURCE_GLOB = 'weather_in_*.csv'
DEFAULT_PROCESSED = ROOT / 'dataset/shared/weather/processed/weather_hourly.csv'
SOURCE_REPAIR_REPORT = ROOT / '.hermes/plans/weather-may-gap-fill-report.json'

RT = ['rt_tt2', 'rt_dt', 'rt_sh', 'rt_ssr', 'rt_ws10', 'rt_rain', 'rt_ps', 'rt_uu', 'rt_vv']
PRED = ['pred_ssrd', 'pred_tsdsr', 'pred_s_tsr', 'pred_ws10', 'pred_wd10', 'pred_tt2',
        'pred_rh', 'pred_ps', 'pred_tcc', 'pred_rain', 'pred_ws100', 'pred_wd100', 'pred_dt']
SIX_MAPPING = {'rt_tt2': 'pred_tt2', 'cal_rh': 'pred_rh', 'rt_ssr': 'pred_ssrd',
               'rt_ws10': 'pred_ws10', 'rt_ps': 'pred_ps', 'rt_rain': 'pred_rain'}

ERA5_PATH = ROOT / 'dataset/shared/weather/extracted/api/open_meteo_era5_hourly_20250101_20250902.csv'
# ERA5 → 供应商列的单位换算映射（ERA5 无 cloud_cover/100m 风：pred_tcc/pred_ws100/pred_wd100 保持 NaN）
ERA5_MAP = {
    'rt_tt2': ('temperature_2m', 'K_from_degC'), 'rt_dt': ('dew_point_2m', 'K_from_degC'),
    'rt_sh': ('relative_humidity_2m', 'identity'), 'rt_ssr': ('shortwave_radiation', 'identity'),
    'rt_ws10': ('wind_speed_10m', 'identity'), 'rt_rain': ('rain', 'identity'),
    'rt_ps': ('surface_pressure', 'Pa_from_hPa'),
    'pred_tt2': ('temperature_2m', 'K_from_degC'), 'pred_rh': ('relative_humidity_2m', 'identity'),
    'pred_ssrd': ('shortwave_radiation', 'identity'), 'pred_tsdsr': ('direct_radiation', 'identity'),
    'pred_s_tsr': ('diffuse_radiation', 'identity'), 'pred_ws10': ('wind_speed_10m', 'identity'),
    'pred_wd10': ('wind_direction_10m', 'identity'), 'pred_rain': ('rain', 'identity'),
    'pred_ps': ('surface_pressure', 'Pa_from_hPa'),
}


def era5_convert(values, rule):
    if rule == 'K_from_degC':
        return values + 273.15
    if rule == 'Pa_from_hPa':
        return values * 100.
    return values


def fill_gaps(hourly, era5_path=None):
    """缺口策略 A（用户 2026-09-07 裁决）：≤3h 连续缺测线性插值；>3h 用 ERA5 替代（单位换算）；
    ERA5 未覆盖时段保持 NaN。返回 (frame, audit)。"""
    era5 = pd.read_csv(ERA5_PATH if era5_path is None else era5_path)
    era5['ts'] = pd.to_datetime(era5['ts'])
    era5 = era5.set_index('ts')
    filled = hourly.copy()
    audit = {}
    for col in [c for c in RT + PRED if c in filled.columns]:
        series = filled[col]
        isnan = series.isna()
        if not isnan.any():
            continue
        run_lengths = isnan.groupby(isnan.ne(isnan.shift()).cumsum()).transform('sum')
        short_mask = isnan & (run_lengths <= 3)
        long_mask = isnan & ~short_mask
        interpolated = series.interpolate(method='linear', limit_area='inside')
        entry = {'interpolated_hours': [str(t) for t in series.index[short_mask]],
                 'era5_substituted_hours': [], 'remaining_nan_hours': []}
        filled.loc[short_mask, col] = interpolated[short_mask]
        if long_mask.any() and col in ERA5_MAP:
            era5_col, rule = ERA5_MAP[col]
            for ts in series.index[long_mask]:
                if ts in era5.index and pd.notna(era5.at[ts, era5_col]):
                    filled.at[ts, col] = float(era5_convert(era5.at[ts, era5_col], rule))
                    entry['era5_substituted_hours'].append(str(ts))
        # 用户 2026-09-07 裁决：辐射列在全年无日照时段（20:00-02:00）的缺测记 0（物理定性，非插值）
        entry['night_zero_hours'] = []
        if col in {'rt_ssr', 'pred_ssrd', 'pred_tsdsr', 'pred_s_tsr'}:
            still_nan = filled[col].isna()
            dark = still_nan & filled.index.hour.isin({20, 21, 22, 23, 0, 1, 2})
            if dark.any():
                filled.loc[dark, col] = 0.0
                entry['night_zero_hours'] = [str(t) for t in series.index[dark]]
        still = filled[col].isna()
        entry['remaining_nan_hours'] = [str(t) for t in series.index[still]]
        audit[col] = entry
    return filled, audit



def calc_rh(tt2_k, dt_k):
    """Magnus-Tetens：Kelvin → %。用户 2026-09-07 裁决：露点>气温的轻微过饱和（原始噪声）RH 截断 100，单元格记审计。"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rh = 100. * np.exp(17.2693 * t_dew / (237.29 + t_dew) - 17.2693 * t_air / (237.29 + t_air))
    rh = pd.Series(rh, index=tt2_k.index, dtype=float)
    supersaturated = (t_dew > t_air) & np.isfinite(rh)
    clipped = rh.copy()
    clipped[supersaturated] = 100.0
    clipped[~np.isfinite(rh)] = np.nan
    clipped.attrs['supersaturated_clipped_hours'] = [str(t) for t in rh.index[supersaturated]]
    return clipped


def invert_dewpoint(tt2_k, rh_pct):
    """Magnus 反解：气温 K + 相对湿度 % → 露点 K；非法输入 NaN。"""
    t_air = tt2_k - 273.15
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        e = (rh_pct / 100.) * 6.1078 * np.exp(17.2693 * t_air / (237.29 + t_air))
        ratio = e / 6.1078
        td = 237.29 * np.log(ratio) / (17.2693 - np.log(ratio))
    td = pd.Series(td, index=tt2_k.index, dtype=float)
    td[(ratio <= 0) | ~np.isfinite(td)] = np.nan
    return td + 273.15


def load_sources(source_dir=None):
    """加载 extracted/actual/ 全部增量分片，按 ts 零冲突合并。

    合同：分片命名 weather_in_<起始yyyymmdd>_<截止yyyymmdd>.csv（起止为含数据日期，
    inclusive）；分片间允许时间重叠，但同 ts 同列不得存在两个不同的非空值（RAISE，
    不 keep-last）；一方为空另一方非空则采纳非空值。返回 (分片信息列表, 合并后小时表)。
    """
    source_dir = SOURCE_DIR if source_dir is None else Path(source_dir)
    paths = sorted(source_dir.glob(SOURCE_GLOB))
    if not paths:
        raise FileNotFoundError(f'{source_dir} 下无 {SOURCE_GLOB} 分片')
    merged = None
    conflicts = []
    sources = []
    for path in paths:
        df = pd.read_csv(path)
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.sort_values('ts').set_index('ts')
        dup = df.index.duplicated()
        if dup.any():
            raise ValueError(f'{path.name} 内部 ts 重复: {[str(t) for t in df.index[dup]]}')
        for col in RT + PRED:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:  # 测试/外部目录：保留绝对路径
            rel = str(path)
        sources.append({'file': rel,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'rows': len(df),
                        'span': [str(df.index.min()), str(df.index.max())]})
        if merged is None:
            merged = df
            continue
        overlap = merged.index.intersection(df.index)
        for ts in overlap:
            for col in df.columns:
                old, new = merged.at[ts, col], df.at[ts, col]
                if pd.notna(old) and pd.notna(new) and not np.isclose(old, new):
                    conflicts.append({'ts': str(ts), 'column': col,
                                      'existing': float(old), 'incoming': float(new),
                                      'incoming_source': path.name})
                elif pd.isna(old) and pd.notna(new):
                    merged.at[ts, col] = new
        new_rows = df.loc[df.index.difference(merged.index)]
        merged = pd.concat([merged, new_rows])
    if conflicts:
        raise ValueError(f'分片零冲突合并失败，共 {len(conflicts)} 个冲突单元格: '
                         + json.dumps(conflicts[:10], ensure_ascii=False))
    return sources, merged.sort_index()


def publish(df, dest, extra_meta, sources):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(df.to_csv(index=False))
    shard_shas = {s['sha256'] for s in sources}
    source_repairs = []
    if SOURCE_REPAIR_REPORT.is_file():
        repair = json.loads(SOURCE_REPAIR_REPORT.read_text())
        if repair.get('source_sha256_after') in shard_shas:
            source_repairs.append({
                'report': relative_path(SOURCE_REPAIR_REPORT),
                'window': repair.get('window'),
                'changed_cells': repair.get('changed_cells'),
                'semantic_status': repair.get('semantic_status'),
            })
    # 逐分片绑定离线修补 sidecar（<同名>.six_features_repair.json，按分片自身 sha256 核对）
    for src in sources:
        sidecar = (ROOT / src['file']).with_suffix('.six_features_repair.json')
        if sidecar.is_file():
            repair = json.loads(sidecar.read_text())
            if repair.get('source_sha256_after') == src['sha256']:
                source_repairs.append({
                    'report': relative_path(sidecar),
                    'sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                    'semantic_status': repair['semantic_status'],
                })
    meta = {
        'file': dest.name, 'sha256_file': hashlib.sha256(dest.read_bytes()).hexdigest(),
        'rows': len(df), 'sources': sources, 'source_repairs': source_repairs,
        'builder': 'scripts/build_scenario_weather.py',
        'rules': 'audited source repairs plus builder gap policy; complete-coverage aggregates else NaN; no available_at in data; '
                 'both rt_ and pred_ families preserved; training uses rt_, inference uses pred_',
        **extra_meta,
    }
    (dest.parent / (dest.stem + '.meta.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {'file': relative_path(dest), 'rows': len(df), 'sha256': meta['sha256_file']}


def resample_hold(hourly, freq, start, end):
    """小时 → 细网格 hold：仅在原生小时气泡内保持，缺口不传播。"""
    last_hour = pd.Timestamp(end).floor('1h')
    index = pd.date_range(start, min(pd.Timestamp(end), last_hour + pd.Timedelta('1h') - pd.Timedelta(freq)), freq=freq)
    out = hourly.reindex(index.floor('1h'))
    out.index = index
    out.index.name = 'ts'
    return out.reset_index()


def aggregate(frame, rule, pairs):
    """完整覆盖聚合；pairs: (输出列, 源列, 聚合方式)。缺 1 小时即整行 NaN。"""
    counts = frame.notna().resample(rule).sum()
    if rule == '1D':
        expected = pd.Series(24, index=counts.index)
    else:  # 1ME
        expected = pd.Series([pd.Timestamp(p).days_in_month * 24 for p in counts.index], index=counts.index)
    complete = counts.eq(expected, axis=0).all(axis=1)
    agg = pd.DataFrame(index=counts.index)
    for out_col, src_col, how in pairs:
        if how == 'sum':
            agg[out_col] = frame[src_col].resample(rule).sum(min_count=1)
        else:
            agg[out_col] = getattr(frame[src_col].resample(rule), how)()
    agg.loc[~complete, list(agg.columns)] = np.nan
    return agg.reset_index(names='ts'), int((~complete).sum())


def repair_source_intervals(hourly):
    """已授权供应商源缺口：9/16 内部缺口线性插值，不回写原始分片。"""
    filled = hourly.copy()
    audit = []
    for col, end in (('rt_tt2', '23:00'), ('rt_dt', '23:00'),
                     ('rt_ws10', '21:00'), ('rt_rain', '21:00')):
        index = pd.date_range('2026-09-16 15:00', f'2026-09-16 {end}', freq='1h')
        if col not in filled or not index.isin(filled.index).all():
            continue
        segment = filled.loc[index, col]
        if not np.isfinite(segment.iloc[[0, -1]].to_numpy(dtype=float)).all():
            raise ValueError(f'{col}: 源插值缺少授权窗口的双端锚点')
        interpolated = segment.interpolate(method='time', limit_area='inside')
        for ts in segment.index[segment.isna()]:
            left = segment.loc[:ts].last_valid_index()
            right = segment.loc[ts:].first_valid_index()
            value = float(interpolated.at[ts])
            filled.at[ts, col] = value
            audit.append({'ts': str(ts), 'column': col, 'old_value': None, 'new_value': value,
                          'method': 'linear_interpolation', 'left_anchor': str(left),
                          'left_value': float(segment.at[left]), 'right_anchor': str(right),
                          'right_value': float(segment.at[right]), 'dependency_end': str(right)})
    if audit:
        rh = calc_rh(filled['rt_tt2'], filled['rt_dt'])
        repaired_times = {row['ts'] for row in audit if row['column'] in ('rt_tt2', 'rt_dt')}
        for timestamp in sorted(repaired_times):
            ts = pd.Timestamp(timestamp)
            if pd.isna(filled.at[ts, 'cal_rh']):
                filled.at[ts, 'cal_rh'] = rh.at[ts]
                audit.append({'ts': timestamp, 'column': 'cal_rh', 'old_value': None,
                              'new_value': float(rh.at[ts]), 'method': 'derived_relative_humidity',
                              'inputs': ['rt_tt2', 'rt_dt'], 'dependency_end': '2026-09-16 23:00:00'})
    return filled, audit


def relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def require_finite(frame, columns):
    if frame.empty or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError('天气必需列缺失或非有限')


def read_processed(path):
    """读取共享资产并校验内容身份；round_trip 避免二次CSV解析改变浮点值。"""
    path = Path(path).resolve()
    meta_path = path.with_suffix('.meta.json')
    metadata = json.loads(meta_path.read_text())
    if metadata.get('role') != 'shared_processed':
        raise ValueError('expected shared processed weather')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256_file']:
        raise ValueError('processed weather hash mismatch')
    frame = pd.read_csv(path, float_precision='round_trip')
    times = pd.DatetimeIndex(pd.to_datetime(frame.pop('ts')))
    if (times.empty or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing
            or not times.equals(pd.date_range(times[0], times[-1], freq='1h'))):
        raise ValueError('processed weather requires regular hourly timeline')
    frame.index = times
    require_finite(frame, list(SIX_MAPPING) + list(SIX_MAPPING.values()))
    metadata['processed_asset'] = {'file': relative_path(path), 'sha256': metadata['sha256_file'],
                                   'metadata': relative_path(meta_path),
                                   'metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest()}
    return frame, metadata


def build_shared(source_dir, era5_path, output):
    source_dir, era5_path, output = Path(source_dir).resolve(), Path(era5_path).resolve(), Path(output).resolve()
    if (source_dir in output.parents or output == era5_path
            or (ROOT / 'dataset/shared/weather/extracted').resolve() in output.parents):
        raise ValueError('processed output must not overwrite raw weather')
    sources, raw = load_sources(source_dir)
    full_grid = pd.date_range(raw.index.min(), raw.index.max(), freq='1h')
    raw = raw.reindex(full_grid)
    hourly, gap_audit = fill_gaps(raw, era5_path)
    hourly['cal_rh'] = calc_rh(hourly['rt_tt2'], hourly['rt_dt'])
    hourly['pred_dt'] = invert_dewpoint(hourly['pred_tt2'], hourly['pred_rh'])
    hourly, interpolation = repair_source_intervals(hourly)
    require_finite(hourly, list(SIX_MAPPING) + list(SIX_MAPPING.values()))
    meta = {'role': 'shared_processed', 'freq': '1h', 'gap_fill_audit': gap_audit,
            'offline_interpolation': interpolation,
            'era5': {'file': relative_path(era5_path), 'sha256': hashlib.sha256(era5_path.read_bytes()).hexdigest()},
            'start': str(hourly.index.min()), 'end': str(hourly.index.max()),
            'availability_assumption': 'offline repairs; no supplier vintage evidence; not verified ex-ante'}
    return publish(hourly.reset_index(names='ts'), output, meta, sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--era5-path', type=Path, default=ERA5_PATH)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_shared(args.source_dir, args.era5_path, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
