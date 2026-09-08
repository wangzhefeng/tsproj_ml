"""从共享供应商宽表（extracted/actual/weather_in_20250101_20260831.csv）严格生成各场景天气数据。

当前场景均为历史评估，history 覆盖 2025-01-01 至 2026-08-31；训练读 rt_，
测试预测读同一 history 的 pred_。不生成伪 future，不删除旧资产。
真正未来资产由实际预报另行提供，只使用预报列。既有离线缺口、湿度派生、
完整覆盖聚合规则不变；不修改共享源，不把 ERA5 补值宣称为实际发布预报。
ESS 输出到 exogenous_weather_raw/；联通由独立场景准备脚本生成。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'dataset/shared/weather/extracted/actual/weather_in_20250101_20260831.csv'
REPORT = ROOT / '.hermes/plans/weather-scenario-build-report.json'
SOURCE_REPAIR_REPORT = ROOT / '.hermes/plans/weather-may-gap-fill-report.json'

RT = ['rt_tt2', 'rt_dt', 'rt_sh', 'rt_ssr', 'rt_ws10', 'rt_rain', 'rt_ps', 'rt_uu', 'rt_vv']
PRED = ['pred_ssrd', 'pred_tsdsr', 'pred_s_tsr', 'pred_ws10', 'pred_wd10', 'pred_tt2',
        'pred_rh', 'pred_ps', 'pred_tcc', 'pred_rain', 'pred_ws100', 'pred_wd100', 'pred_dt']
HISTORY_START, HISTORY_END = pd.Timestamp('2025-01-01 00:00:00'), pd.Timestamp('2026-08-31 23:59:59')

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


def fill_gaps(hourly):
    """缺口策略 A（用户 2026-09-07 裁决）：≤3h 连续缺测线性插值；>3h 用 ERA5 替代（单位换算）；
    ERA5 未覆盖时段保持 NaN。返回 (frame, audit)。"""
    era5 = pd.read_csv(ERA5_PATH)
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


def publish(df, dest, extra_meta):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(df.to_csv(index=False))
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    source_repairs = []
    if SOURCE_REPAIR_REPORT.is_file():
        repair = json.loads(SOURCE_REPAIR_REPORT.read_text())
        if repair.get('source_sha256_after') == source_sha256:
            source_repairs.append({
                'report': str(SOURCE_REPAIR_REPORT.relative_to(ROOT)),
                'window': repair.get('window'),
                'changed_cells': repair.get('changed_cells'),
                'semantic_status': repair.get('semantic_status'),
            })
    six_features_repair = SOURCE.with_suffix('.six_features_repair.json')
    if six_features_repair.is_file():
        repair = json.loads(six_features_repair.read_text())
        if repair.get('source_sha256_after') == source_sha256:
            source_repairs.append({
                'report': str(six_features_repair.relative_to(ROOT)),
                'sha256': hashlib.sha256(six_features_repair.read_bytes()).hexdigest(),
                'semantic_status': repair['semantic_status'],
            })
    meta = {
        'file': dest.name, 'sha256_file': hashlib.sha256(dest.read_bytes()).hexdigest(),
        'rows': len(df), 'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': source_sha256, 'source_repairs': source_repairs,
        'builder': 'scripts/build_scenario_weather.py',
        'rules': 'audited source repairs plus builder gap policy; complete-coverage aggregates else NaN; no available_at in data; '
                 'both rt_ and pred_ families preserved; training uses rt_, inference uses pred_',
        **extra_meta,
    }
    (dest.parent / (dest.stem + '.meta.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {'file': str(dest.relative_to(ROOT)), 'rows': len(df), 'sha256': meta['sha256_file']}


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


AGG_PAIRS = [('rt_tt2', 'rt_tt2', 'mean'), ('rt_tt2_max', 'rt_tt2', 'max'), ('rt_tt2_min', 'rt_tt2', 'min'),
             ('cal_rh', 'cal_rh', 'mean'), ('rt_ssr', 'rt_ssr', 'sum'),
             ('rt_ws10', 'rt_ws10', 'mean'), ('rt_dt', 'rt_dt', 'mean')]
PRED_AGG_PAIRS = [('pred_tt2', 'pred_tt2', 'mean'), ('pred_tt2_max', 'pred_tt2', 'max'), ('pred_tt2_min', 'pred_tt2', 'min'),
                  ('pred_rh', 'pred_rh', 'mean'), ('pred_ssrd', 'pred_ssrd', 'sum'),
                  ('pred_ws10', 'pred_ws10', 'mean'), ('pred_dt', 'pred_dt', 'mean')]


def main():
    raw = pd.read_csv(SOURCE)
    raw['ts'] = pd.to_datetime(raw['ts'])
    raw = raw.sort_values('ts').set_index('ts')
    assert not raw.index.duplicated().any()
    for col in RT + PRED:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
    # 补全完整小时网格（源文件缺 209 个缺测行），再按裁决策略 A 填补/替代
    full_grid = pd.date_range(raw.index.min(), raw.index.max(), freq='1h')
    raw = raw.reindex(full_grid)
    hourly, gap_audit = fill_gaps(raw)
    hourly['cal_rh'] = calc_rh(hourly['rt_tt2'], hourly['rt_dt'])
    # pred_dt：pred 无露点列，由 pred_tt2+pred_rh 经 Magnus 反解（显式声明的派生）
    hourly['pred_dt'] = invert_dewpoint(hourly['pred_tt2'], hourly['pred_rh'])
    hist_slice = hourly.loc[HISTORY_START:HISTORY_END]
    report = []

    # ---------------- 15min ×3 ----------------
    for family in ('aidc_load_15min_daily', 'aidc_load_15min_rolling', 'aidc_load_15min_short'):
        hist = resample_hold(hist_slice, '15min', HISTORY_START, HISTORY_END)
        report.append({'scenario': family, **publish(
            hist, ROOT / f'dataset/{family}/weather_history_15min_20250101_20260831.csv',
            {'role': 'history', 'freq': '15min'})})

    # ---------------- ESS 5min（exogenous_weather_raw/；2026-09-07 裁决：不构造统计特征，只用原始特征） ----------------
    ess_dir = ROOT / 'dataset/aidc_ess_selfuse_load/exogenous_weather_raw'

    # 无窗口特征 → 无预热 NaN，保留用户指定的完整起点 2025-01-01 00:00
    hist5 = resample_hold(hist_slice, '5min', HISTORY_START, HISTORY_END)
    report.append({'scenario': 'aidc_ess_selfuse_load', **publish(
        hist5, ess_dir / 'weather_history_5min_20250101_20260831.csv', {'role': 'history', 'freq': '5min'})})

    # ---------------- power_month 日/月（rt_ 统计 + pred_ 统计并列） ----------------
    for freq_dir, rule, hist_name in (
        ('freq_1day', '1D', 'weather_history_1day_20250101_20260831.csv'),
        ('freq_1month', '1ME', 'weather_history_1month_20250131_20260831.csv'),
    ):
        hist_agg, hist_inc = aggregate(hist_slice[list(dict.fromkeys(c for _, c, _ in AGG_PAIRS))], rule, AGG_PAIRS)
        hist_pred, hist_pred_inc = aggregate(hist_slice[list(dict.fromkeys(c for _, c, _ in PRED_AGG_PAIRS))], rule, PRED_AGG_PAIRS)
        hist_out = hist_agg.merge(hist_pred.drop(columns=[]), on='ts')
        # rt_ 侧（模型训练列）必须完整；pred_ 侧允许 NaN（ignored 声明，推理触及由 registry 请求级 RAISE）
        assert not hist_agg.drop(columns=['ts']).isna().any(axis=None), f'{freq_dir} history rt_ 聚合后仍含 NaN'

        report.append({'scenario': f'aidc_power_month/{freq_dir}', **publish(
            hist_out, ROOT / f'dataset/aidc_power_month/{freq_dir}/{hist_name}',
            {'role': 'history', 'freq': rule, 'incomplete_rows_nan_pred': hist_pred_inc})})


    REPORT.write_text(json.dumps({'status': 'ok', 'outputs': report, 'gap_fill_audit': gap_audit}, ensure_ascii=False, indent=2))
    print(json.dumps({'status': 'ok', 'count': len(report)}, ensure_ascii=False))
    for row in report:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
