# -*- coding: utf-8 -*-
"""周期检测纯数值核心（FFT / ACF / STL 季节强度）。

本模块是跨包复用的 L1 公开算法：decomposition 的残差频谱诊断与
data_process 的离线周期检测工具共用同一实现，禁止各自复制。

检测方法：
  1. FFT 幅度谱主导频率：对序列做 FFT，取正频段幅度最大的频率分量；
  2. 自相关函数（ACF）峰值间距：acf 在 lag=0 后首个显著局部极大值对应的滞后步数即周期；
  3. STL 季节性分解：用检测/配置的季节周期做 STL 分解，输出季节成分与强度指标。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf

# ---------------------------------------------------------------------------
# 默认参数（调用方可覆盖）
# ---------------------------------------------------------------------------
DEFAULT_MAX_LAGS = 2000
DEFAULT_TOP_N_PERIODS = 3
DEFAULT_MIN_ACF = 0.1  # ACF 周期候选的最小相关值（噪声产生的假峰通常 < 0.1）


# ---------------------------------------------------------------------------
# 核心检测算法
# ---------------------------------------------------------------------------
def fft_dominant_period(y: np.ndarray) -> dict:
    """FFT 幅度谱主导频率 -> 主导周期（样本数）。

    fftfreq 以采样间隔为单位，返回频率为「每周期的样本数」的倒数。
    """
    n = len(y)
    if n < 4:
        return {"dominant_period_samples": None, "dominant_amplitude": None}
    y_detrend = y - np.mean(y)
    spectrum = np.abs(fft(y_detrend))
    freqs = fftfreq(n)
    # 正频段（排除直流）
    mask = freqs > 0
    freqs_pos = freqs[mask]
    amp_pos = spectrum[mask]
    if len(freqs_pos) == 0:
        return {"dominant_period_samples": None, "dominant_amplitude": None}
    idx = int(np.argmax(amp_pos))
    period_samples = float(1.0 / freqs_pos[idx]) if freqs_pos[idx] > 0 else None
    return {
        "dominant_period_samples": period_samples,
        "dominant_amplitude": float(amp_pos[idx]),
    }


def acf_periods(y: np.ndarray, max_lags: int, top_n: int, min_acf: float = DEFAULT_MIN_ACF) -> list:
    """ACF 峰值间距 -> 周期候选列表。

    从 lag=1 起找局部极大值（避免 lag=0 的自相关=1 干扰），
    仅保留 acf > min_acf 的正相关显著峰（负值区与噪声假峰不是周期证据），
    返回按相关值排序的前 top_n 个滞后步数作为周期候选。
    """
    n = len(y)
    nlags = min(max_lags, n - 1)
    if nlags < 2:
        return []
    values = acf(y, nlags=nlags, fft=True)
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1] and values[i] > min_acf:
            peaks.append((int(i), float(values[i])))
    # 按相关值降序，返回 (lag, value)
    peaks.sort(key=lambda kv: kv[1], reverse=True)
    return peaks[:top_n]


def _stl_seasonal_component(y: np.ndarray, period: int) -> Optional[np.ndarray]:
    """STL 分解季节性成分；周期无效或分解失败时返回 None。"""
    if period is None or period < 2 or 2 * period > len(y):
        return None
    try:
        from statsmodels.tsa.seasonal import STL

        result = STL(y, period=period, robust=True).fit()
        return result.seasonal
    except (ValueError, FloatingPointError):
        return None


def detect_periodicity(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    *,
    max_lags: int = DEFAULT_MAX_LAGS,
    seasonal_period: Optional[int] = None,
    top_n_periods: int = DEFAULT_TOP_N_PERIODS,
    min_acf: float = DEFAULT_MIN_ACF,
) -> dict:
    """对 DataFrame 执行周期检测，返回结构化报告字典。"""
    frame = df[[time_col, target_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame = frame.sort_values(time_col)
    y = pd.to_numeric(frame[target_col], errors="coerce").dropna().to_numpy()
    if len(y) < 4:
        raise ValueError(f"目标列有效样本不足（<4）：{target_col}")

    # 采样间隔（秒）
    ts = frame[time_col].to_numpy()
    diff_seconds = float(np.median(np.diff(ts.astype("datetime64[ns]")).astype("int64")) / 1e9) if len(ts) > 1 else None

    # 线性去趋势：强趋势会淹没周期信号（ACF 单调下降无局部峰、FFT 主导=序列长度级）
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_detrended = y - (slope * x + intercept)

    report: dict[str, Any] = {
        "n_samples": int(len(y)),
        "sample_interval_seconds": diff_seconds,
    }

    # 1. FFT 主导周期（在去趋势序列上）
    fft_info = fft_dominant_period(y_detrended)
    report["fft_dominant_period_samples"] = fft_info["dominant_period_samples"]
    report["fft_dominant_amplitude"] = fft_info["dominant_amplitude"]
    if fft_info["dominant_period_samples"] and diff_seconds:
        report["fft_dominant_period_seconds"] = fft_info["dominant_period_samples"] * diff_seconds
        report["fft_dominant_period_days"] = fft_info["dominant_period_samples"] * diff_seconds / 86400.0

    # 2. ACF 周期候选（在去趋势序列上）
    acf_result = acf_periods(y_detrended, max_lags, top_n_periods, min_acf)
    report["acf_periods"] = [
        {"lag": lag, "acf": value} for lag, value in acf_result
    ]
    report["acf_dominant_period_samples"] = acf_result[0][0] if acf_result else None
    if acf_result and diff_seconds:
        report["acf_dominant_period_days"] = acf_result[0][0] * diff_seconds / 86400.0

    # 3. STL 季节性成分（可选）
    period = seasonal_period if seasonal_period is not None else (acf_result[0][0] if acf_result else None)
    seasonal = _stl_seasonal_component(y, period)
    report["stl_seasonal_period_used"] = period if seasonal is not None else None
    report["stl_has_seasonal_component"] = seasonal is not None
    if seasonal is not None:
        seasonal_std = float(np.std(seasonal))
        residual_std = float(np.std(y - seasonal))
        report["stl_seasonal_std"] = seasonal_std
        report["stl_residual_std"] = residual_std
        # legacy 指标（分母含趋势，非标准季节强度），保留供历史结果对照
        report["stl_seasonal_ratio"] = float(
            seasonal_std / (residual_std + 1e-12)
        )
        # 标准强度指标（FPP: Fs = max(0, 1 - Var(R)/Var(S+R)); Ft = max(0, 1 - Var(R)/Var(T+R))）
        from statsmodels.tsa.seasonal import STL as _STL

        decomposed = _STL(y, period=period, robust=True).fit()
        remainder = np.asarray(decomposed.resid, dtype=float)
        trend_component = np.asarray(decomposed.trend, dtype=float)
        seasonal_component = np.asarray(decomposed.seasonal, dtype=float)
        var = lambda v: float(np.var(np.asarray(v, dtype=float)))
        report["stl_seasonal_strength"] = max(
            0.0, 1.0 - var(remainder) / (var(seasonal_component + remainder) + 1e-12)
        )
        report["stl_trend_strength"] = max(
            0.0, 1.0 - var(remainder) / (var(trend_component + remainder) + 1e-12)
        )

    return report
