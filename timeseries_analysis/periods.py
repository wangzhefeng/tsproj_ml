"""FFT/ACF 周期检测与 STL 强度；无 DataFrame 编排、项目依赖或 IO。"""
import numpy as np
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL

DEFAULT_MAX_LAGS = 2000
DEFAULT_TOP_N_PERIODS = 3
DEFAULT_MIN_ACF = 0.1


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


def fft_top_periods(y: np.ndarray, top_k: int) -> list:
    """FFT 幅度谱前 top_k 个主导频率 -> [(period_samples, frequency, amplitude)]。

    排除直流分量；调用方负责去趋势（detect_periodicity 先线性去趋势再调用）。
    幅度为原始谱幅值（与 fft_dominant_period 同口径），仅供相对排序诊断。
    """
    n = len(y)
    if n < 4 or top_k < 1:
        return []
    y_detrend = y - np.mean(y)
    spectrum = np.abs(fft(y_detrend))
    freqs = fftfreq(n)
    mask = freqs > 0
    freqs_pos = np.asarray(freqs[mask], dtype=float)
    amp_pos = np.asarray(spectrum[mask], dtype=float)
    if len(freqs_pos) == 0:
        return []
    order = np.argsort(amp_pos)[::-1][:top_k]
    return [
        (float(1.0 / freqs_pos[i]), float(freqs_pos[i]), float(amp_pos[i]))
        for i in order
    ]


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


def stl_strength(y: np.ndarray, period: int | None) -> dict:
    report = {"stl_seasonal_period_used": None, "stl_has_seasonal_component": False}
    if period is None or period < 2 or 2 * period > len(y):
        return report
    try:
        decomposed = STL(y, period=period, robust=True).fit()
    except (ValueError, FloatingPointError):
        return report
    seasonal = np.asarray(decomposed.seasonal, dtype=float)
    remainder = np.asarray(decomposed.resid, dtype=float)
    trend = np.asarray(decomposed.trend, dtype=float)
    seasonal_std = float(np.std(seasonal))
    residual_std = float(np.std(y - seasonal))
    report.update({
        "stl_seasonal_period_used": period,
        "stl_has_seasonal_component": True,
        "stl_seasonal_std": seasonal_std,
        "stl_residual_std": residual_std,
        "stl_seasonal_ratio": seasonal_std / (residual_std + 1e-12),
        "stl_seasonal_strength": max(0.0, 1.0 - float(np.var(remainder)) / (float(np.var(seasonal + remainder)) + 1e-12)),
        "stl_trend_strength": max(0.0, 1.0 - float(np.var(remainder)) / (float(np.var(trend + remainder)) + 1e-12)),
    })
    return report
