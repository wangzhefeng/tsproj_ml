# -*- coding: utf-8 -*-
"""频域/小波特征纯函数（trailing 窗，as-of 安全）。

所有函数只消费调用方传入的可见历史窗口，不接触时间轴；
as-of 合同由调用方（FeatureCompiler）保证。返回的 dict 键为
特征后缀，编译器负责拼接完整特征名列。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pywt


def signal_entropy(values: np.ndarray | Sequence[float]) -> float:
    """Shannon 熵：p = |y| / sum|y|，H = -sum(p * log2(p))；全零序列返回 0.0。"""
    y = np.abs(np.asarray(values, dtype=float))
    total = float(y.sum())
    if total == 0.0:
        return 0.0
    p = y / total
    return float(-np.sum(p * np.log2(p + 1e-12)))


def normalize_band_periods(value: object) -> tuple[tuple[int, int], ...]:
    """校验频带周期区间序列：每段为 [lo, hi) 正整数对（单位：步）。"""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("fourier.band_periods must be a sequence of [lo, hi) pairs")
    bands: list[tuple[int, int]] = []
    for pair in value:
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise TypeError("fourier.band_periods entries must be [lo, hi) pairs")
        lo, hi = pair
        for bound in (lo, hi):
            if isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0:
                raise ValueError("fourier.band_periods bounds must be positive integers")
        if lo >= hi:
            raise ValueError(f"fourier.band_periods requires lo < hi, got [{lo}, {hi})")
        bands.append((int(lo), int(hi)))
    return tuple(bands)


def fourier_features(
    values: np.ndarray | Sequence[float],
    *,
    top_k: int = 5,
    band_periods: Sequence[tuple[int, int]] = (),
) -> dict[str, float]:
    """trailing 窗 FFT 特征。

    返回键：
      amp_1..k / freq_1..k / phase_1..k  —— 振幅最大的 k 个正频成分
        （振幅为单边幅值 2|X|/N，频率单位 cycles/step，相位单位 rad）
      centroid                             —— 功率加权谱质心（cycles/step）
      bandenergy_1..m                      —— 各频带能量占比（按周期区间，
        周期 = 1/频率，区间 [lo, hi) 以步计）
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("fourier.top_k must be a positive integer")
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < 2:
        raise ValueError("fourier requires at least 2 samples")
    y = y - y.mean()

    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # 排除 DC bin（已被去均值置零），只保留正频
    amps = 2.0 * np.abs(spectrum[1:]) / n
    pos_freqs = freqs[1:]
    power = np.abs(spectrum[1:]) ** 2
    phases = np.angle(spectrum[1:])

    if top_k > len(amps):
        raise ValueError(
            f"fourier.top_k={top_k} exceeds positive frequency bins ({len(amps)}); "
            "use a longer window or smaller top_k"
        )
    top_indices = np.flip(np.argsort(amps, kind="stable"))[:top_k]

    features: dict[str, float] = {}
    for rank, index in enumerate(top_indices, start=1):
        features[f"amp_{rank}"] = float(amps[index])
        features[f"freq_{rank}"] = float(pos_freqs[index])
        features[f"phase_{rank}"] = float(phases[index])

    total_power = float(power.sum())
    if total_power > 0.0:
        features["centroid"] = float(np.sum(pos_freqs * power) / total_power)
    else:
        features["centroid"] = 0.0

    for band_index, (lo, hi) in enumerate(band_periods, start=1):
        with np.errstate(divide="ignore"):
            periods = np.where(pos_freqs > 0.0, 1.0 / pos_freqs, np.inf)
        mask = (periods >= lo) & (periods < hi)
        ratio = float(power[mask].sum() / total_power) if total_power > 0.0 else 0.0
        features[f"bandenergy_{band_index}"] = ratio
    return features


def wavelet_energy_features(
    values: np.ndarray | Sequence[float],
    *,
    wavelet: str = "db4",
    level: int = 3,
) -> dict[str, float]:
    """trailing 窗 DWT 各分量能量占比。

    返回键：a{level}（近似）与 d{level}..d1（细节），值 sum 恒为 1
    （零能量序列全部为 0）。
    """
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError("wavelet.level must be a positive integer")
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except ValueError as exc:
        raise ValueError(f"unknown wavelet: {wavelet!r}") from exc
    y = np.asarray(values, dtype=float)
    max_level = pywt.dwt_max_level(len(y), wavelet_obj.dec_len)
    if level > max_level:
        raise ValueError(
            f"wavelet.level={level} exceeds max level {max_level} "
            f"for window length {len(y)} with wavelet {wavelet!r}"
        )
    coeffs = pywt.wavedec(y, wavelet_obj, level=level)
    energies = [float(np.sum(np.square(coeff))) for coeff in coeffs]
    total = sum(energies)

    features: dict[str, float] = {}
    # coeffs = [cA_level, cD_level, cD_{level-1}, ..., cD_1]
    names = [f"a{level}"] + [f"d{j}" for j in range(level, 0, -1)]
    for name, energy in zip(names, energies):
        features[name] = energy / total if total > 0.0 else 0.0
    return features


__all__ = [
    "fourier_features",
    "normalize_band_periods",
    "signal_entropy",
    "wavelet_energy_features",
]
