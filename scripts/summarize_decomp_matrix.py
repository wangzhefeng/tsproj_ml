# -*- coding: utf-8 -*-
"""最终分解方法矩阵汇总（一次性脚本）。"""
from pathlib import Path
import collections
import yaml


def deep_get(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


matrix = collections.Counter()
for scene in ["aidc_ess_selfuse_load", "aidc_load_15min_daily", "aidc_load_15min_rolling",
              "aidc_load_15min_short", "aidc_load_month", "aidc_power_month"]:
    for p in sorted(Path("config", scene).rglob("*.yaml")):
        data = yaml.safe_load(p.read_text())
        if not isinstance(data, dict) or "base_config" not in data:
            continue
        pre = deep_get(data, "overrides", "preprocessing") or {}
        m = pre.get("decomposition_method", "none")
        if m == "none":
            continue
        deg = pre.get("decomposition_trend_degree", 1)
        tf = pre.get("decomposition_trend_forecast", "polynomial")
        periods = pre.get("decomposition_periods", []) or []
        if m == "linear":
            variant = "quadratic" if deg == 2 else ("damped" if tf == "damped" else "linear")
        elif m == "stl":
            variant = f"stl{periods[0]}"
        else:
            joined = "+".join(str(x) for x in periods)
            variant = f"mstl{joined}"
        matrix[(scene, variant)] += 1

for (scene, v), cnt in sorted(matrix.items()):
    print(f"  {scene}: {v} x {cnt}")
