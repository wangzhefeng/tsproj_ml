# -*- coding: utf-8 -*-
"""完整分解矩阵重新汇总（含 ensemble，一次性脚本）。"""
from pathlib import Path
import collections
import yaml


def deep_get(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


SUFFIXES = ["_decomp_linear", "_decomp_stl96", "_decomp_stl288", "_decomp_stl7",
            "_decomp_mstl96-672", "_decomp_mstl288-2016", "_decomp_quadratic",
            "_decomp_damped", "_quadratic", "_damped", "_linear", "_none"]

for scene in ["aidc_ess_selfuse_load", "aidc_load_15min_daily", "aidc_load_15min_rolling",
              "aidc_load_15min_short", "aidc_load_month", "aidc_power_month"]:
    print(f"===== {scene} =====")
    detail = collections.defaultdict(list)
    for p in sorted(Path("config", scene).rglob("*.yaml")):
        s = str(p)
        if "add_decomposition" not in s and "/decomposition/" not in s and "/ensemble/" not in s:
            continue
        data = yaml.safe_load(p.read_text())
        if not isinstance(data, dict) or "base_config" not in data:
            continue
        pre = deep_get(data, "overrides", "preprocessing") or {}
        m = pre.get("decomposition_method", "none")
        deg = pre.get("decomposition_trend_degree", 1)
        tf = pre.get("decomposition_trend_forecast", "polynomial")
        periods = pre.get("decomposition_periods", []) or []
        if m == "linear":
            variant = "quadratic" if deg == 2 else ("damped" if tf == "damped" else "linear")
        elif m == "stl":
            variant = f"stl{periods[0]}"
        elif m == "mstl":
            variant = "mstl" + "+".join(str(x) for x in periods)
        else:
            variant = m
        subdir = p.parts[3] if len(p.parts) > 4 else "decomposition"
        if "/ensemble/" in s:
            subdir = "ensemble"
        stem = p.stem
        for suf in SUFFIXES:
            stem = stem.replace(suf, "")
        detail[(subdir, variant)].append(stem)
    for (subdir, variant), models in sorted(detail.items()):
        uniq = sorted(set(models))
        print(f"  {subdir} | {variant}: {len(models)} 个 ({len(uniq)} 模型)")
        if len(uniq) <= 12:
            print(f"    {uniq}")
    print()
