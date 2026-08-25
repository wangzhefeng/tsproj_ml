# -*- coding: utf-8 -*-
"""确认 rolling baseline 与 weather-date 组全部关闭分解（一次性脚本）。"""
from pathlib import Path
import yaml


def deep_get(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


bad = 0
total_b = total_w = 0
for p in Path("config/aidc_load_15min_rolling").rglob("*.yaml"):
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict) or "base_config" not in data:
        continue
    m = deep_get(data, "overrides", "preprocessing", "decomposition_method") or "none"
    s = str(p)
    if "/baseline/" in s:
        total_b += 1
        if m != "none":
            bad += 1
            print(f"BASELINE VIOLATION {p}: {m}")
    if "add_exogenous_weather_date/" in s:
        total_w += 1
        if m != "none":
            bad += 1
            print(f"WEATHER VIOLATION {p}: {m}")
print(f"baseline={total_b} weather={total_w} violations={bad}")
raise SystemExit(1 if bad else 0)
