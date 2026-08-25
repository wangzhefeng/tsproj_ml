# -*- coding: utf-8 -*-
"""验证 add_decomposition 配置矩阵（一次性脚本）。"""
import sys
from pathlib import Path
import collections

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config_loader import load_yaml_config

SCENES = ["aidc_ess_selfuse_load", "aidc_load_15min_daily", "aidc_load_15min_rolling",
          "aidc_load_15min_short", "aidc_load_month", "aidc_power_month"]
NPD = {"5min": 288, "15min": 96, "1h": 24, "1D": 1}

created = []
for scene in SCENES:
    for p in sorted(Path("config", scene).rglob("*.yaml")):
        s = str(p)
        if "add_decomposition" in s or "/decomposition/" in s:
            created.append(p)

print(f"add_decomposition total: {len(created)}")
matrix = collections.Counter()
errors = []
for p in created:
    cfg = load_yaml_config(p)
    m = cfg.decomposition_method
    periods = tuple(cfg.decomposition_periods)
    freq = cfg.freq
    npd = NPD.get(freq, 1)
    wl = cfg.window_length
    twl = getattr(cfg, "train_window_length", None)
    hm = getattr(cfg, "horizon_mode", "fixed_steps")
    train_rows = twl if hm == "calendar_month" else (wl * npd if wl else None)
    if m == "mstl" and len(periods) < 2:
        errors.append((p, "mstl<2 periods"))
    for per in periods:
        if train_rows and 2 * per > train_rows:
            errors.append((p, f"{m} period={per} needs {2*per} rows, have {train_rows}"))
    matrix[(p.parts[1], m, periods)] += 1

print(f"errors={len(errors)}")
for p, e in errors[:5]:
    print(f"  {p}: {e}")
print("== matrix ==")
for (scene, m, per), cnt in sorted(matrix.items()):
    print(f"  {scene}: {m} {list(per)} x {cnt}")
