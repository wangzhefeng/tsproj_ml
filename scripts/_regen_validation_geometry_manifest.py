# -*- coding: utf-8 -*-
"""就地再生成 docs/validation_geometry_migration_manifest.yaml。

规则（与 tests/test_validation_geometry_manifest.py 门禁口径一致）：
- entries 与现役 is_model_yaml 集合精确对齐；已删除路径条目移除；
- 幸存条目保留 old_* 历史字段，new_fingerprint 全量重算；
- 新增条目 old_*=None、formula=None；subday 单模型按真实数据计算
  expected_training_contract（与测试同一公式）；ensemble/日历条目 actual=None。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config_loader import is_model_yaml, load_yaml_config  # noqa: E402
from forecasting_core.specs import FixedStepBacktestSpec, ForecastConfigSpec  # noqa: E402
from model_forecasting.design import minimum_history_rows  # noqa: E402

MANIFEST = ROOT / "docs/validation_geometry_migration_manifest.yaml"
SUBDAY_EXCLUDE = {"1D", "1ME", "1MS"}


def actual_final_training_count(config, cache):
    """与 test_validation_geometry_manifest 同一公式。"""
    target_sources = [
        s for s in config.data.sources if any(c.role.value == "target" for c in s.columns)
    ]
    assert len(target_sources) == 1
    source = target_sources[0]
    history_path = Path(source.history_path)
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    key = (str(history_path), source.time_col)
    if key not in cache:
        values = pd.to_datetime(pd.read_csv(history_path)[source.time_col])
        cache[key] = pd.DatetimeIndex(values).drop_duplicates().sort_values()
    origin = pd.Timestamp(config.validation["forecast_origin"])
    timeline = cache[key]
    timeline = timeline[timeline <= origin]
    pos = int(timeline.get_indexer([origin])[0])
    available = max(0, pos - config.problem.horizon + 1 - (minimum_history_rows(config) - 1))
    geometry = config.validation.backtest
    candidate = min(available, geometry.history_steps)
    return min(candidate, geometry.train_window_steps), available


def main() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    old_entries = {e["path"]: e for e in manifest["entries"]}

    active = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "config").rglob("*.yaml")
        if is_model_yaml(p)
    )
    cache: dict = {}
    entries = []
    counts: dict[str, int] = dict.fromkeys(
        ("single", "ensemble", "fixed_steps", "calendar_month", "subday", "subday_single_models"),
        0,
    )
    for rel in active:
        path = ROOT / rel
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        is_ensemble = "ensemble" in raw
        cfg = load_yaml_config(path)
        kind = "ensemble" if is_ensemble else "single"
        freq = cfg.problem.freq
        horizon_mode = str(raw["validation"].get("horizon_mode", "fixed_steps"))
        validation = raw["validation"]
        if horizon_mode == "calendar_month":
            geometry = {
                "train_window_days": validation["train_window_days"],
                "fold_count": validation["fold_count"],
                "stride_months": validation["stride_months"],
            }
        else:
            geometry = {
                "history_steps": validation["history_steps"],
                "train_window_steps": validation["train_window_steps"],
                "fold_count": validation["fold_count"],
                "stride_steps": validation["stride_steps"],
            }
        counts[kind] += 1
        counts[horizon_mode] += 1
        subday = freq not in SUBDAY_EXCLUDE
        if subday:
            counts["subday"] += 1
            if not is_ensemble:
                counts["subday_single_models"] += 1

        prior = old_entries.get(rel)
        entry = {
            "path": rel,
            "kind": kind,
            "freq": freq,
            "horizon_mode": horizon_mode,
            "old_validation_geometry": prior["old_validation_geometry"] if prior else None,
            "new_validation_geometry": geometry,
            "formula": prior["formula"] if prior else None,
        }
        if prior is not None:
            contract = prior["expected_training_contract"]
        elif is_ensemble:
            contract = {
                "unit": "supervised_origin_steps",
                "configured_final_train_window": validation.get("train_window_steps"),
                "actual_final_training_origin_count": None,
            }
        elif horizon_mode == "calendar_month":
            contract = {
                "unit": "raw_days",
                "configured_final_train_window": validation["train_window_days"],
                "safe_supervised_count": "derived_per_target_month",
                "actual_final_training_origin_count": None,
            }
        elif subday and isinstance(cfg, ForecastConfigSpec):
            actual, available = actual_final_training_count(cfg, cache)
            contract = {
                "unit": "supervised_origin_steps",
                "configured_final_train_window": validation["train_window_steps"],
                "actual_available_supervised_origins": available,
                "actual_final_training_origin_count": actual,
            }
        else:
            contract = {
                "unit": "supervised_origin_steps",
                "configured_final_train_window": validation.get("train_window_steps"),
                "actual_final_training_origin_count": None,
            }
        entry["expected_training_contract"] = contract
        entry["old_fingerprint"] = prior["old_fingerprint"] if prior else None
        entry["new_fingerprint"] = cfg.fingerprint()
        entries.append(entry)

    manifest["counts"] = {"total": len(entries), **counts}
    manifest["entries"] = entries
    MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    kept = sum(1 for e in entries if e["path"] in old_entries)
    print(f"entries={len(entries)} kept={kept} new={len(entries) - kept}")
    print("counts:", manifest["counts"])


if __name__ == "__main__":
    main()
