"""生成红太阳五个任务的非递归四方法 canonical YAML。"""
from pathlib import Path
import argparse
import json
import math
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forecasting_core.specs.config import parse_model_config
from model_pipeline.supervised_design import minimum_history_rows
from data_loading.calendar_generator.named_holidays import NAMED_HOLIDAY_FEATURES

TASKS = (("xinnengyuan", "demand_load", False),
         ("xinnengyuan", "demand_load", True),
         ("guangdianchang", "demand_load", False),
         ("guangdianchang", "demand_load", True),
         ("guangdianchang", "pv_load", False))
STRATEGIES = ("direct", "direct-pointwise", "direct-pointwise-horizon", "mimo")


def model_document(site: str, target: str, daily: bool, strategy: str) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown scenario method: {strategy}")
    freq = "1D" if daily else "15min"
    scenario = f"hongtaiyang_cesuan/{site}/{target}/freq_{'1day' if daily else '15min'}"
    data_path = f"dataset/hongtaiyang_cesuan/{site}/" + ("freq_1day/" if daily else "") + f"{target}.csv"
    transforms = {"advanced": {
        "rolling": {"columns": ["value"], "windows": [3, 7] if daily else [96, 288, 672],
                    "stats": ["mean", "std", "min", "max"]},
        "expanding": {"columns": ["value"], "stats": ["mean", "std"]}}}
    if strategy == "direct":
        transforms["direct"] = {"layout": "independent_models", "align_to_target": False}
    elif strategy.startswith("direct-pointwise"):
        transforms["direct"] = {
            "layout": "single_model_horizon", "align_to_target": False,
            "horizon_feature": {"enabled": strategy.endswith("-horizon"),
                                "name": "forecast_horizon_idx", "cyclical": False}}
    document = {
        "schema_version": 2,
        "problem": {"time_col": "time", "freq": freq, "horizon": 31 if daily else 96,
                    "targets": ["value"], "training_scope": "local", "series_id_cols": []},
        "data": {"sources": [
            {"name": "target_history", "source_type": "file",
             "columns": [{"name": "value", "role": "target", "categorical": False}],
             "history_path": data_path, "time_col": "time", "series_id_cols": [], "availability": "source_time"},
            {"name": "chinese_holiday", "source_type": "generated", "generator": "chinese_holiday",
             "columns": [{"name": name, "role": "known_future", "categorical": False}
                         for name in ("is_holiday", "next_holiday_days")],
             "time_col": "time", "series_id_cols": [], "availability": "generator_defined"}]},
        "features": {"target_lags": {"value": [1, 2, 7] if daily else [1, 4, 96, 192, 672]},
                     "observed_past_lags": {},
                     "datetime_features": ([] if daily else ["minute", "hour"]) +
                         ["day", "day_of_week", "month", "days_in_month", "day_of_year"],
                     "transformations": transforms},
        "strategy": {"name": "direct" if strategy.startswith("direct") else strategy},
        "estimator": {"model_type": "lightgbm", "target_adapter": "independent",
                      "params": {"n_estimators": 100, "num_leaves": 7 if daily else 31,
                                 "min_child_samples": 2 if daily else 20,
                                 "n_jobs": 1, "random_state": 2025}},
        "probabilistic": {"mode": "point"},
        "validation": ({"forecast_origin": "2025-12-31T00:00:00", "horizon_mode": "calendar_month",
                        "train_window_days": 59, "fold_count": 1, "stride_months": 1} if daily else
                       {"forecast_origin": "2025-12-31T23:45:00", "horizon_mode": "fixed_steps",
                        "schedule_mode": "daily", "history_steps": 35040,
                        "train_history_steps": 2880, "train_window_steps": 2,
                        "fold_count": 334, "stride_steps": 96}),
        "output": {"scenario_subpath": scenario, "results_root": "results"}}
    if not daily:
        config = parse_model_config(document, source="generated")
        document["validation"]["train_window_steps"] = 2880 - minimum_history_rows(config) - 96 + 1
    return document


def apply_options(document: dict, options: dict) -> dict:
    if not isinstance(options, dict) or set(options) - {'objective', 'calendar', 'halflife_days'}:
        raise ValueError('unknown optimization options')
    if 'calendar' in options and not isinstance(options['calendar'], bool):
        raise ValueError('calendar option must be boolean')
    half = options.get('halflife_days')
    if half is not None and (isinstance(half, bool) or not isinstance(half, (int, float)) or not math.isfinite(half) or half <= 0):
        raise ValueError('halflife_days must be finite and positive')
    if 'objective' in options:
        if options['objective'] not in ('regression_l1', 'regression'):
            raise ValueError('objective must be regression_l1 or regression')
        document['estimator']['params']['objective'] = options['objective']
    if options.get('calendar', False):
        names = (*NAMED_HOLIDAY_FEATURES, 'is_adjusted_workday')
        source = document['data']['sources'][1]
        source['columns'].extend({'name': name, 'role': 'known_future', 'categorical': False} for name in names)
    if options.get('halflife_days') is not None:
        document['validation']['training'] = {'sample_weight': {
            'method': 'exponential', 'halflife_days': options['halflife_days']}}
    return document


def generate(freq: str = '1D', profile: dict | None = None) -> list[str]:
    if profile is not None:
        if set(profile) != {'guangdianchang', 'xinnengyuan'} or any(set(methods) != set(STRATEGIES) for methods in profile.values()):
            raise ValueError('profile must explicitly cover two sites and all four methods')
    documents = []
    for site, target, daily in TASKS:
        if ('1D' if daily else '15min') != freq:
            continue
        for strategy in STRATEGIES:
            document = model_document(site, target, daily, strategy)
            if profile and target == 'demand_load':
                apply_options(document, profile[site][strategy])
            path = ROOT / "config" / document["output"]["scenario_subpath"] / f"lgbm_{strategy}.yaml"
            parse_model_config(document, source=path)
            if not (ROOT / document['data']['sources'][0]['history_path']).is_file():
                raise ValueError('target asset missing before configuration generation')
            documents.append((path, document))
    paths = []
    # 整批通过严格校验后再写，避免坏profile造成只更新一半配置。
    for path, document in documents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 年度入口：config/hongtaiyang_cesuan/annual_backtest.py；扩展窗口及2月例外见场景 README。\n" +
                        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(str(path.relative_to(ROOT)))
    expected = sum(('1D' if daily else '15min') == freq for _, _, daily in TASKS) * len(STRATEGIES)
    if not expected or len(set(paths)) != expected:
        raise ValueError("incomplete configuration matrix")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freq', choices=('1D', '15min'), default='1D')
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text()) if args.profile else None
    print("\n".join(generate(args.freq, profile)))
