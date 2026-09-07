"""只读天气取数包络规划；目标几何不等于逐请求天气/可得性验收。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.config_loader import load_yaml_config
from data_loading import SourceRegistry
from forecasting_core.specs import CalendarMonthBacktestSpec, FixedStepBacktestSpec, ForecastConfigSpec
from model_ensemble.loader import resolve_members, validate_member_sources
from model_pipeline.supervised_design import minimum_history_rows
from model_testing.geometry import calendar_month_folds


def plan_single_model(config, root, *, origin=None, coverage_cache=None):
    """复核真实目标时间轴，给出runtime候选监督原点与输出时间包络。

    不materialize旧天气，不模拟模型预测；未枚举target-history辅助调用或
    OOF逐折调用，不能作为exact runtime request trace或发布时间证据。
    """
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError('single-model config required')
    key = json.dumps(config.data.canonical_payload(), sort_keys=True)
    cache = coverage_cache if coverage_cache is not None else {}
    if key not in cache:
        cache[key] = SourceRegistry(config.data, root).target_history_coverage()
    coverage = cache[key]
    times = coverage[0].times
    for source in coverage:
        if not source.times.equals(times) or any(not index.equals(times) for index in source.times_by_series.values()):
            raise ValueError('target series must have identical complete timelines for this planner')
    raw_origin = origin if origin is not None else config.validation.get('forecast_origin')
    origin = pd.Timestamp(raw_origin) if raw_origin is not None else times[-1]
    times = times[times <= origin]
    if times.empty or origin not in times:
        raise ValueError('forecast origin missing from target timeline')
    position = int(times.get_loc(origin))
    offset = pd.tseries.frequencies.to_offset(config.problem.freq)

    def candidates(current):
        # 与supervised_design._supervised_arrays的切片合同一致；不导入私有函数。
        start = minimum_history_rows(current) - 1
        stop = position - current.problem.horizon + 1
        if stop <= start:
            raise ValueError('insufficient actual history for supervised origins')
        values = times[start:stop]
        backtest = current.validation.backtest
        if isinstance(backtest, FixedStepBacktestSpec):
            values = values[-backtest.history_steps:]
        if len(values) < 2:
            raise ValueError('at least two actual supervised origins required')
        required_start = values[0] - (minimum_history_rows(current) - 1) * offset
        relevant_times = times[times >= required_start]
        if not relevant_times.equals(pd.date_range(required_start, origin, freq=config.problem.freq)):
            raise ValueError('requested target timeline must be regular at problem frequency')
        return values

    main_origins = candidates(config)
    variants = {config.problem.horizon: (config, main_origins)}
    backtest = config.validation.backtest
    if isinstance(backtest, CalendarMonthBacktestSpec):
        folds = calendar_month_folds(times, train_window_days=backtest.train_window_days, fold_count=backtest.fold_count, stride_months=backtest.stride_months)
        if not folds:
            raise ValueError('no complete calendar-month folds')
        for fold in folds:
            validation = {k: v for k, v in dict(config.validation).items() if k not in {'train_window_days', 'stride_months'}}
            validation.update(horizon_mode='fixed_steps', history_steps=len(main_origins), train_window_steps=min(backtest.train_window_days, len(main_origins) - 1), fold_count=1, stride_steps=fold.horizon, seasonal_naive_lag=max(fold.horizon, 1))
            current = replace(config, problem=replace(config.problem, horizon=fold.horizon), validation=validation)
            values = candidates(current)
            if fold.origin not in values:
                raise ValueError('calendar holdout absent from dynamic supervised timeline')
            # 相同horizon的main可能比动态runner有更多原点；保留二者并集。
            if fold.horizon in variants:
                values = values.union(variants[fold.horizon][1]).sort_values()
            variants[fold.horizon] = (current, values)
    groups = []
    for horizon, (_, values) in sorted(variants.items()):
        groups.append({'freq': config.problem.freq, 'horizon': horizon, 'supervised_origins': [t.isoformat() for t in values], 'final_origin': origin.isoformat(), 'label_start': (values[0] + offset).isoformat(), 'label_end': (origin + horizon * offset).isoformat()})
    return {'groups': groups, 'label_start': min(g['label_start'] for g in groups), 'label_end': max(g['label_end'] for g in groups), 'runtime_requests_verified': False, 'weather_values_verified': False, 'scope': 'supervised_and_final_output_envelope_not_exact_request_trace'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = json.loads(args.inventory.read_text())
    entries, errors, plans, cache, coverage = [], [], {}, {}, {}
    for row in inventory['configs']:
        path = root / row['config']
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['config_sha256']:
                raise ValueError('stale inventory: config bytes changed')
            config = load_yaml_config(path)
            if isinstance(config, ForecastConfigSpec):
                members = [(row['config'], config)]
            else:
                resolved = resolve_members(config, base_dir=root)
                validate_member_sources(config, resolved)
                members = [(ref.config_ref, load_yaml_config(root / ref.config_ref)) for ref in config.members]
            references = []
            for name, member in members:
                payload = member.canonical_payload()
                key = json.dumps({k: payload.get(k) for k in ('problem', 'data', 'features', 'strategy', 'validation')}, sort_keys=True) + str(config.validation.get('forecast_origin'))
                if key not in cache:
                    planned = plan_single_model(member, root, origin=config.validation.get('forecast_origin'), coverage_cache=coverage)
                    digest = hashlib.sha256(json.dumps(planned, sort_keys=True).encode()).hexdigest()
                    plans[digest] = planned
                    cache[key] = digest
                references.append({'member_config': name, 'plan': cache[key]})
            entries.append({'config': row['config'], 'references': references})
        except (ValueError, TypeError, OSError, KeyError) as exc:
            errors.append({'config': row['config'], 'error': str(exc)})
    report = {'scope': 'output_envelopes_not_full_asof_window_audit', 'expected_config_count': len(inventory['configs']), 'planned_config_count': len(entries), 'unique_plan_count': len(plans), 'errors': errors, 'production_eligible': False, 'configs': entries, 'plans': plans}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {'configs', 'plans'}}, ensure_ascii=False))
    return 0 if entries and len(entries) == len(inventory['configs']) and not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
