"""unittest 分组入口；保留原生发现全集，不以 skip 隐藏慢测试。"""

from __future__ import annotations

import argparse

import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

# 仅将已确认的轻量合同/算法测试纳入快测。新文件默认 integration，绝不漏收。
FAST_MODULES = frozenset("""
test_suite_runner test_forecast_tensor test_forecast_problem_spec
 test_forecast_data_spec test_forecast_config_fingerprint test_canonical_strategy_spec
 test_calendar_month_horizon test_decomposition_spec test_crossing_method_config
 test_forecast_model_bundle test_forecast_result_schema test_probabilistic_types
 test_probabilistic_contracts test_probabilistic_metrics
 test_conformal test_ensemble_specs test_ensemble_methods test_ensemble_quantile_blending
 test_standard_strategy_executors test_oof_gap_contract test_oof_gap_identity
 test_package_layering test_dependency_contract test_model_parameter_validation
""".split())

# 全仓配置/真实资产审计和场景数据链：有意单独执行，而不是禁用。
AUDIT_MODULES = frozenset("""
test_active_config_runtime_contract test_aidc_load_15min_design_audit
 test_audit_forecast_configs test_audit_ensemble_configs test_runtime_asset_audit
 test_validation_geometry_manifest test_batch_eligibility_audit
 test_aidc_date_window_configs test_ess_quantile_config_matrix
 test_generate_load_15min_matrix test_public_config_field_removal
 test_adjust_leadership_day_summary_metrics test_select_aidc_leadership_days
 test_aidc_point_power_processing test_point_load_aggregate test_fuse_aidc_results
 test_computility_process test_computility_weather_derivation
 test_ess_strategy_clustering test_ess_strategy_pipeline test_ess_strategy_profiles
 test_ess_strategy_similar_day test_ess_strategy_states test_ess_strategy_windows
 test_ess_weather_derivation test_power_month_event_features
""".split())
AUDIT_IDS = frozenset({
    "test_config_entrypoints.Task27ExecutionMatrixTest.test_task27_checker_validates_all_model_configs_read_only",
})


def classify(test_id):
    module = test_id.split(".")[0]
    if module in AUDIT_MODULES or test_id in AUDIT_IDS:
        return "audit"
    if module in FAST_MODULES:
        return "fast"
    return "integration"


def partition(test_ids):
    groups = {name: [] for name in ("fast", "integration", "audit")}
    seen = set()
    for test_id in test_ids:
        if test_id in seen:
            raise ValueError(f"duplicate test ID: {test_id}")
        seen.add(test_id)
        groups[classify(test_id)].append(test_id)
    return groups


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def discover():
    # 与 `unittest discover -s tests` 相同的短模块名，兼容已有 fixture 导入。
    sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    tests = list(flatten(loader.discover(str(ROOT / "tests"), pattern="test_*.py")))
    if loader.errors:
        raise RuntimeError("\n".join(map(str, loader.errors)))
    return tests


class TimedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = []

    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        self.timings.append({"id": test.id(), "seconds": time.perf_counter() - self.started})
        super().stopTest(test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("fast", "integration", "audit", "all"))
    parser.add_argument("--match", default="", help="测试 ID 子串过滤；无匹配报错")
    parser.add_argument("--list", action="store_true", help="只发现并列出，不执行测试")
    parser.add_argument("--report", type=Path, help="保存实际执行耗时及失败 ID（JSON）")
    args = parser.parse_args(argv)
    try:
        tests = discover()
        groups = partition(test.id() for test in tests)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    selected = [
        test for test in tests
        if (args.suite == "all" or classify(test.id()) == args.suite)
        and args.match in test.id()
    ]
    if not selected:
        print("No tests selected", file=sys.stderr)
        return 2
    print(f"discovered={len(tests)} groups={dict((key, len(value)) for key, value in groups.items())} selected={len(selected)}", flush=True)
    if args.list:
        for test in selected:
            print(test.id())
        return 0
    result = unittest.TextTestRunner(verbosity=2, resultclass=TimedResult).run(unittest.TestSuite(selected))
    assert isinstance(result, TimedResult)
    if args.report:
        payload = {
            "suite": args.suite,
            "discovered": len(tests),
            "selected": len(selected),
            "run": result.testsRun,
            "failures": [test.id() for test, _ in result.failures],
            "errors": [test.id() for test, _ in result.errors],
            "skipped": [{"id": test.id(), "reason": reason} for test, reason in result.skipped],
            "unexpected_successes": [test.id() for test in result.unexpectedSuccesses],
            "timings": sorted(result.timings, key=lambda row: row["seconds"], reverse=True),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
