"""固定合成负载的编译基准；不训练模型、不写 results、不设 CI 时间阈值。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering import FeatureCompiler
from pipeline.supervised_design import SupervisedDesignBuilder
from test_compiler_batch_design import CompilerBatchDesignTest
from test_compiler_origin_statistics import advanced_statistics


def run_benchmark(*, global_scope: bool, horizon: int, origins: int, repeats: int) -> dict:
    for name, value in (("horizon", horizon), ("origins", origins), ("repeats", repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    # 共用已有 48 点合成 fixture，预热长度由 horizon 与最深 lag 决定。
    start = max(6, horizon + 2)
    if start + origins > 48:
        raise ValueError("workload exceeds the 48-row fixture history")
    fixture = CompilerBatchDesignTest()
    fixture.setUp()
    try:
        config = fixture._global_config("direct", None) if global_scope else fixture._config("direct")
        columns = ["load", "power"] if global_scope else ["load"]
        config = replace(config,
            problem=replace(config.problem, horizon=horizon),
            features=replace(config.features,
                target_lags={name: (horizon, horizon + 1) for name in columns},
                transformations={"advanced": advanced_statistics(columns)},
            ),
        )
        builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, fixture.root))
        requests = tuple(builder.request(cast(pd.Timestamp, pd.Timestamp(value)))
                         for value in fixture.times.to_numpy()[start:start + origins])
        materialize_start = perf_counter()
        infos = tuple(builder.registry.materialize(request) for request in requests)
        materialize_seconds = perf_counter() - materialize_start
        single_compiler = FeatureCompiler(config)
        batch_compiler = FeatureCompiler(config)
        if not batch_compiler.batch_eligibility(requests).eligible:
            raise AssertionError("benchmark requires an eligible batch workload")

        def single():
            return tuple(single_compiler.compile(info, request) for info, request in zip(infos, requests))

        def batch():
            return batch_compiler.compile_batch(infos, requests)

        # 正确性检查兼作预热，不计入计时；数值及证据不等价则不出性能结果。
        expected, actual = single(), batch()
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left.frame, right.frame, check_exact=True)
            if (left.schema != right.schema or left.visibility_proof != right.visibility_proof
                    or left.source_lineage != right.source_lineage):
                raise AssertionError("compiler evidence differs")
        measurements: dict[str, list[float]] = {"single": [], "batch": []}
        for repeat in range(repeats):
            methods = (("single", single), ("batch", batch))
            # 交替顺序，减少总是先测一种路径的偏差。
            for name, operation in methods if repeat % 2 == 0 else reversed(methods):
                started = perf_counter()
                result = operation()
                measurements[name].append(perf_counter() - started)
                del result
        single_median, batch_median = median(measurements["single"]), median(measurements["batch"])
        return {
            "scope": "global" if global_scope else "local",
            "origin_count": origins, "horizon": horizon,
            "series_count": 2 if global_scope else 1,
            "history_rows_per_series": len(fixture.times),
            "feature_count": len(actual[0].schema.feature_names),
            "equivalent": True, "proof_mode": "materialize",
            "materialize_seconds": materialize_seconds,
            "single_seconds": measurements["single"], "batch_seconds": measurements["batch"],
            "single_median_seconds": single_median, "batch_median_seconds": batch_median,
            "single_over_batch": single_median / batch_median,
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
            "platform": platform.platform(),
            "boundary": "synthetic compiler-only; inputs pre-materialized; no model fitting, disk cache, or RSS measurement",
        }
    finally:
        fixture.tearDown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--origins", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [run_benchmark(global_scope=global_scope, horizon=args.horizon,
                             origins=args.origins, repeats=args.repeats)
               for global_scope in (False, True)]
    payload = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        # 基准证据只新建，避免覆盖此前测量结果。
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
