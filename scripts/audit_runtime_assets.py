#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读审计现役模型配置引用的运行数据资产与声明列。

默认从项目根解析相对 source path。缺失任一活动资产或非 ignored 声明列时
退出 1；物理额外列按 DataSourceSpec 投影视图合同忽略。`--json` 输出稳定
JSON，供 CI/单元测试消费。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_loader import is_model_yaml, load_yaml_config  # noqa: E402
from forecasting_core.specs import ForecastConfigSpec  # noqa: E402
from model_ensemble.specs import EnsembleConfigSpec  # noqa: E402


_PATH_FIELDS = ("history_path", "backtest_path", "future_path")


def _required_columns(source: Mapping[str, Any]) -> set[str]:
    """返回 DataSourceSpec 投影视图要求存在的物理列。"""
    required = {
        str(column.get("name"))
        for column in source.get("columns", ())
        if isinstance(column, Mapping)
        and column.get("name")
        and column.get("role") != "ignored"
    }
    time_col = source.get("time_col")
    if time_col:
        required.add(str(time_col))
    required.update(str(column) for column in source.get("series_id_cols", ()))
    return required


def _missing_required_columns(
    source: Mapping[str, Any],
    actual_columns: set[str],
) -> list[str]:
    """返回投影视图中缺失的非 ignored 列；物理额外列不属于错误。"""
    return sorted(_required_columns(source) - actual_columns)


def _config_sources(config: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(config, (ForecastConfigSpec, EnsembleConfigSpec)):
        raise TypeError(f"unsupported model config type: {type(config).__name__}")
    payload_sources = config.data.canonical_payload()["sources"]
    if not isinstance(payload_sources, list):
        raise TypeError("canonical data.sources must be a list")
    return tuple(
        source for source in payload_sources if isinstance(source, Mapping)
    )


def audit_runtime_assets(
    config_root: str | Path,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    config_dir = Path(config_root)
    repo = Path(repository_root).resolve()
    model_paths = sorted(
        path for path in config_dir.rglob("*.yaml") if is_model_yaml(path)
    )
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    affected_configs: set[str] = set()

    for config_path in model_paths:
        config = load_yaml_config(config_path)
        relative_config = config_path.resolve().relative_to(repo).as_posix()
        for source in _config_sources(config):
            source_name = str(source.get("name", ""))
            for path_role in _PATH_FIELDS:
                raw_path = source.get(path_role)
                if not raw_path:
                    continue
                normalized = Path(str(raw_path)).as_posix()
                references[normalized].append(
                    {
                        "config": relative_config,
                        "source": source_name,
                        "path_role": path_role,
                        "required_columns": sorted(_required_columns(source)),
                    }
                )

    missing_paths = []
    missing_reference_count = 0
    missing_declared_columns = []
    missing_declared_column_reference_count = 0
    column_affected_configs: set[str] = set()
    for raw_path in sorted(references):
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = repo / resolved
        if resolved.exists():
            with resolved.open(newline="", encoding="utf-8-sig") as stream:
                try:
                    actual_columns = set(next(csv.reader(stream)))
                except StopIteration as exc:
                    raise ValueError(f"runtime source is empty: {resolved}") from exc
            missing_sites = []
            for site in references[raw_path]:
                missing = sorted(set(site["required_columns"]) - actual_columns)
                if not missing:
                    continue
                missing_declared_column_reference_count += 1
                column_affected_configs.add(site["config"])
                missing_sites.append(
                    {
                        "config": site["config"],
                        "source": site["source"],
                        "path_role": site["path_role"],
                        "missing_columns": missing,
                    }
                )
            if missing_sites:
                missing_declared_columns.append(
                    {
                        "path": raw_path,
                        "reference_count": len(missing_sites),
                        "references": sorted(
                            missing_sites,
                            key=lambda item: (
                                item["config"],
                                item["source"],
                                item["path_role"],
                            ),
                        ),
                    }
                )
            continue
        sites = sorted(
            references[raw_path],
            key=lambda item: (item["config"], item["source"], item["path_role"]),
        )
        missing_reference_count += len(sites)
        affected_configs.update(site["config"] for site in sites)
        missing_paths.append(
            {
                "path": raw_path,
                "reference_count": len(sites),
                "references": sites,
            }
        )

    return {
        "model_config_count": len(model_paths),
        "unique_source_path_count": len(references),
        "missing_unique_path_count": len(missing_paths),
        "missing_reference_count": missing_reference_count,
        "affected_config_count": len(affected_configs),
        "missing_paths": missing_paths,
        "missing_declared_column_reference_count": (
            missing_declared_column_reference_count
        ),
        "column_affected_config_count": len(column_affected_configs),
        "missing_declared_columns": missing_declared_columns,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit runtime source assets referenced by active model YAMLs."
    )
    parser.add_argument("--root", default=str(ROOT / "config"))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = audit_runtime_assets(args.root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "model_configs={model_config_count} unique_paths={unique_source_path_count} "
            "missing_unique={missing_unique_path_count} missing_refs={missing_reference_count} "
            "affected_configs={affected_config_count} "
            "missing_column_refs={missing_declared_column_reference_count} "
            "column_affected_configs={column_affected_config_count}".format(**payload)
        )
        for item in payload["missing_paths"]:
            print(f"MISSING {item['path']} refs={item['reference_count']}")
        for item in payload["missing_declared_columns"]:
            print(f"MISSING_COLUMNS {item['path']} refs={item['reference_count']}")
    return 1 if (
        payload["missing_unique_path_count"]
        or payload["missing_declared_column_reference_count"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
