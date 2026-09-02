# -*- coding: utf-8 -*-
"""维护并校验三个 AIDC 15min 负荷场景的 canonical 配置矩阵。

默认只读校验；传 ``--write-derived`` 时，仅从现役 ``add_exogenous``
单模型生成四个 ``add_ensemble`` 顶层配置。融合直接引用现役 direct 与
recursive 单模型，不创建重复的 ``ensemble_members`` 副本。
基础单模型是人工审定的语义输入，本脚本不会覆盖它们。
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config_loader import load_yaml_config  # noqa: E402
from forecasting_core.specs import ForecastConfigSpec  # noqa: E402
from model_ensemble.loader import (  # noqa: E402
    load_ensemble_config,
    resolve_members,
    validate_member_sources,
)
from model_ensemble.specs import EnsembleConfigSpec  # noqa: E402

SCENARIOS = (
    "aidc_load_15min_daily",
    "aidc_load_15min_rolling",
    "aidc_load_15min_short",
)
ROUTES = ("route_A", "route_B")
ENSEMBLE_METHODS = (
    "averaging",
    "weighted",
    "linear_blending",
    "stacking",
)
GROUP_FILES = {
    "baseline": {
        "enet_direct.yaml",
        "lasso_direct.yaml",
        "lgbm_direct.yaml",
        "lgbm_direct_horizon.yaml",
        "lgbm_direct_pointwise.yaml",
        "lgbm_recmo.yaml",
        "lgbm_recursive.yaml",
        "ridge_direct.yaml",
        "st_recursive.yaml",
    },
    "add_exogenous": {
        "enet_direct.yaml",
        "lasso_direct.yaml",
        "lgbm_direct.yaml",
        "lgbm_direct_horizon.yaml",
        "lgbm_direct_pointwise.yaml",
        "lgbm_recmo.yaml",
        "lgbm_recursive.yaml",
        "ridge_direct.yaml",
    },
    "add_endogenous_cross_route": {
        "lgbm_direct.yaml",
        "lgbm_mimo.yaml",
        "lgbm_recursive.yaml",
    },
    "add_endogenous_state": {
        "enet_direct.yaml",
        "lasso_direct.yaml",
        "lgbm_direct.yaml",
        "lgbm_direct_horizon.yaml",
        "lgbm_direct_pointwise.yaml",
        "lgbm_recmo.yaml",
        "lgbm_recursive.yaml",
        "ridge_direct.yaml",
    },
    "add_decomposition": {
        f"lgbm_{strategy}-decomp-{variant}.yaml"
        for strategy in ("direct", "recmo", "recursive")
        for variant in ("linear", "stl96", "mstl96-672")
    },
    "add_ensemble": {
        f"lgbm_ensemble_{method.replace('_', '-')}.yaml"
        for method in ENSEMBLE_METHODS
    },
}
def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _set_output(payload: dict[str, Any], scenario: str, route: str) -> None:
    output = payload["output"]
    output["scenario_subpath"] = f"{scenario}/{route}/add_ensemble"
    output["setting_suffix"] = ""


def _make_source_defaults_explicit(payload: dict[str, Any]) -> None:
    for source in payload["data"]["sources"]:
        source.setdefault("series_id_cols", [])


def build_derived_configs(scenario: str, route: str) -> dict[Path, dict[str, Any]]:
    """由 add_exogenous 的 direct/recursive 生成融合顶层配置。"""
    route_dir = ROOT / "config" / scenario / route
    direct = _load(route_dir / "add_exogenous/lgbm_direct.yaml")
    derived: dict[Path, dict[str, Any]] = {}

    for method in ENSEMBLE_METHODS:
        ensemble = copy.deepcopy(direct)
        _make_source_defaults_explicit(ensemble)
        for section in ("features", "strategy", "estimator"):
            ensemble.pop(section, None)
        _set_output(ensemble, scenario, route)
        ensemble["ensemble"] = {
            "members": [
                {
                    "name": "direct",
                    "config_ref": "../add_exogenous/lgbm_direct.yaml",
                },
                {
                    "name": "recursive",
                    "config_ref": "../add_exogenous/lgbm_recursive.yaml",
                },
            ],
            "oof": {
                "train_window_steps": 30,
                "fold_count": 1,
                "stride_steps": 1,
            },
            "method": {"name": method, "params": {}},
        }
        filename = f"lgbm_ensemble_{method.replace('_', '-')}.yaml"
        derived[route_dir / "add_ensemble" / filename] = ensemble
    return derived


def write_derived_configs() -> int:
    written = 0
    for scenario in SCENARIOS:
        for route in ROUTES:
            for path, payload in build_derived_configs(scenario, route).items():
                _dump(payload, path)
                written += 1
    return written


def _assert_file_matrix(route_dir: Path) -> None:
    for group, expected in GROUP_FILES.items():
        group_dir = route_dir / group
        actual = {path.name for path in group_dir.glob("*.yaml")}
        if actual != expected:
            raise AssertionError(
                f"{group_dir}: expected={sorted(expected)}, actual={sorted(actual)}"
            )
    for group in GROUP_FILES:
        stale = list((route_dir / group / "ensemble_members").glob("*.yaml"))
        if stale:
            raise AssertionError(f"duplicate ensemble member configs are forbidden: {stale}")


def validate_matrix() -> tuple[int, int]:
    single_count = 0
    ensemble_count = 0
    for scenario in SCENARIOS:
        for route in ROUTES:
            route_dir = ROOT / "config" / scenario / route
            _assert_file_matrix(route_dir)
            paths = [
                path
                for group in GROUP_FILES
                for path in (route_dir / group).glob("*.yaml")
            ]
            if len(paths) != 41:
                raise AssertionError(f"{route_dir}: expected 41 YAML files, got {len(paths)}")

            for path in sorted(paths):
                config = load_yaml_config(path)
                payload = _load(path)
                if config.probabilistic["mode"] != "point":
                    raise AssertionError(f"non-point config: {path}")
                relative = path.relative_to(route_dir)
                group = relative.parts[0]
                expected_subpath = f"{scenario}/{route}/{group}"
                actual_subpath = config.output["scenario_subpath"]
                if actual_subpath != expected_subpath:
                    raise AssertionError(
                        f"{path}: scenario_subpath={actual_subpath!r}, "
                        f"expected={expected_subpath!r}"
                    )
                if isinstance(config, EnsembleConfigSpec):
                    members = resolve_members(config, base_dir=path.parent)
                    validate_member_sources(config, members)
                    ensemble_count += 1
                elif isinstance(config, ForecastConfigSpec):
                    single_count += 1
                else:
                    raise TypeError(f"unexpected config type for {path}: {type(config)}")
                if relative == Path("add_endogenous_cross_route/lgbm_mimo.yaml"):
                    if payload["strategy"]["name"] != "mimo":
                        raise AssertionError(f"canonical filename/strategy mismatch: {path}")
    return single_count, ensemble_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-derived",
        action="store_true",
        help="重写 24 个 add_ensemble 顶层配置（不创建 member 副本）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_derived:
        written = write_derived_configs()
        print(f"written derived configs: {written}")
    single_count, ensemble_count = validate_matrix()
    print(
        "matrix OK: "
        f"scenarios={len(SCENARIOS)} routes={len(SCENARIOS) * len(ROUTES)} "
        f"single={single_count} ensemble={ensemble_count} total={single_count + ensemble_count}"
    )


if __name__ == "__main__":
    main()
