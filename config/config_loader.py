# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-06-13
# * Version     : 1.0.061316
# * Description : 配置加载器：canonical schema 严格解析
# ***************************************************


from pathlib import Path
from typing import Any, Mapping

from forecasting_core.specs.config import parse_model_config
from model_ensemble.loader import parse_ensemble_document

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    yaml = None

MODEL_CONFIG_FIELDS = frozenset(
    {
        "problem",
        "data",
        "features",
        "strategy",
        "estimator",
        "probabilistic",
        "validation",
        "output",
    }
)
MODEL_GROUP_FIELDS = frozenset({"problem", "data", "features", "strategy", "estimator"})
ENSEMBLE_GROUP_FIELDS = frozenset({"problem", "data", "ensemble", "output"})

def _strict_yaml_load(text: str, source: str | Path) -> Any:
    if yaml is None:
        raise ImportError("PyYAML is required for YAML configs. Install dependency: pyyaml")

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ValueError(f"Unhashable YAML mapping key in {source}: {key!r}") from exc
            if duplicate:
                raise ValueError(f"Duplicate YAML key {key!r} in {source}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return yaml.load(text, Loader=UniqueKeyLoader)

def is_model_yaml(path: str | Path) -> bool:
    """按顶层 schema 识别模型配置，避免把独立数据工具 YAML 当默认模型。"""
    config_path = Path(path)
    payload = _strict_yaml_load(
        config_path.read_text(encoding="utf-8"),
        config_path,
    )
    if not isinstance(payload, dict):
        return False
    # 代码即版本：按顶层字段形状识别模型配置（单模型或引用式 Ensemble）。
    # 仍声明 schema_version 等未知字段的模型形状文件会被路由到 load_yaml_config，
    # 由严格 parser 按未知字段 RAISE，而不是在审计中被静默跳过。
    return bool(
        MODEL_GROUP_FIELDS.issubset(payload)
        or ENSEMBLE_GROUP_FIELDS.issubset(payload)
    )

def _load_yaml_file(config_yaml: str | Path) -> Mapping[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for YAML configs. Install dependency: pyyaml")

    config_path = Path(config_yaml)
    if not config_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {config_path}")

    loaded = _strict_yaml_load(
        config_path.read_text(encoding="utf-8"),
        config_path,
    ) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return loaded

def load_yaml_config(config_yaml: str | Path):
    loaded = _load_yaml_file(config_yaml)
    # v4 §5.2: route by mutually exclusive field sets. An `ensemble` mapping
    # marks a reference-based ensemble config; anything else must be a
    # single-model base config. 严格 parser 对未知字段（含历史 schema_version）一律 RAISE。
    if isinstance(loaded.get("ensemble"), Mapping):
        return parse_ensemble_document(loaded, source_path=config_yaml)
    return parse_model_config(loaded, source=config_yaml)
