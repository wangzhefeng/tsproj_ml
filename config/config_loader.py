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

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    yaml = None

MODEL_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
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
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, int) and not isinstance(schema_version, bool) and schema_version == 2:
        return True
    if "schema_version" in payload and set(payload) & MODEL_GROUP_FIELDS:
        return True
    # A model-schema-shaped file with a missing schema_version is still a model config;
    # route it to load_yaml_config so the missing version is reported instead
    # of silently skipping the file during audits.
    return bool(MODEL_GROUP_FIELDS.issubset(set(payload)))

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
    if "schema_version" not in loaded and MODEL_GROUP_FIELDS.issubset(set(loaded)):
        raise ValueError(f"Missing YAML schema_version for canonical model config: {config_yaml}")
    if "schema_version" not in loaded:
        raise ValueError(
            f"Non-canonical YAML (missing schema_version): {config_yaml}"
        )
    schema_version = loaded["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 2
    ):
        raise ValueError(
            f"Unknown YAML schema_version {schema_version!r}: {config_yaml}"
        )
    # v4 §5.2: route by mutually exclusive field sets. An `ensemble` mapping
    # marks a reference-based ensemble config; anything else must be a
    # single-model base config.
    from model_ensemble.loader import parse_ensemble_document
    from model_forecasting.specs.config import parse_model_config

    if isinstance(loaded.get("ensemble"), Mapping):
        return parse_ensemble_document(loaded, source_path=config_yaml)
    return parse_model_config(loaded, source=config_yaml)
