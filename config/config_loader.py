# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-06-13
# * Version     : 1.0.061316
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import datetime
import importlib
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    yaml = None


def load_model_config(
    config_module: str = "config.univariate_config",
    config_class: str = "ModelConfig",
    instantiate: bool = False,
):
    module = importlib.import_module(config_module)
    model_config = getattr(module, config_class, None)
    if model_config is None:
        raise ImportError(f"Config class '{config_class}' not found in module: {config_module}")
    if instantiate:
        return model_config()
    return model_config


def _load_yaml_file(config_yaml: str | Path) -> Mapping[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for YAML configs. Install dependency: pyyaml")

    config_path = Path(config_yaml)
    if not config_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {config_path}")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return loaded


def _coerce_override_value(field_name: str, current_value: Any, new_value: Any) -> Any:
    if new_value is None:
        return None
    if isinstance(current_value, datetime.datetime) and isinstance(new_value, str):
        return datetime.datetime.fromisoformat(new_value)
    if field_name in {"now_time", "start_time", "future_time"} and isinstance(new_value, str):
        return datetime.datetime.fromisoformat(new_value)
    return new_value


def apply_config_overrides(cfg: Any, overrides: Mapping[str, Any], source: str = "YAML config") -> Any:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown config override '{key}' in {source}")
        current_value = getattr(cfg, key)
        setattr(cfg, key, _coerce_override_value(key, current_value, value))
    return cfg


def _flatten_overrides(overrides: Mapping[str, Any], cfg: Any, source: str) -> dict[str, Any]:
    flattened = {}
    for key, value in overrides.items():
        if hasattr(cfg, key):
            flattened[key] = value
            continue
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, Mapping) and not hasattr(cfg, nested_key):
                    raise AttributeError(
                        f"Nested config groups are not supported under '{key}.{nested_key}' in {source}"
                    )
                if nested_key in flattened:
                    raise ValueError(f"Duplicate config override '{nested_key}' in {source}")
                flattened[nested_key] = nested_value
            continue
        flattened[key] = value
    return flattened


def load_yaml_config(
    config_yaml: str | Path,
    config_module: Optional[str] = None,
    config_class: Optional[str] = None,
):
    loaded = _load_yaml_file(config_yaml)
    base_config = loaded.get("base_config") or config_module or "config.univariate_config"
    class_name = config_class or "ModelConfig"
    overrides = loaded.get("overrides", {})

    if not isinstance(base_config, str) or not base_config:
        raise ValueError(f"YAML field 'base_config' must be a non-empty string: {config_yaml}")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError(f"YAML field 'config_class' must be a non-empty string: {config_yaml}")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"YAML field 'overrides' must be a mapping: {config_yaml}")

    cfg = load_model_config(base_config, class_name, instantiate=True)
    flattened_overrides = _flatten_overrides(overrides, cfg, source=str(config_yaml))
    return apply_config_overrides(cfg, flattened_overrides, source=str(config_yaml))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
