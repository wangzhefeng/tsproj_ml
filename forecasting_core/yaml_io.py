"""配置入口共用的严格 YAML 读取原语；不依赖任何配置分派器。"""
from pathlib import Path
from typing import Any

import yaml


def strict_yaml_load(text: str, source: str | Path) -> Any:
    """安全解析，包含嵌套映射与 merge 后的重复键均拒绝。"""
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

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(text, Loader=UniqueKeyLoader)
