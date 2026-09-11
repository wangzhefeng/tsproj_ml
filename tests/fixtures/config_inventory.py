"""独立读取物理 YAML 的测试清单，不委托被测 loader/catalog 的发现规则。"""
from pathlib import Path

import yaml


def model_config_inventory(config_root: Path) -> dict[str, str]:
    """返回相对路径到模型类型的映射；非模型工具配置不进入清单。"""
    inventory = {}
    for path in sorted(config_root.rglob('*.yaml')):
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            continue
        if 'estimator' in payload and 'ensemble' in payload:
            raise AssertionError(f'ambiguous model configuration: {path}')
        if 'ensemble' in payload:
            kind = 'ensemble'
        elif 'estimator' in payload:
            kind = 'single_model'
        elif payload.get('schema_version') == 2:
            raise AssertionError(f'canonical model lacks estimator/ensemble: {path}')
        else:
            continue
        inventory[path.relative_to(config_root).as_posix()] = kind
    return inventory
