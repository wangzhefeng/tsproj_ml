"""HVAC 数据阶段的显式版本路径；raw_data 始终共享、只读。"""
from pathlib import Path

ROUTE_DATA_VERSIONS = ('data_v1', 'data_v2')
DATA_VERSIONS = (*ROUTE_DATA_VERSIONS, 'data_v3')
ARTIFACTS = ('outlier_remove_data', 'imputed_data', 'forecast_data', 'analysis', 'weather_data')


def artifact_path(root, category, data_version=None):
    """None 表示独立准备子根；场景 CLI 必须显式选择数据版本。"""
    if category not in ARTIFACTS or data_version not in (*DATA_VERSIONS, None):
        raise ValueError(f'未知阶段或数据版本: {category}/{data_version}')
    path = Path(root) / category
    return path / data_version if data_version else path


def artifact_relative(category, data_version=None):
    return str(artifact_path(Path('.'), category, data_version))
