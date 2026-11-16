# -*- coding: utf-8 -*-
"""构建 HVAC 四组物理 YAML；只读数据，预检全部配置后写入，不训练模型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from config.config_loader import load_yaml_config
from scripts.weather_data.build_scenario_weather import weather_source
from feature_engineering.compiler import FeatureCompiler
from forecasting_core.specs.config import parse_model_config
from model_pipeline.supervised_design import minimum_history_rows

DIRECTORY = ROOT / 'config/aidc_hvac_load_5min'
TEMPLATES = ROOT / 'config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/baseline'
DATA = Path('dataset/aidc_hvac_load_5min')
GROUPS = ('baseline', 'add_weather', 'add_endogenous_it', 'add_endogenous_route')
VERSIONS = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')
TRAIN_DAYS = {'A1': 14, 'A2': 14, 'A3': 7, 'ALL': 7}
VARIANTS = ('direct', 'direct-pointwise', 'direct-pointwise-horizon', 'recursive',
            'mimo', 'recmo', 'dirrec', 'dirmo', 'dirrecmo')
DAY = 288


def read_frame(path, time_col, cache):
    """缓存只在本次生成内共享；每份输入必须是完整自然日5min网格。"""
    key = (str(path), time_col)
    if key not in cache:
        frame = pd.read_csv(ROOT / path, float_precision='round_trip')
        times = pd.DatetimeIndex(pd.to_datetime(frame[time_col], errors='raise'))
        if (times.empty or times[0] != times[0].normalize()
                or len(times) % DAY
                or not times.equals(pd.date_range(times[0], periods=len(times), freq='5min'))):
            raise ValueError(f'非完整自然日5min网格: {path}')
        cache[key] = (frame, times)
    return cache[key]


def rename_target(value, target):
    """精确替换模板目标标识，不做字符串子串替换。"""
    if isinstance(value, dict):
        return {(target if key == 'value' else key): rename_target(item, target)
                for key, item in value.items()}
    if isinstance(value, list):
        return [rename_target(item, target) for item in value]
    return target if value == 'value' else value


def publish(documents, output_dir, *, check=False, config_version='a1'):
    """先核对全部文件；不同内容拒绝覆盖，绝不清除清单外文件。"""
    if config_version != 'a1':
        raise ValueError(f'未知配置版本: {config_version}')
    groups = (*GROUPS, 'add_datetime_holiday', 'add_context') if config_version == 'a1' else GROUPS
    existing = {p.relative_to(output_dir) for group in groups
                for p in (output_dir / group).rglob('*.yaml')}
    # 两个生成器各自管理明确命名空间，不互相覆盖，也不忽略本空间未知文件。
    existing = {p for p in existing if (p.parts[1:2] == ('A1_all',)) == (config_version == 'a1')}
    if any((p.parts[1:2] == ('A1_all',)) != (config_version == 'a1')
           or p.is_absolute() or '..' in p.parts or p.parts[0] not in groups
           or (config_version == 'a1' and (len(p.parts) != 4 or p.parts[2] not in VERSIONS))
           for p in documents):
        raise ValueError('配置路径越过当前生成器命名空间')
    unexpected = existing - documents.keys()
    if unexpected:
        raise ValueError(f'发现清单外配置，拒绝处理: {sorted(map(str, unexpected))}')
    missing = []
    for relative, text in documents.items():
        path = output_dir / relative
        if path.exists():
            if path.read_text(encoding='utf-8') != text:
                raise FileExistsError(f'拒绝覆盖已有不同配置: {path}')
        else:
            missing.append(relative)
    if check and missing:
        raise FileNotFoundError(f'缺少 {len(missing)} 份配置，例如 {missing[0]}')
    if not check:
        for relative in missing:
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                stream.write(documents[relative])
    return {'configs': len(documents), 'created': 0 if check else len(missing),
            'mode': 'check' if check else 'create', 'output_dir': str(output_dir)}
