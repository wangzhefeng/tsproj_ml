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


def build_documents():
    templates = {}
    for variant in VARIANTS:
        path = TEMPLATES / f'lgbm_{variant}.yaml'
        load_yaml_config(path)
        templates[variant] = yaml.safe_load(path.read_text(encoding='utf-8'))
    cache = {}
    documents = {}
    for group in GROUPS:
        for version in VERSIONS:
            for route in ROUTES:
                for building, train_days in TRAIN_DAYS.items():
                    stem = 'data' if building == 'ALL' else building + '_data'
                    if group == 'add_endogenous_it':
                        stem += '_with_it'
                    target_path = DATA / 'forecast_data/data_v1' / version / route / (stem + '.csv')
                    _, times = read_frame(target_path, 'time', cache)
                    fold_count = len(times) // DAY - train_days
                    if fold_count < 1:
                        raise ValueError(f'不足一个完整训练+预测窗口: {target_path}')
                    target = 'hvac_total_load_' + route[-1]
                    for variant, template in templates.items():
                        relative = Path(group) / version / route / building / f'lgbm_{variant}.yaml'
                        payload = rename_target(deepcopy(template), target)
                        payload['data']['sources'][0]['history_path'] = str(target_path)
                        features = payload['features']
                        # 短训练窗不能借窗外数据预热；保留模板的1/2天rolling，lag取1/2/3天。
                        if train_days == 7:
                            features['target_lags'][target] = [DAY, 2 * DAY, 3 * DAY]
                            features['transformations']['advanced']['rolling']['windows'] = [DAY, 2 * DAY]
                        if group == 'add_weather':
                            snippet = ROOT / DATA / 'weather_data/data_v1' / version / route / stem / 'weather.source.yaml'
                            weather = yaml.safe_load(snippet.read_text(encoding='utf-8'))['data']['sources']
                            if len(weather) != 1 or weather[0]['name'] != 'weather':
                                raise ValueError(f'未知天气片段结构: {snippet}')
                            payload['data']['sources'].extend(weather)
                        elif group in ('add_endogenous_it', 'add_endogenous_route'):
                            columns = ['it_total_load'] if group == 'add_endogenous_it' else [
                                'hvac_total_load_' + ('B' if route == 'route_A' else 'A')]
                            if group == 'add_endogenous_route' and building == 'ALL':
                                columns.append('hvac_total_load_AB')
                            payload['data']['sources'].append({
                                'name': 'covariate_history', 'source_type': 'file',
                                'columns': [{'name': col, 'role': 'observed_past', 'categorical': False}
                                            for col in columns],
                                'history_path': str(target_path), 'time_col': 'time',
                                'series_id_cols': [], 'availability': 'source_time', 'provider': 'persistence',
                            })
                            # safe-lag 不请求预测区间真值，provider 不参与这些设计请求。
                            features['observed_past_lags'] = {col: [DAY, 2 * DAY] for col in columns}
                        config = parse_model_config(payload, source=relative)
                        train_window = train_days * DAY - minimum_history_rows(config) - DAY + 1
                        if train_window < 1:
                            raise ValueError(f'无有效训练样本: {relative}')
                        payload['validation'].update({
                            'forecast_origin': times[-1].isoformat(),
                            'train_history_steps': train_days * DAY,
                            'train_window_steps': train_window,
                            'fold_count': fold_count, 'stride_steps': DAY,
                            # 训练origin与首个holdout origin须相距H步，避免标签重叠；不跳过测试日。
                            'history_steps': train_window + DAY + (fold_count - 1) * DAY,
                        })
                        payload['output']['scenario_subpath'] = 'aidc_hvac_load_5min/' + str(relative.parent)
                        config = parse_model_config(payload, source=relative)
                        FeatureCompiler(config)
                        config.strategy.resolve(DAY)
                        for source in config.data.sources:
                            if source.source_type != 'file':
                                continue
                            frame, source_times = read_frame(source.history_path, source.time_col, cache)
                            if not source_times.equals(times):
                                raise ValueError(f'协变量时间轴不匹配: {relative}/{source.name}')
                            # 包括 inference_columns 对应 ignored 物理列，未声明原料列不入模。
                            names = [col.name for col in source.columns]
                            if not np.isfinite(frame[names].to_numpy(dtype=float)).all():
                                raise ValueError(f'模型输入有缺失/非有限值: {relative}/{source.name}')
                        documents[relative] = (
                            '# HVAC历史回测配置；严格训练窗口，仅支持 --backtest-only。\n'
                            '# 离线补值和天气可得性假设不构成实盘证据；未执行模型训练。\n'
                            + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
                        )
    return documents


def publish(documents, output_dir, *, check=False, config_version='v1'):
    """先核对全部文件；不同内容拒绝覆盖，绝不清除清单外文件。"""
    if config_version not in ('v1', 'v3'):
        raise ValueError(f'未知配置版本: {config_version}')
    existing = {p.relative_to(output_dir) for group in GROUPS
                for p in (output_dir / group).rglob('*.yaml')}
    # 两个生成器各自管理明确命名空间，不互相覆盖，也不忽略本空间未知文件。
    existing = {p for p in existing if (p.parts[1:3] == ('A1_all', 'v3')) == (config_version == 'v3')}
    if any((p.parts[1:3] == ('A1_all', 'v3')) != (config_version == 'v3')
           or p.is_absolute() or '..' in p.parts or p.parts[0] not in GROUPS for p in documents):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='只核验资产及物理配置与生成规则一致')
    args = parser.parse_args()
    print(json.dumps(publish(build_documents(), DIRECTORY, check=args.check), ensure_ascii=False))


if __name__ == '__main__':
    main()
