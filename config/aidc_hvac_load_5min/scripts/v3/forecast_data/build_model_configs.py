"""A1_all 六组配置，消费 data_v3；完整预检后独立发布，不训练模型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from config.aidc_hvac_load_5min.scripts.forecast_data.build_model_configs import (
    DATA, DAY, DIRECTORY, TEMPLATES, VARIANTS, VERSIONS, publish, read_frame, rename_target,
)
from feature_engineering.compiler import FeatureCompiler
from forecasting_core.specs.config import parse_model_config
from model_pipeline.supervised_design import minimum_history_rows

TARGET = 'hvac_total_load_AB'
GROUPS = ('baseline', 'add_datetime_holiday', 'add_weather', 'add_endogenous_it',
          'add_endogenous_route', 'add_context')
TARGET_LAG_DAYS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)
ROLLING_DAYS = (1, 2, 4, 7, 14, 28)
COVARIATE_LAG_DAYS = (1, 2, 7, 14, 28)


def build_documents():
    templates = {}
    for variant in VARIANTS:
        path = TEMPLATES / f'lgbm_{variant}.yaml'
        load_yaml_config(path)
        templates[f'lgbm_{variant}.yaml'] = yaml.safe_load(path.read_text(encoding='utf-8'))
    for model in ('ridge', 'xgboost', 'st'):
        for variant in VARIANTS[:4]:
            payload = deepcopy(templates[f'lgbm_{variant}.yaml'])
            payload['estimator']['model_type'] = model
            # ST仅消费目标lag；纯目标基线不按星期拆分权重。
            payload['estimator']['params'] = {'day_type_split': False} if model == 'st' else {}
            templates[f'{model}_{variant}.yaml'] = payload
    ets_path = TEMPLATES / 'ets.yaml'
    load_yaml_config(ets_path)
    templates['ets.yaml'] = yaml.safe_load(ets_path.read_text(encoding='utf-8'))
    grid = pd.date_range('2026-04-08', '2026-09-16 23:55', freq='5min')
    documents, cache = {}, {}
    for group in GROUPS:
        train_days = 150 if group == 'add_context' else 90
        for devices in VERSIONS:
            target_path = DATA / 'forecast_data/data_v3' / devices / 'A1_all/data.csv'
            _, times = read_frame(target_path, 'time', cache)
            if not times.equals(grid):
                raise ValueError(f'v3 目标偏离固定窗口: {target_path}')
            for filename, template in templates.items():
                if group not in ('baseline', 'add_context') and not filename.startswith('lgbm_'):
                    continue
                relative = Path(group) / 'A1_all' / devices / filename
                payload = rename_target(deepcopy(template), TARGET)
                if not isinstance(payload, dict):
                    raise TypeError(f'模型模板必须是映射: {filename}')
                payload['data']['sources'][0]['history_path'] = str(target_path)
                if group != 'add_datetime_holiday':
                    payload['data']['sources'] = payload['data']['sources'][:1]
                    payload['features']['datetime_features'] = []
                if filename != 'ets.yaml':
                    payload['features']['target_lags'][TARGET] = [d * DAY for d in TARGET_LAG_DAYS]
                    payload['features']['transformations']['advanced']['rolling']['windows'] = [
                        d * DAY for d in ROLLING_DAYS]
                if group == 'add_weather':
                    snippet = ROOT / DATA / 'weather_data/data_v3' / devices / 'A1_all/data/weather.source.yaml'
                    sources = yaml.safe_load(snippet.read_text(encoding='utf-8'))['data']['sources']
                    if len(sources) != 1 or sources[0]['name'] != 'weather':
                        raise ValueError(f'非法天气片段: {snippet}')
                    payload['data']['sources'].extend(sources)
                elif group in ('add_endogenous_it', 'add_endogenous_route'):
                    columns = ['it_subset_load'] if group == 'add_endogenous_it' else [
                        'hvac_total_load_A', 'hvac_total_load_B']
                    payload['data']['sources'].append({
                        'name': 'covariate_history', 'source_type': 'file',
                        'columns': [{'name': c, 'role': 'observed_past', 'categorical': False} for c in columns],
                        'history_path': str(target_path), 'time_col': 'time', 'series_id_cols': [],
                        'availability': 'source_time', 'provider': 'persistence',
                    })
                    payload['features']['observed_past_lags'] = {
                        c: [d * DAY for d in COVARIATE_LAG_DAYS] for c in columns}
                config = parse_model_config(payload, source=relative)
                train_window = train_days * DAY - minimum_history_rows(config) - DAY + 1
                if train_window < 1:
                    raise ValueError(f'训练历史不足: {relative}')
                folds = len(times) // DAY - train_days
                if folds < 1:
                    raise ValueError(f'不足一个完整训练+预测窗口: {relative}')
                payload['validation'].update({
                    'forecast_origin': times[-1].isoformat(), 'train_history_steps': train_days * DAY,
                    'train_window_steps': train_window, 'fold_count': folds, 'stride_steps': DAY,
                    'history_steps': train_window + folds * DAY,
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
                        raise ValueError(f'v3 协变量网格不匹配: {relative}/{source.name}')
                    if not np.isfinite(frame[[c.name for c in source.columns]].to_numpy(dtype=float)).all():
                        raise ValueError(f'v3 模型声明列缺失或非有限: {relative}/{source.name}')
                documents[relative] = (
                    '# A1 双路合计 data_v3；IT 为固定204点子集，不是全楼IT总负荷。\n'
                    '# 无异常清洗；离线补值/天气可得性假设不构成实盘证据。\n'
                    f'# {train_days}天训练、1天预测；仅支持 --backtest-only，未执行正式模型。\n'
                    + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
                )
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish(build_documents(), DIRECTORY, check=args.check, config_version='a1'),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
