"""新能源跨年周期：校验源数据、生成日均数据及两份 pointwise 配置。"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from forecasting_core.specs.config import parse_model_config
from model_pipeline.supervised_design import minimum_history_rows
from prepare import validate_frame

PERIOD = {'start': '2025-09-01', 'end': '2026-09-01'}
SITE = 'xinnengyuan_2026'


def prepare():
    source = ROOT / 'dataset/hongtaiyang_cesuan' / SITE / 'demand_load.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = validate_frame(pd.read_csv(source), **PERIOD)
    daily = frame.set_index('time').resample('1D').mean().reset_index()
    validate_frame(daily, '1D', **PERIOD)
    documents = []
    for freq in ('15min', '1day'):
        template = ROOT / f'config/hongtaiyang_cesuan/xinnengyuan/demand_load/freq_{freq}/lgbm_direct-pointwise.yaml'
        doc = yaml.safe_load(template.read_text())
        doc['data']['sources'][0]['history_path'] = doc['data']['sources'][0]['history_path'].replace('/xinnengyuan/', f'/{SITE}/')
        doc['output']['scenario_subpath'] = doc['output']['scenario_subpath'].replace('/xinnengyuan/', f'/{SITE}/')
        validation = doc['validation']
        validation['forecast_origin'] = '2026-08-31T00:00:00' if freq == '1day' else '2026-08-31T23:45:00'
        # YAML仅表达末窗参考几何；逐窗变长历史由年度入口调度。
        if freq == '1day':
            validation['train_window_days'] = 92
        else:
            validation['fold_count'] = 1
            validation['train_history_steps'] = 92 * 96
            config = parse_model_config(doc, source='reference')
            validation['train_window_steps'] = 92 * 96 - minimum_history_rows(config) - 96 + 1
        parse_model_config(doc, source='generated')
        documents.append((ROOT / 'config' / doc['output']['scenario_subpath'] / 'lgbm_direct-pointwise.yaml', doc))
    destination = source.parent / 'freq_1day/demand_load.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(destination, index=False)
    report = {'source': str(source.relative_to(ROOT)), 'source_sha256': digest,
              'source_rows': len(frame), 'output_rows': len(daily), 'method': 'mean',
              'period': PERIOD, 'source_freq': '15min', 'target_freq': '1D'}
    destination.with_suffix('.aggregate.json').write_text(json.dumps(report, indent=2))
    for path, doc in documents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# 逐窗历史由 annual_backtest.py 调度；不要用 run.py 代替全年回放。\n' + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    return {**report, 'configs': [str(p.relative_to(ROOT)) for p, _ in documents]}


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
