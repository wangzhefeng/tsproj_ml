"""显式分段回放：保留旧窗口，切换后30天训练、预测次日；不修改父结果。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from model_pipeline.supervised_design import minimum_history_rows
from annual_backtest import forecast_window, schedule, assemble_year, write_json
from annual_reporting import write_annual_results
from prepare import validate_frame
from model_performance.checkpoints import implementation_fingerprint

PERIOD = {'start': '2025-09-01', 'end': '2026-09-01'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_window(csv, times):
    audit = json.loads(csv.with_suffix('.json').read_text())
    frame = pd.read_csv(csv, parse_dates=['time'], float_precision='round_trip')
    if (list(frame.columns) != ['time', 'y_pred'] or not pd.DatetimeIndex(frame.time).equals(times)
            or not np.isfinite(frame.y_pred).all() or sha(csv) != audit['csv_sha256']):
        raise ValueError(f'invalid checkpoint: {csv}')
    return frame, audit


def run(config_path, previous, switch, max_new=None):
    config = load_yaml_config(config_path)
    if config.problem.freq != '15min' or config.result_method()['method_label'] != 'direct-pointwise':
        raise ValueError('only 15min direct-pointwise supported')
    if max_new is not None and max_new < 1:
        raise ValueError('max-new must be positive')
    source = ROOT / config.data.sources[0].history_path
    actual = validate_frame(pd.read_csv(source), '15min', **PERIOD)
    windows = schedule('15min', period=PERIOD)
    switch = pd.Timestamp(switch)
    if switch not in [w[2][0] for w in windows]:
        raise ValueError('switch must be a scheduled day')
    parent = json.loads((previous / 'status.json').read_text())
    if parent.get('completed_windows') != sum(w[2][0] < switch for w in windows):
        raise ValueError('parent completed window count differs from switch')
    inherited = []
    fingerprints = []
    for _, _, times in windows:
        if times[0] >= switch:
            break
        csv = previous / 'windows' / f'{times[0]:%Y%m%d}.csv'
        frame, audit = checked_window(csv, times)
        inherited.append((frame, {**audit, 'segment': 'inherited_three_month_policy', 'inherited_from': str(csv)}))
        fingerprints.append({'csv': sha(csv), 'audit': sha(csv.with_suffix('.json'))})
    doc = yaml.safe_load(config_path.read_text())
    doc['validation']['train_history_steps'] = 2880
    doc['validation']['train_window_steps'] = 2880 - minimum_history_rows(config) - 96 + 1
    # 物理配置与分段配方一起保留；训练窗口由下方逐窗显式截断。
    from_config = config.fingerprint()
    config = replace(config, validation=doc['validation'])
    recipe = {'period': PERIOD, 'config': config.fingerprint(), 'parent_config': from_config,
              'source_sha256': sha(source), 'switch': switch.isoformat(), 'new_history_days': 30,
              'parent_output': str(previous), 'inherited_checkpoints': fingerprints,
              'evaluation_limit': max_new, 'runner_sha256': sha(__file__),
              'implementation': implementation_fingerprint(),
              'annual_code_sha256': sha(Path(__file__).with_name('annual_backtest.py')),
              'reporting_code_sha256': sha(Path(__file__).with_name('annual_reporting.py')),
              'semantics': 'mixed-policy backtest; September actual passthrough'}
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / 'results/results_test' / config.output['scenario_subpath'] / config.result_identity() / f'mixed_{identity}'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'windows').mkdir(exist_ok=True)
    (output / 'model_config.yaml').write_text(yaml.safe_dump(doc, sort_keys=False))
    write_json(output / 'recipe.json', recipe)
    frames, audits = [f for f, _ in inherited], [a for _, a in inherited]
    for (frame, audit), window in zip(inherited, windows):
        name = f'{window[2][0]:%Y%m%d}'
        # 保留原CSV字节和哈希，新增审计仅记录来源，不伪装成30天结果。
        csv = output / 'windows' / (name + '.csv')
        csv.write_bytes((previous / 'windows' / csv.name).read_bytes())
        write_json(csv.with_suffix('.json'), audit)
    new_count = 0
    write_json(output / 'status.json', {'status': 'running', 'completed_windows': len(frames), 'inherited_windows': len(inherited), 'expected_windows': len(windows)})
    try:
        for _, origin, times in windows[len(inherited):]:
            if max_new is not None and new_count >= max_new:
                break
            csv = output / 'windows' / f'{times[0]:%Y%m%d}.csv'
            if csv.exists() and csv.with_suffix('.json').exists():
                frame, audit = checked_window(csv, times)
            else:
                frame, audit = forecast_window(config, actual, (times[0] - pd.Timedelta(days=30), origin, times))
                audit['segment'] = 'rolling_30day'
                frame.to_csv(csv, index=False)
                audit['csv_sha256'] = sha(csv)
                write_json(csv.with_suffix('.json'), audit)
            frames.append(frame)
            audits.append(audit)
            new_count += 1
            write_json(output / 'status.json', {'status': 'running', 'completed_windows': len(frames), 'inherited_windows': len(inherited), 'expected_windows': len(windows)})
            print(f'{len(frames)}/{len(windows)} {times[0]:%Y-%m-%d} history={audit["history_rows"]} seconds={audit["seconds"]:.1f}', flush=True)
        complete = len(frames) == len(windows)
        write_json(output / 'audit.json', {**recipe, 'windows': audits, 'complete': complete})
        if complete:
            annual = assemble_year(actual, frames, '15min', period=PERIOD)
            annual.to_csv(output / 'prediction.csv', index=False)
            write_json(output / 'scores.json', write_annual_results(output, annual, '15min', recipe))
            rows = []
            for label, group in [('before_switch', annual[(annual.time >= '2025-10-01') & (annual.time < switch)]), ('after_switch', annual[annual.time >= switch])]:
                error = group.y_pred - group.y_true
                rows.append({'segment': label, 'rows': len(group), 'MAE': float(error.abs().mean()), 'RMSE': float(np.sqrt((error ** 2).mean())), 'Bias': float(error.mean())})
            pd.DataFrame(rows).to_csv(output / 'segment_scores.csv', index=False)
        state = {'status': 'completed' if complete else 'partial', 'completed_windows': len(frames), 'inherited_windows': len(inherited), 'expected_windows': len(windows), 'output': str(output)}
        write_json(output / 'status.json', state)
        return state
    except BaseException as error:
        write_json(output / 'status.json', {'status': 'failed', 'completed_windows': len(frames), 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-yaml', type=Path, required=True)
    parser.add_argument('--previous-output', type=Path, required=True)
    parser.add_argument('--switch', required=True)
    parser.add_argument('--max-new', type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config_yaml, args.previous_output, args.switch, args.max_new), indent=2))
