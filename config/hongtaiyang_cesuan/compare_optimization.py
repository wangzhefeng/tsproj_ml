"""全年复核与逐配置推广门槛；仅读取已有结果，不执行模型。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from forecasting_core.specs.config import parse_model_config
from generate_configs import STRATEGIES, model_document
from optimize_1day import metrics


def compare(report: dict) -> dict:
    if report['completed'] != report['expected'] or report['expected'] != 8:
        raise ValueError('need all eight verified current 1D results')
    baseline = json.loads((ROOT / '.hermes/plans/hongtaiyang-optimization-baseline.json').read_text())
    by_identity = {}
    for item in baseline:
        out = ROOT / item['path']
        by_identity[json.loads((out / 'audit.json').read_text())['config']] = out
    profile = json.loads(Path(__file__).with_name('active_1day_profile.json').read_text())
    promoted = {site: {method: {} for method in STRATEGIES} for site in ('guangdianchang', 'xinnengyuan')}
    rows = []
    for item in report['results']:
        config = Path(item['config'])
        site, method = config.parts[2], config.stem.removeprefix('lgbm_')
        identity = parse_model_config(model_document(site, 'demand_load', True, method), source='baseline').fingerprint()
        old_path = by_identity[identity] / 'prediction.csv'
        new_path = Path(item['output']) / 'prediction.csv'
        old, new = [pd.read_csv(path, parse_dates=['time']) for path in (old_path, new_path)]
        if not old.time.equals(new.time):
            raise ValueError('baseline/current coverage differs')
        row = {'site': site, 'method': method, 'output': item['output'], 'options': profile[site][method], 'scores': {}}
        for name, months in [('all_year', range(1, 13)), ('model_period', range(2, 13)),
                             ('cold', (2,)), ('regular_model_months', range(3, 13))]:
            before, after = [metrics(frame[frame.time.dt.month.isin(months)]) for frame in (old, new)]
            row['scores'][name] = {'before': before, 'after': after,
                                  'mae_improvement_percent': 100 * (1 - after['MAE'] / before['MAE'])}
        regular = row['scores']['regular_model_months']
        annual = row['scores']['all_year']
        passed = (regular['after']['MAE'] <= .98 * regular['before']['MAE']
                  and regular['after']['RMSE'] <= 1.02 * regular['before']['RMSE']
                  and annual['after']['MAE'] < annual['before']['MAE'])
        row['regular_changes_passed'] = passed
        if passed:
            promoted[site][method] = profile[site][method]
        rows.append(row)
    return {'results': rows, 'profile_for_15min_load_only': promoted,
            'note': '15min only config transfer; no 15min runtime or PV efficacy validation; cold-start policy is not transferred'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.report.read_text())), ensure_ascii=False, indent=2))
