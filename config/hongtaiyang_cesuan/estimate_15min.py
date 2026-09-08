"""读取已冻结的历史15min估算；原始计时结果已删除，不重估当前配置。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from generate_configs import STRATEGIES


def estimate(daily_report: dict) -> dict:
    if daily_report['expected'] != 8 or daily_report['completed'] != 8:
        raise ValueError('need the eight current non-recursive 1D results')
    path = ROOT / '.hermes/plans/hongtaiyang-15min-estimate.json'
    historical = json.loads(path.read_text())
    rows = [row for row in historical['methods'] if row['method'] in STRATEGIES]
    if len(rows) != len(STRATEGIES):
        raise ValueError('frozen historical report has incomplete method coverage')
    return {'historical_only': True, 'original_measurements_available': False,
            'ran_15min_models': False, 'source_report': str(path), 'methods': rows,
            'warning': '原始15min计时结果已按授权删除；这里只展示旧配置的冻结估算，不代表当前优化配置耗时，不能作为新预算。'}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(estimate(json.loads(args.daily_report.read_text())), ensure_ascii=False, indent=2))
