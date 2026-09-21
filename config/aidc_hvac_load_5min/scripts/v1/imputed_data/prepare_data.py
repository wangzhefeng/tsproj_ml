"""data_v1 初始缺失填补：直接使用raw，后续再按旧孤立/红框流程清洗。"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import impute_dataset
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import DEFAULT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = impute_dataset(args.root, data_version='data_v1')
    print(f"data_v1 初始填补完成: {len(result['files'])} 份表；已有资产拒绝覆盖")


if __name__ == '__main__':
    main()
