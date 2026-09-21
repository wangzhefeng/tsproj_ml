"""data_v2：先统一屏蔽异常，再从屏蔽后的原观测统一填补，不继承data_v1估计值。"""
from pathlib import Path
import argparse
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config.aidc_hvac_load_5min.scripts.v2.outlier_remove_data.clean_hvac_data import build_cleaned, RECIPE
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import impute_dataset, publish_stage
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import DEFAULT_ROOT, VERSIONS, sha256_file


def prepare(root=DEFAULT_ROOT, recipe_path=RECIPE, *, excluded_it_points=None):
    root = Path(root).resolve()
    for name in ('outlier_remove_data', 'imputed_data', 'analysis'):
        if (root / name / 'data_v2').exists():
            raise FileExistsError(f'data_v2 已存在，拒绝覆盖: {name}')
    with tempfile.TemporaryDirectory(prefix='.prepare-v2-', dir=root) as tmp:
        stage = Path(tmp)
        anomalies = build_cleaned(root, recipe_path, output_root=stage)
        events = {}
        for row in anomalies.to_dict('records'):
            if not pd.notna(row['detection_known_at']):
                row['detection_known_at'] = None
            for version in VERSIONS:
                key = f"{version}/{row['route']}/{row['building']}_data.csv"
                events.setdefault(key, []).append(row)
        inventory = impute_dataset(stage, excluded_it_points, data_version='data_v2',
                                   source_root=stage / 'outlier_remove_data/data_v2', detection_events=events)
        audit = stage / 'analysis/data_v2'
        inventory['outlier_cleaning'] = {'manifest': 'analysis/data_v2/outliers/manifest.json',
                                         'sha256': sha256_file(audit / 'outliers/manifest.json')}
        inventory['causality'] = ('OFFLINE preparation: manual red boxes and centered isolated detection; '
                                 'values and method scoring use only pre-gap masked original observations. '
                                 'No historical imputed values are reused; unresolved gaps stay NaN; not deployable.')
        (audit / 'imputation/manifest.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
        (audit / 'README.md').write_text(
            '# data_v2 数据准备\n\n'
            'raw_data → outlier_remove_data/data_v2（异常置NaN） → imputed_data/data_v2（统一填补） → forecast_data/data_v2。\n\n'
            '异常判定保留孤立点和人工红框授权范围；不清洗IT。原缺失和异常缺失合并判断长度，'
            '长于6小时、首尾或验证不足的段保留NaN，不恢复异常值。不消费data_v1填充值。\n\n'
            'outliers/保存候选、屏蔽点和来源；imputation/保存全部缺口、选型与掩码。'
            '孤立点使用后侧15min、红框为人工离线决定；不能宣称严格在线可得。\n')
        # 发布前源哈希复核；现行data_v1和模型结果始终只读。
        manifest = json.loads((audit / 'outliers/manifest.json').read_text())
        for key, digest in manifest['source_sha256'].items():
            if sha256_file(root / 'raw_data' / key) != digest:
                raise ValueError(f'原始源改变: {key}')
        publish_stage(stage, root)
    return inventory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    args = parser.parse_args()
    result = prepare(args.root, args.recipe)
    print(f"data_v2 填补完成: {len(result['files'])}份表；未修改data_v1，未运行模型")
