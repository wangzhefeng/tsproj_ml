"""显式批准的跨楼事件修正；仅替换点位表及其严格汇总，不触碰预测输入。"""
from pathlib import Path
import argparse
import json
import shutil
import sys
from tempfile import TemporaryDirectory
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import read_table, write_csv
from config.aidc_hvac_load_5min.scripts.imputed_data.prepare_buildings import aggregate, DEVICES, ROUTES
from config.aidc_hvac_load_5min.scripts.outlier_remove_data.adjust_demand_response import correct_points
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file
AUDIT = Path('analysis/building_extension/event_corrections')


def prepare(root, recipe):
    root = Path(root).resolve()
    if recipe.get('approved') is not True or set(recipe['events']) != {'A2', 'A3'}:
        raise ValueError('未明确批准跨楼事件')
    if (root / AUDIT).exists():
        raise FileExistsError('事件修正审计已存在，拒绝重复修正')
    expected = {f'{category}/{device}/{route}/{building}_data.csv'
                for category in ('raw_data', 'imputed_data') for device in DEVICES
                for route in ROUTES for building in ('A2', 'A3')}
    if set(recipe['input_sha256']) != expected:
        raise ValueError('输入身份清单不完整')
    for name, digest in recipe['input_sha256'].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f'输入身份变化: {name}')
    report = {'recipe': recipe, 'forecast_data_touched': False, 'methods': [], 'tables': {}}
    with TemporaryDirectory(prefix='hvac-events-') as tmp:
        stage, backup = Path(tmp) / 'stage', Path(tmp) / 'backup'
        for building, events in recipe['events'].items():
            for route in ROUTES:
                key = f'hvac_all_devices/{route}/{building}_data.csv'
                raw = read_table(root / 'raw_data' / key).drop(columns='total_load')
                parent = read_table(root / 'imputed_data' / key).drop(columns='total_load')
                corrected, methods, changes, blocks = correct_points(raw, parent, events, on_unfillable='keep')
                report['methods'].extend(methods.assign(building=building, route=route).to_dict('records'))
                for name, frame in [('methods', methods), ('changes', changes), ('validation_blocks', blocks)]:
                    target = stage / AUDIT / building / route / (name + '.csv')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(target, index=False)
                for device in DEVICES:
                    key = f'{device}/{route}/{building}_data.csv'
                    subset = read_table(root / 'imputed_data' / key).drop(columns='total_load')
                    pd.testing.assert_frame_equal(parent[subset.columns], subset)
                    table = corrected[subset.columns].copy()
                    table['total_load'] = table.sum(axis=1, min_count=len(subset.columns))
                    relative = Path('outlier_remove_data') / key
                    if not (root / relative).is_file():
                        raise FileNotFoundError(root / relative)
                    write_csv(table, stage / relative)
                    report['tables'][str(relative)] = {'before_sha256': sha256_file(root / relative),
                                                       'after_sha256': sha256_file(stage / relative)}
        for device in DEVICES:
            for route in ROUTES:
                family = f'{device}/{route}'
                parts = {}
                for building in ('A1', 'A2', 'A3'):
                    rel = Path('outlier_remove_data') / family / f'{building}_data.csv'
                    parts[building] = read_table(stage / rel if (stage / rel).exists() else root / rel)
                rel = Path('outlier_remove_data') / family / 'data.csv'
                write_csv(aggregate(parts, read_table(root / 'raw_data' / family / 'data.csv').index), stage / rel)
                report['tables'][str(rel)] = {'before_sha256': sha256_file(root / rel), 'after_sha256': sha256_file(stage / rel)}
        report['unresolved_point_events'] = sum(r['method'] == 'unresolved' for r in report['methods'])
        report['estimated_point_events'] = len(report['methods']) - report['unresolved_point_events']
        report['output_sha256'] = {str(p.relative_to(stage)): sha256_file(p) for p in stage.rglob('*.csv')}
        manifest = stage / AUDIT / 'manifest.json'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10))
        parent_rel = Path('analysis/building_extension/manifest.json')
        parent_report = json.loads((root / parent_rel).read_text())
        parent_report['pre_event_output_sha256'] = parent_report['output_sha256'].copy()
        for name, row in report['tables'].items():
            parent_report['output_sha256'][name] = row['after_sha256']
        parent_report['anomaly_status'] = 'five authorized intervals estimated; unresolved point-events explicitly audited'
        parent_report['event_corrections'] = str(AUDIT / 'manifest.json')
        (stage / parent_rel).write_text(json.dumps(parent_report, ensure_ascii=False, indent=2) + chr(10))
        # Preflight all reads and hashes before replacing anything. Per-file rollback, no cross-directory atomicity claim.
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        originals = {p.relative_to(stage): (root / p.relative_to(stage)).exists() for p in files}
        for rel, exists in originals.items():
            if exists:
                dest = backup / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, dest)
        for name, digest in recipe['input_sha256'].items():
            if sha256_file(root / name) != digest:
                raise ValueError('发布前源发生变化')
        applied = []
        try:
            for source in files:
                rel = source.relative_to(stage)
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                source.replace(dest)
                applied.append(rel)
        except Exception:
            for rel in reversed(applied):
                dest = root / rel
                if originals[rel]:
                    (backup / rel).replace(dest)
                else:
                    dest.unlink()
            raise
    return {'estimated_point_events': report['estimated_point_events'],
            'unresolved_point_events': report['unresolved_point_events'], 'tables': len(report['tables'])}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'dataset/aidc_hvac_load_5min')
    parser.add_argument('--recipe', type=Path, default=Path(__file__).with_name('building_events.json'))
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, json.loads(args.recipe.read_text())), ensure_ascii=False))
