"""天气迁移前置只读审计；不把资产存在误报为全窗口验收。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_loader import is_model_yaml, load_yaml_config
from data_loading.sources.assets import source_paths


def audit_weather_configs(config_root, *, repository_root=ROOT):
    root = Path(repository_root).resolve()
    entries, errors = [], []
    model_count = 0
    for path in sorted(Path(config_root).rglob('*.yaml')):
        try:
            if not is_model_yaml(path):
                continue
            model_count += 1
            config = load_yaml_config(path)
            sources = [source for source in config.data.sources if source.generator == 'weather' or 'weather' in source.name.lower() or any('weather' in value.lower() for _, value in source_paths(source))]
            if not sources and 'weather' not in path.stem.lower():
                continue
            payload = config.canonical_payload()
            blockers, asset_errors = [], []
            source_records = []
            for source in sources:
                paths = dict(source_paths(source))
                if source.generator == 'weather':
                    # 2026-09-07 起活动配置已全部切换到 file 两段制 + inference_columns，generator 源仅存于研究/下载工具链
                    blockers.append('generator_source_retired_from_active_configs')
                else:
                    inference_map = dict(getattr(source, 'inference_columns', None) or ())
                    for name in (c.name for c in source.columns if c.role.value == 'known_future'):
                        if name not in inference_map:
                            blockers.append('known_future_column_without_inference_mapping')
                    for key, value in inference_map.items():
                        if not any(c.name == value and c.role.value == 'ignored' for c in source.columns):
                            blockers.append(f'inference_target_not_declared_ignored:{key}->{value}')
                source_records.append({'name': source.name, 'generator': source.generator, 'paths': paths, 'inference_columns': dict(getattr(source, 'inference_columns', None) or ()), 'columns': [{'name': column.name, 'role': column.role.value} for column in source.columns]})
            if not sources:
                blockers.append('weather_named_config_without_identified_source')
            if blockers:
                entries.append({'config': str(path.resolve().relative_to(root)), 'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'semantic_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), 'problem': payload['problem'], 'validation': payload.get('validation'), 'features': payload.get('features'), 'strategy': payload.get('strategy'), 'estimator': payload.get('estimator'), 'ensemble': payload.get('ensemble'), 'sources': source_records, 'asset_errors': asset_errors, 'status': 'blocked', 'blockers': sorted(set(blockers)), 'windows_checked': 0})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({'config': str(path), 'error': str(exc)})
    counts = Counter(reason for entry in entries for reason in entry['blockers'])
    return {'audit_scope': 'migration_preflight_not_runtime_window_acceptance', 'model_count': model_count, 'weather_config_count': len(entries), 'blocked_count': len(entries), 'blocker_counts': dict(sorted(counts.items())), 'errors': errors, 'complete': model_count > 0 and not entries and not errors, 'configs': entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_root = root / 'config' if (root / 'config').is_dir() else root
    report = audit_weather_configs(config_root, repository_root=root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'configs'}, ensure_ascii=False))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
