"""原字节迁移v3需求响应点位表；不重算预测输入、估计或图片。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

AUDIT = Path('analysis/data_v3/demand_response')
OLD = AUDIT / 'point_tables'
NEW = Path('outlier_remove_data/data_v3/demand_response')


def relocate(root):
    root = Path(root).resolve()
    source, destination = root / OLD, root / NEW
    history = root / AUDIT / 'path_migration'
    if destination.exists() or history.exists():
        raise FileExistsError('新目录或迁移记录已存在，拒绝覆盖')
    expected = {Path(d) / r / 'A1_data.csv'
                for d in ('hvac_all_devices', 'hvac_remove_devices') for r in ('route_A', 'route_B')}
    actual = {p.relative_to(source) for p in source.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('旧目录必须恰有四份A1点位表')
    manifest_path = root / AUDIT / 'manifest.json'
    original_bytes = manifest_path.read_bytes()
    original_sha = sha256_file(manifest_path)
    manifest = json.loads(original_bytes)
    mapping = {str(OLD / p): str(NEW / p) for p in sorted(expected)}
    if not set(mapping).issubset(manifest['artifact_sha256']):
        raise ValueError('manifest缺少旧点位表身份')
    for path, digest in manifest['artifact_sha256'].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f'迁移前产物SHA错误: {path}')
    # 冻结本版本的CSV/PNG，验证搬迁不改变值、mask、审计或可视化。
    protected = {str(p.relative_to(root)): sha256_file(p)
                 for category in ('analysis/data_v3', 'forecast_data/data_v3', 'imputed_data/data_v3', 'weather_data/data_v3')
                 for p in (root / category).rglob('*') if p.is_file() and p.suffix in ('.csv', '.png')}
    record = {'operation': 'path_only_no_recomputation', 'path_mapping': mapping,
              'original_manifest': str(AUDIT / 'path_migration/original_manifest.json'),
              'original_manifest_sha256': original_sha,
              'preserved_files_sha256': {mapping.get(p, p): digest for p, digest in protected.items()},
              'migration_script_sha256': sha256_file(Path(__file__))}
    manifest['artifact_sha256'] = {mapping.get(p, p): digest for p, digest in manifest['artifact_sha256'].items()}
    manifest['path_migration'] = str(AUDIT / 'path_migration/migration.json')
    # 生成时recipe/code SHA原样保留；当前路径与历史生成事实分开。
    history.mkdir(parents=True)
    with (history / 'original_manifest.json').open('xb') as stream:
        stream.write(original_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    for path, digest in record['preserved_files_sha256'].items():
        if sha256_file(root / path) != digest:
            raise AssertionError(f'迁移后字节改变: {path}')
    (history / 'migration.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary_manifest = history / 'current_manifest.tmp'
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary_manifest.replace(manifest_path)
    if source.exists() or {p.relative_to(destination) for p in destination.rglob('*.csv')} != expected:
        raise AssertionError('迁移目录集合错误')
    return {'moved_tables': len(mapping), 'preserved_csv_png': len(protected),
            'destination': str(destination), 'manifest': str(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'dataset/aidc_hvac_load_5min')
    args = parser.parse_args()
    print(json.dumps(relocate(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
