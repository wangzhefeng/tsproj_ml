# -*- coding: utf-8 -*-
"""一次性迁移 HVAC 原始宽表与 IT 分析结果；不重算、不覆盖，SHA256 验收。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / 'dataset/aidc_hvac_load_5min'
VERSIONS = ('hvac_all_devices', 'hvac_remove_devices')
ROUTES = ('route_A', 'route_B')
BUILDINGS = ('A1', 'A2', 'A3')
FAMILIES = tuple(f'{v}/{r}' for v in VERSIONS for r in ROUTES) + ('IT_load',)
FILES = tuple(f'{b}_data.csv' for b in BUILDINGS) + ('data.csv',)


def require_new_files(paths):
    for path in paths:
        if path.exists():
            raise FileExistsError(f'原始归档只允许首次生成，禁止覆盖: {path}')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root, manifest):
    for row in manifest['raw_files'] + manifest['analysis_files']:
        if sha256_file(root / row['destination']) != row['sha256']:
            raise ValueError(f"迁移文件哈希改变: {row['destination']}")


def migrate(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    manifest_path = root / 'analysis/raw_migration.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any((root / name).exists() for name in (*VERSIONS, 'IT_load')):
            raise FileExistsError('迁移记录已存在，但旧源目录再次出现；拒绝猜测来源')
        verify_manifest(root, manifest)
        return manifest
    for family in FAMILIES:
        for name in FILES:
            source = root / family / name
            if not source.is_file() or source.is_symlink():
                raise ValueError(f'原始输入不存在或是软链接: {source}')
    for name in (*VERSIONS, 'IT_load'):
        if (root / 'raw_data' / name).exists():
            raise FileExistsError(f'归档目标已存在: {name}')
    analysis_moves = []
    for source in sorted((root / 'IT_load/analysis').glob('*')):
        if not source.is_file() or source.is_symlink():
            raise ValueError(f'不支持的分析资产: {source}')
        name = 'IT_load_' + source.name if source.suffix == '.csv' and not source.name.startswith('IT_load_') else source.name
        destination = root / 'analysis' / name
        if destination.exists():
            raise FileExistsError(f'分析目标已存在: {destination}')
        analysis_moves.append((source, destination))
    raw_files = []
    for family in FAMILIES:
        for name in FILES:
            rel = f'{family}/{name}'
            raw_files.append({'source': rel, 'destination': 'raw_data/' + rel,
                              'sha256': sha256_file(root / rel)})
    analysis_files = [{'source': str(src.relative_to(root)), 'destination': str(dst.relative_to(root)),
                       'sha256': sha256_file(src)} for src, dst in analysis_moves]
    manifest = {'raw_files': raw_files, 'analysis_files': analysis_files}
    (root / 'raw_data').mkdir(exist_ok=True)
    (root / 'analysis').mkdir(exist_ok=True)
    moves = analysis_moves + [(root / name, root / 'raw_data' / name) for name in (*VERSIONS, 'IT_load')]
    completed = []
    try:
        for source, destination in moves:
            source.rename(destination)
            completed.append((source, destination))
        verify_manifest(root, manifest)
        with manifest_path.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    except Exception:
        for source, destination in reversed(completed):
            destination.rename(source)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = migrate(args.root)
    print(f"迁移核验通过: raw={len(manifest['raw_files'])}, analysis={len(manifest['analysis_files'])}")


if __name__ == '__main__':
    main()
