"""显式本地天气资产准备；默认 dry-run，从不删除或覆盖旧文件。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data_loading.weather_generator.adapters import open_meteo_frame, vendor_frame
from data_loading.weather_generator.assets import WeatherAssetStore, checked_bytes, strict_json
from data_loading.weather_generator.contracts import validate_snapshot_metadata


def publish(path, raw):
    """临时文件完成后原子 hard-link 发布；存在时逐字节核对，不覆盖。"""
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"immutable weather destination conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError(f"concurrent weather destination conflict: {path}")
    finally:
        temporary.unlink()


def register(args):
    metadata_bytes = args.metadata.read_bytes()
    meta = strict_json(metadata_bytes)
    validate_snapshot_metadata(meta)
    for field in ('source_id','location_id','snapshot_id'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', meta[field]):
            raise ValueError(f'{field} must be a safe path identifier')
    base_dir = args.base_dir.resolve()
    args.output_root.resolve().relative_to(base_dir)
    def configured_path(path):
        return path.resolve().relative_to(base_dir).as_posix()
    raw = args.input.read_bytes()
    if not any((args.base_dir / r['path']).resolve() == args.input.resolve() and hashlib.sha256(raw).hexdigest() == r['sha256'] for r in meta['raw']):
        raise ValueError('input must be declared in hashed raw dependencies')
    if args.adapter == 'vendor':
        if args.availability_csv is not None:
            raise ValueError('vendor uses --available-at-col, not --availability-csv')
        frame = vendor_frame(pd.read_csv(io.BytesIO(raw)), meta, time_col=args.time_col, available_at_col=args.available_at_col)
    else:
        availability = None
        if args.available_at_col is not None:
            raise ValueError('Open-Meteo uses --availability-csv, not --available-at-col')
        if args.availability_csv is not None:
            matches = [ref for ref in meta['raw'] if (args.base_dir / ref['path']).resolve() == args.availability_csv.resolve()]
            if len(matches) != 1:
                raise ValueError('availability CSV must be declared exactly once in raw dependencies')
            availability = pd.read_csv(io.BytesIO(checked_bytes(args.base_dir, matches[0])))
        frame = open_meteo_frame(strict_json(raw), meta, availability=availability)
    writes = []
    references = []
    metadata_reference = {'path': str(args.metadata.resolve()), 'sha256': hashlib.sha256(metadata_bytes).hexdigest()}
    if metadata_reference not in meta['raw']:
        meta['raw'].append(metadata_reference)
    for dependency in meta['raw']:
        content = checked_bytes(args.base_dir, dependency)
        path = args.output_root / 'raw' / 'objects' / dependency['sha256'] / 'payload'
        writes.append((path, content))
        if dependency['path'] == meta['evidence_ref']:
            meta['evidence_ref'] = configured_path(path)
        references.append({'path': configured_path(path), 'sha256': dependency['sha256']})
        origin = json.dumps({'original_path': str((args.base_dir / dependency['path']).resolve()), 'sha256': dependency['sha256']}, sort_keys=True).encode()
        writes.append((path.parent / ('origin-' + hashlib.sha256(origin).hexdigest() + '.json'), origin))
    meta['raw'] = references
    normalized = frame.to_csv(index=False).encode()
    norm_hash = hashlib.sha256(normalized).hexdigest()
    norm_path = args.output_root / 'normalized' / norm_hash / 'data.csv'
    writes.append((norm_path, normalized))
    meta['normalized'] = {'path': configured_path(norm_path), 'sha256': norm_hash}
    manifest = json.dumps({'schema_version': 'weather_asset_v1', 'snapshots': [meta]}, ensure_ascii=False, sort_keys=True).encode()
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    manifest_path = args.output_root / 'manifests' / (manifest_hash + '.json')
    reference = {'manifest': configured_path(manifest_path), 'sha256': manifest_hash}
    index_path = args.output_root / 'raw' / 'by-source' / meta['source_id'] / meta['location_id'] / meta['snapshot_id'] / (manifest_hash + '.json')
    if args.write:
        for path, content in writes:
            publish(path, content)
        publish(manifest_path, manifest)
        WeatherAssetStore(args.base_dir).load(reference)
        publish(index_path, json.dumps(reference, sort_keys=True).encode())
    return {'write': args.write, 'reference': reference, 'rows': len(frame)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    archive = sub.add_parser('archive', help='原样保全未知/已知来源文件，不赋单位或可得性')
    archive.add_argument('--input', type=Path, required=True)
    archive.add_argument('--output-root', type=Path, required=True)
    archive.add_argument('--source-id', required=True)
    archive.add_argument('--write', action='store_true')
    registration = sub.add_parser('register', help='核实本地元数据与原响应，规范化并发布不可变 manifest')
    registration.add_argument('--input', type=Path, required=True)
    registration.add_argument('--metadata', type=Path, required=True)
    registration.add_argument('--adapter', choices=['vendor', 'open_meteo'], required=True)
    registration.add_argument('--time-col', default='time')
    registration.add_argument('--available-at-col')
    registration.add_argument('--availability-csv', type=Path)
    registration.add_argument('--base-dir', type=Path, required=True)
    registration.add_argument('--output-root', type=Path, required=True)
    registration.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.command == 'register':
        print(json.dumps(register(args)))
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.source_id):
        parser.error('source-id must be a simple identifier')
    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    path = args.output_root / 'raw' / 'objects' / digest / 'payload'
    provenance = {'original_path': str(args.input.resolve()), 'source_id': args.source_id, 'sha256': digest, 'production_eligible': False, 'status': 'raw_preservation_only_metadata_unverified'}
    provenance_bytes = json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode()
    origin_id = hashlib.sha256(str(args.input.resolve()).encode()).hexdigest()
    if args.write:
        publish(path, raw)
        publish(path.parent / f'origin-{args.source_id}-{origin_id}.json', provenance_bytes)
    print(json.dumps({'write': args.write, 'path': str(path), 'sha256': digest, 'production_eligible': False}))


if __name__ == '__main__':
    main()
