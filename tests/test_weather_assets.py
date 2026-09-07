"""共享天气资产的真实临时文件验证；所有元数据均为合成测试 fixture。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


def asset_fixture(root, *, kind="forecast"):
    raw = root / "raw.csv"
    raw.write_text("fixture source,not real supplier data\n")
    evidence = root / "evidence.txt"
    evidence.write_text("SYNTHETIC TEST FIXTURE release contract; not operational evidence\n")
    frame = pd.DataFrame({"time": ["2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"], "temperature_2m": [10.0, 12.0], "available_at": ["2026-01-01T00:00:00Z"] * 2})
    normalized = root / "normalized.csv"
    frame.to_csv(normalized, index=False)
    def ref(path):
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    snapshot = {
        "source_id": "synthetic", "product": "fixture", "model": "fixture", "snapshot_id": "run-1",
        "location_id": "fixture-site", "latitude": 30.0, "longitude": 120.0, "coordinate_system": "WGS84",
        "data_kind": kind, "timezone": "UTC", "init_time": "2025-12-31T18:00:00Z", "issued_at": "2025-12-31T23:00:00Z", "received_at": "2026-01-01T00:00:00Z",
        "evidence_class": "received_snapshot", "evidence_ref": evidence.name,
        "variables": [{"name": "temperature_2m", "column": "temperature_2m", "unit": "degC", "semantics": "point", "native_freq": "1h", "label": "left"}],
        "raw": [ref(raw), ref(evidence)], "normalized": ref(normalized),
    }
    manifest = {"schema_version": "weather_asset_v1", "snapshots": [snapshot]}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return {"manifest": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, manifest


class WeatherAssetsTest(unittest.TestCase):
    def test_wide_normalized_contract(self):
        """宽表契约：缺变量列/inf 拒绝，NaN 缺测单元格允许，逐行 available_at 合同不变。"""
        from data_loading.weather_generator.assets import WeatherAssetStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, manifest = asset_fixture(root)
            snapshot = WeatherAssetStore(root).load(reference)[0]
            self.assertEqual(snapshot.frame["value"].tolist(), [10.0, 12.0])
            # NaN 缺测单元格允许存储
            frame = pd.read_csv(root / "normalized.csv")
            frame.loc[0, "temperature_2m"] = ""
            frame.to_csv(root / "normalized.csv", index=False)
            manifest['snapshots'][0]['normalized']['sha256'] = hashlib.sha256((root / "normalized.csv").read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            reference = {'manifest': 'manifest.json', 'sha256': hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
            loaded = WeatherAssetStore(root).load(reference)[0]
            self.assertTrue(pd.isna(loaded.frame["value"].iloc[0]))
            # inf 拒绝
            frame = pd.read_csv(root / "normalized.csv")
            frame.loc[0, "temperature_2m"] = float("inf")
            frame.to_csv(root / "normalized.csv", index=False)
            manifest['snapshots'][0]['normalized']['sha256'] = hashlib.sha256((root / "normalized.csv").read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            reference = {'manifest': 'manifest.json', 'sha256': hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                WeatherAssetStore(root).load(reference)
            # 缺变量列拒绝
            frame = pd.DataFrame({"time": ["2026-01-01T01:00:00Z"], "available_at": ["2026-01-01T00:00:00Z"]})
            frame.to_csv(root / "normalized.csv", index=False)
            manifest['snapshots'][0]['normalized']['sha256'] = hashlib.sha256((root / "normalized.csv").read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            reference = {'manifest': 'manifest.json', 'sha256': hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "columns"):
                WeatherAssetStore(root).load(reference)

    def test_missing_invalid_metadata_and_late_evidence_are_rejected(self):
        from data_loading.weather_generator.assets import WeatherAssetStore
        mutations = [
            lambda m: m['snapshots'][0].update(latitude=None),
            lambda m: m['snapshots'][0]['variables'][0].update(unit='unknown'),
            lambda m: m['snapshots'][0].update(evidence_class='invented'),
            lambda m: m['snapshots'][0].update(received_at='2026-01-02T00:00:00Z'),
            lambda m: m['snapshots'].append(dict(m['snapshots'][0])),
            lambda m: m['snapshots'][0].update(evidence_ref='missing.txt'),
            lambda m: m['snapshots'][0].update(init_time='2026-01-02T00:00:00Z', issued_at=None),
            lambda m: m['snapshots'][0].update(issued_at='2026-01-01T00:15:00Z'),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest = asset_fixture(root)
                mutate(manifest)
                path = root / 'manifest.json'
                path.write_text(json.dumps(manifest))
                ref = {'manifest': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                with self.assertRaises((ValueError, TypeError)):
                    WeatherAssetStore(root).load(ref)

    def test_missing_asset_raises_instead_of_fetching(self):
        from data_loading.weather_generator.assets import WeatherAssetStore
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                WeatherAssetStore(directory).load({'manifest': 'absent.json', 'sha256': 'a' * 64})

    def test_read_verified_local_snapshot_and_reject_changed_dependency(self):
        from data_loading.weather_generator.assets import WeatherAssetStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, _ = asset_fixture(root)
            store = WeatherAssetStore(root)
            snapshots = store.load(reference)
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].frame["value"].tolist(), [10.0, 12.0])
            (root / "raw.csv").write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                store.load(reference)


if __name__ == "__main__":
    unittest.main()
