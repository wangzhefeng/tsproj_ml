"""点位产物迁移：字节保真、历史身份保留及冲突前置拒绝。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from config.aidc_hvac_load_5min.scripts.v3.outlier_remove_data.relocate_point_tables import relocate
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

OLD = Path('analysis/data_v3/demand_response/point_tables')
NEW = Path('outlier_remove_data/data_v3/demand_response')
AUDIT = OLD.parent


class DemandResponseRelocationTest(unittest.TestCase):
    def fixture(self, root):
        hashes = {}
        for device in ('hvac_all_devices', 'hvac_remove_devices'):
            for route in ('route_A', 'route_B'):
                path = root / OLD / device / route / 'A1_data.csv'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'time,value\n2026-07-10 00:00:00,1.25\n')
                hashes[str(path.relative_to(root))] = sha256_file(path)
        plot = root / 'analysis/data_v3/forecast_data_visual/existing.png'
        plot.parent.mkdir(parents=True)
        plot.write_bytes(b'untouched-existing-artifact')
        manifest = root / AUDIT / 'manifest.json'
        manifest.write_text(json.dumps({'builder_sha256': 'historical-builder', 'artifact_sha256': hashes}))
        return manifest.read_bytes()

    def test_relocates_without_recomputing_and_preserves_generation_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.fixture(root)
            result = relocate(root)
            self.assertEqual(result['moved_tables'], 4)
            self.assertFalse((root / OLD).exists())
            report = json.loads((root / AUDIT / 'manifest.json').read_text())
            self.assertEqual(report['builder_sha256'], 'historical-builder')
            self.assertEqual((root / AUDIT / 'path_migration/original_manifest.json').read_bytes(), original)
            migration = json.loads((root / report['path_migration']).read_text())
            self.assertEqual(len(migration['path_mapping']), 4)
            for path, digest in migration['preserved_files_sha256'].items():
                self.assertEqual(sha256_file(root / path), digest)
            for path, digest in report['artifact_sha256'].items():
                self.assertTrue(path.startswith(str(NEW)))
                self.assertEqual(sha256_file(root / path), digest)
            with self.assertRaises(FileExistsError):
                relocate(root)

    def test_conflict_or_corrupt_source_fails_before_any_move(self):
        for conflict in (True, False):
            with self.subTest(conflict=conflict), TemporaryDirectory() as tmp:
                root = Path(tmp)
                original = self.fixture(root)
                if conflict:
                    (root / NEW).mkdir(parents=True)
                    error = FileExistsError
                else:
                    path = next((root / OLD).rglob('*.csv'))
                    path.write_text('corrupt')
                    error = ValueError
                with self.assertRaises(error):
                    relocate(root)
                self.assertEqual(len(list((root / OLD).rglob('*.csv'))), 4)
                self.assertEqual((root / AUDIT / 'manifest.json').read_bytes(), original)
                self.assertFalse((root / AUDIT / 'path_migration').exists())


if __name__ == '__main__':
    unittest.main()
