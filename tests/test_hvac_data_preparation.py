"""AIDC HVAC 离线准备合同；新测试默认 integration。"""
from pathlib import Path
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'config/aidc_hvac_load_5min/scripts'
sys.path.insert(0, str(SCRIPTS))
from config.aidc_hvac_load_5min.scripts.imputed_data import impute_hvac_data as imputer


class MigrationTest(unittest.TestCase):
    def test_collision_fails_before_any_move(self):
        migration = importlib.import_module('config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in migration.FAMILIES:
                folder = root / family
                folder.mkdir(parents=True)
                for name in migration.FILES:
                    (folder / name).write_text('raw', encoding='utf-8')
            (root / 'IT_load/analysis').mkdir()
            (root / 'IT_load/analysis/missing_summary.csv').write_text('source', encoding='utf-8')
            (root / 'analysis/raw_diagnostics/IT_load').mkdir(parents=True)
            (root / 'analysis/raw_diagnostics/IT_load/IT_load_missing_summary.csv').write_text('destination', encoding='utf-8')
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):
                migration.migrate(root)
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_archive_guard_refuses_any_existing_target(self):
        migration = importlib.import_module('config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'raw_data/A1_data.csv'
            migration.require_new_files([target])
            target.parent.mkdir()
            target.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                migration.require_new_files([root / 'other.csv', target])
            self.assertEqual(target.read_bytes(), b'original')

    def test_migration_preserves_bytes_and_resolves_analysis_collisions(self):
        migration = importlib.import_module('config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            families = ['hvac_all_devices/route_A', 'hvac_all_devices/route_B',
                        'hvac_remove_devices/route_A', 'hvac_remove_devices/route_B', 'IT_load']
            for family in families:
                folder = root / family
                folder.mkdir(parents=True)
                for name in ['A1_data.csv', 'A2_data.csv', 'A3_data.csv', 'data.csv']:
                    (folder / name).write_bytes(b'time,total_load\n2026-01-01,1\n')
            (root / 'analysis').mkdir()
            (root / 'analysis/missing_summary.csv').write_bytes(b'hvac')
            (root / 'IT_load/analysis').mkdir()
            (root / 'IT_load/analysis/missing_summary.csv').write_bytes(b'it')
            result = migration.migrate(root)
            self.assertEqual(len(result['raw_files']), 20)
            self.assertEqual((root / 'analysis/missing_summary.csv').read_bytes(), b'hvac')
            self.assertEqual((root / 'analysis/raw_diagnostics/IT_load/IT_load_missing_summary.csv').read_bytes(), b'it')
            self.assertTrue((root / 'analysis/preparation/raw_migration.json').is_file())
            self.assertFalse((root / 'IT_load').exists())
            self.assertEqual(migration.migrate(root), result)
            for row in result['raw_files']:
                self.assertEqual(migration.sha256_file(root / row['destination']), row['sha256'])


if __name__ == '__main__':
    unittest.main()
