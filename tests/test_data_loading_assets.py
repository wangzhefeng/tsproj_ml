"""真实单模型/融合 YAML 分派与轻量资产检查行为保真。"""
import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.audit_runtime_assets import audit_runtime_assets
from config.config_loader import load_yaml_config
from data_loading import BUILTIN_GENERATORS
from forecasting_core.specs import ColumnSpec, DataSourceSpec, DataSpec, ForecastConfigSpec
from feature_engineering import cache
from unittest.mock import patch
import test_ensemble_loader as fixtures


def model_document(document):
    result = copy.deepcopy(document)
    if 'estimator' in result:
        result['validation'].update(history_steps=32, train_window_steps=8, fold_count=2, stride_steps=2)
    return result


class DataLoadingAssetBehaviorTest(unittest.TestCase):
    def test_typed_asset_core_preserves_header_context(self):
        self.assertIsNotNone(importlib.util.find_spec('data_loading.sources.assets'))
        assets = importlib.import_module('data_loading.sources.assets')
        source = DataSourceSpec(
            name='weather', source_type='file', time_col='time', availability='column', available_at_col='issued_at',
            columns=(ColumnSpec('temperature', 'known_future'), ColumnSpec('optional', 'ignored')),
            history_path='history.csv', backtest_path='backtest.csv', future_path='future.csv',
        )
        self.assertEqual(assets.required_columns(source), {'time', 'temperature'})
        # 轻量表头审计原本不要求 issued_at；完整运行的非 history 校验仍要求它。
        for role, path in assets.source_paths(source):
            self.assertIn(role, ('history_path', 'backtest_path', 'future_path'))
            self.assertIsNone(assets.asset_columns(path, self.root))
            (self.root / path).write_text('time,temperature,extra\n')
            self.assertEqual(assets.asset_columns(path, self.root), {'time', 'temperature', 'extra'})
        generated = DataSourceSpec(name='calendar', source_type='generated', generator='chinese_holiday',
                                   time_col='time', availability='generator_defined',
                                   columns=(ColumnSpec('is_holiday', 'known_future'),))
        self.assertEqual(assets.source_paths(generated), ())

    def test_source_provenance_is_owned_by_data_layer(self):
        self.assertIsNotNone(importlib.util.find_spec('data_loading.sources.provenance'))
        provenance = importlib.import_module('data_loading.sources.provenance')
        path = self.root / 'data.csv'
        path.write_text('time,load\n2026-01-01,1.0\n')
        config = load_yaml_config(self.root / 'direct.yaml')
        assert isinstance(config, ForecastConfigSpec)
        expected = {'targets:history_path': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual(provenance.source_hashes(config.data, self.root), expected)
        self.assertIs(cache.file_sha256, provenance.file_sha256)
        payload = cache.raw_design_provenance(config, base_dir=self.root, origin='2026-01-03', generators={})
        self.assertEqual(payload['source_hashes'], expected)
        self.assertEqual(provenance.generator_hashes(config.data, {}), {})
        calendar = DataSourceSpec(name='calendar', source_type='generated', generator='chinese_holiday',
                                   time_col='time', availability='generator_defined',
                                   columns=(ColumnSpec('is_holiday', 'known_future'),))
        data = DataSpec((*config.data.sources, calendar))
        with self.assertRaisesRegex(ValueError, 'no generator registered for compiled cache source'):
            provenance.generator_hashes(data, {})
        hashes = provenance.generator_hashes(data, BUILTIN_GENERATORS)
        self.assertEqual(set(hashes), {'calendar'})
        self.assertEqual(len(hashes['calendar']), 64)
        calendar_path = 'data_loading/calendar_generator/calendar_features.py'
        self.assertIn(calendar_path, payload['compilation_implementation_hashes'])
        original = cache.file_sha256
        before = cache.compute_raw_design_fingerprint(config, base_dir=self.root, origin='2026-01-03', generators={})
        with patch.object(cache, 'file_sha256', side_effect=lambda path: 'changed-calendar' if Path(path).name == 'calendar_features.py' else original(path)):
            self.assertNotEqual(before, cache.compute_raw_design_fingerprint(config, base_dir=self.root, origin='2026-01-03', generators={}))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, doc in (('direct.yaml', fixtures.SINGLE_MODEL), ('recursive.yaml', fixtures.SINGLE_MODEL),
                          ('ensemble.yaml', fixtures.ENSEMBLE_DOC)):
            (self.root / name).write_text(yaml.safe_dump(model_document(doc)))

    def test_real_single_and_ensemble_missing_paths(self):
        report = audit_runtime_assets(self.root, repository_root=self.root)
        self.assertEqual(report['model_config_count'], 3)
        self.assertEqual(report['missing_reference_count'], 3)
        self.assertEqual(report['missing_unique_path_count'], 1)
        self.assertEqual(report['missing_paths'][0]['path'], 'data.csv')
        self.assertEqual([r['config'] for r in report['missing_paths'][0]['references']],
                         ['direct.yaml', 'ensemble.yaml', 'recursive.yaml'])

    def test_header_missing_extra_and_bom(self):
        (self.root / 'data.csv').write_text('\ufefftime,extra\n')
        report = audit_runtime_assets(self.root, repository_root=self.root)
        self.assertEqual(report['missing_declared_column_reference_count'], 3)
        self.assertEqual(report['missing_declared_columns'][0]['references'][0]['missing_columns'], ['load'])
        (self.root / 'data.csv').write_text('\ufefftime,load,extra\n')
        report = audit_runtime_assets(self.root, repository_root=self.root)
        self.assertEqual(report['missing_declared_columns'], [])
        self.assertEqual(report['missing_paths'], [])

    def test_empty_file_and_directory_errors_are_not_hidden(self):
        (self.root / 'data.csv').write_text('')
        with self.assertRaisesRegex(ValueError, 'runtime source is empty:'):
            audit_runtime_assets(self.root, repository_root=self.root)
        # 不删除测试文件，使用独立临时根构造目录误配。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data.csv').mkdir()
            (root / 'model.yaml').write_text(yaml.safe_dump(model_document(fixtures.SINGLE_MODEL)))
            with self.assertRaises(IsADirectoryError):
                audit_runtime_assets(root, repository_root=root)


if __name__ == '__main__':
    unittest.main()
