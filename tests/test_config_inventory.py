"""测试独立配置清单：集合覆盖不能依赖生产 scanner 自证或历史总数。"""
import tempfile
import unittest
from pathlib import Path

from fixtures.config_inventory import model_config_inventory


class ConfigInventoryTest(unittest.TestCase):
    def test_classifies_models_without_using_filename_or_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'nested').mkdir()
            (root / 'nested/plain.yaml').write_text('schema_version: 2\nestimator: {}\n')
            (root / 'blend.yaml').write_text('schema_version: 2\nensemble: {}\n')
            (root / 'process.yaml').write_text('input: raw.csv\noutput: prepared.csv\n')
            self.assertEqual(model_config_inventory(root), {
                'blend.yaml': 'ensemble', 'nested/plain.yaml': 'single_model',
            })

    def test_new_files_and_invalid_model_versions_are_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(model_config_inventory(root), {})
            # 清单只识别待审计对象；非法版本和缺版本仍交给生产校验报错。
            (root / 'old.yaml').write_text('schema_version: 1\nestimator: {}\n')
            (root / 'missing.yaml').write_text('estimator: {}\n')
            self.assertEqual(model_config_inventory(root), {
                'missing.yaml': 'single_model', 'old.yaml': 'single_model',
            })

    def test_rejects_unclassifiable_or_ambiguous_canonical_models(self):
        for document in ('schema_version: 2\nproblem: {}\n',
                         'schema_version: 2\nestimator: {}\nensemble: {}\n'):
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'invalid.yaml').write_text(document)
                with self.assertRaises(AssertionError):
                    model_config_inventory(root)


if __name__ == '__main__':
    unittest.main()
