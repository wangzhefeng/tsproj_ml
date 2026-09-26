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
            (root / 'nested/plain.yaml').write_text('problem: {}\ndata: {}\nestimator: {}\n')
            (root / 'blend.yaml').write_text('problem: {}\ndata: {}\nensemble: {}\n')
            (root / 'process.yaml').write_text('input: raw.csv\noutput: prepared.csv\n')
            self.assertEqual(model_config_inventory(root), {
                'blend.yaml': 'ensemble', 'nested/plain.yaml': 'single_model',
            })

    def test_stale_version_field_does_not_change_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(model_config_inventory(root), {})
            # 清单只识别待审计对象；仍声明历史 schema_version 的文件照常分类，
            # 字段合法性交给生产 parser 按未知字段报错。
            (root / 'stale.yaml').write_text('schema_version: 2\nproblem: {}\ndata: {}\nestimator: {}\n')
            (root / 'plain.yaml').write_text('problem: {}\ndata: {}\nestimator: {}\n')
            self.assertEqual(model_config_inventory(root), {
                'plain.yaml': 'single_model', 'stale.yaml': 'single_model',
            })

    def test_rejects_unclassifiable_or_ambiguous_canonical_models(self):
        for document in ('problem: {}\ndata: {}\n',
                         'estimator: {}\nensemble: {}\n'):
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'invalid.yaml').write_text(document)
                with self.assertRaises(AssertionError):
                    model_config_inventory(root)


if __name__ == '__main__':
    unittest.main()
