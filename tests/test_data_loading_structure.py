"""数据层子包布局、公共入口和旧序列化路径合同。"""

import importlib
import pickle
from pathlib import Path
import unittest

import data_loading
from data_loading.information import information_set, providers


ROOT = Path(__file__).resolve().parents[1] / 'data_loading'
GROUPS = {
    'sources': ('source_io', 'discovery', 'assets', 'provenance'),
    'processing': ('validation', 'visibility', 'alignment'),
    'information': ('information_set', 'indexing', 'providers'),
}


class DataLoadingStructureTest(unittest.TestCase):
    def test_implementation_layout_without_compatibility(self):
        for group, modules in GROUPS.items():
            self.assertTrue((ROOT / group / '__init__.py').is_file())
            for module in modules:
                with self.subTest(group=group, module=module):
                    loaded = importlib.import_module(f'data_loading.{group}.{module}')
                    assert loaded.__file__ is not None
                    self.assertEqual(Path(loaded.__file__).resolve(), ROOT / group / f'{module}.py')
        self.assertEqual(
            {path.name for path in ROOT.glob('*.py')},
            {'__init__.py', 'registry.py'},
        )


    def test_old_pickle_globals_are_rejected(self):
        # 协议 0 GLOBAL 直接模拟旧模块类路径；不改写既有冻结 fixture。
        for old, module, names in (
            ('information_set', information_set, ('InformationSetRequest', 'MaterializedInformationSet', 'SourceLineage')),
            ('providers', providers, ('AuxiliaryProvider', 'CompositeProvider', 'PersistenceProvider',
                                      'ProvidedScenarioProvider', 'ObservedFutureProvider')),
        ):
            for name in names:
                with self.subTest(module=old, name=name):
                    with self.assertRaises(ModuleNotFoundError):
                        pickle.loads(f'cdata_loading.{old}\n{name}\n.'.encode())
                    canonical = getattr(module, name)
                    self.assertIs(pickle.loads(pickle.dumps(canonical)), canonical)
        self.assertIs(data_loading.InformationSetRequest, information_set.InformationSetRequest)
        self.assertIs(data_loading.PersistenceProvider, providers.PersistenceProvider)


if __name__ == '__main__':
    unittest.main()
