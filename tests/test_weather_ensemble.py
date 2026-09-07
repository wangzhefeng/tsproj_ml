"""融合 OOF 的天气传递身份；不改变既有纯文件缓存身份。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from data_loading import SourceRegistry
from data_loading.sources.provenance import source_hashes
from forecasting_core.specs.data import DataSourceSpec, ColumnSpec
from test_weather_registry import weather_data


class WeatherEnsembleTest(unittest.TestCase):
    def test_weather_assets_and_implementation_enter_member_identity(self):
        from model_ensemble.cache import member_source_hashes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = weather_data(root)
            registry = SourceRegistry(data,root)
            result = member_source_hashes('member',data,root,registry.generators)
            self.assertTrue(any('raw.csv' in key for key in result))
            self.assertIn('member:weather:generator_implementation', result)
            files = replace(data,sources=(data.sources[0],))
            self.assertEqual(member_source_hashes('member',files,root,registry.generators), {'member:'+k:v for k,v in source_hashes(files,root).items()})
            unknown = DataSourceSpec(name='unknown',source_type='generated',generator='custom',time_col='time',availability='generator_defined',columns=(ColumnSpec('custom','known_future'),))
            with self.assertRaises(ValueError):
                member_source_hashes('member',replace(data,sources=(data.sources[0],unknown)),root,{'custom':lambda s,r:None})
            (root / 'raw.csv').write_text('corruption')
            with self.assertRaises(ValueError):
                member_source_hashes('member',data,root,registry.generators)


if __name__ == '__main__':
    unittest.main()
