"""静态审计必须读取 generated weather 的传递资产，不能检查输出表头。"""
from pathlib import Path
import tempfile
import unittest

from scripts.audit_runtime_assets import audit_runtime_assets
from test_weather_compiler import config_fixture


class WeatherAuditTest(unittest.TestCase):
    def test_generated_weather_dependencies_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_fixture(root)
            report = audit_runtime_assets(root, repository_root=root)
            self.assertEqual(report['weather_source_count'], 1)
            self.assertEqual(report['weather_errors'], [])
            self.assertEqual(report['missing_declared_columns'], [])
            (root / 'raw.csv').write_text('corrupt fixture')
            report = audit_runtime_assets(root, repository_root=root)
            self.assertEqual(len(report['weather_errors']), 1)
            self.assertEqual(report['weather_errors'][0]['source'], 'weather')
            self.assertIn('hash', report['weather_errors'][0]['error'].lower())


if __name__ == '__main__':
    unittest.main()
