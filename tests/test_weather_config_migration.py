"""迁移审计：file 两段制 + inference_columns 合同校验；generator 源与缺映射必须 blocked。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from test_weather_compiler import config_fixture


class WeatherMigrationAuditTest(unittest.TestCase):
    def test_generator_and_unmapped_sources_remain_blocked(self):
        from scripts.audit_weather_configs import audit_weather_configs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_fixture(root)
            path = root / 'model.yaml'
            # file 化但砍掉 inference_columns → known_future 列无映射，必须 blocked
            legacy = yaml.safe_load(path.read_text())
            source = legacy['data']['sources'][1]
            source.update(source_type='file', future_path='weather.csv', availability='forecast_origin')
            source.pop('generator')
            source.pop('generator_options')
            legacy_path = root / 'legacy_weather.yaml'
            legacy_path.write_text(yaml.safe_dump(legacy))
            before = {p: p.read_bytes() for p in (path, legacy_path)}
            report = audit_weather_configs(root, repository_root=root)
            self.assertEqual(report['model_count'], 2)
            self.assertEqual(report['weather_config_count'], 2)
            self.assertEqual(report['blocked_count'], 2)
            reasons = {reason for row in report['configs'] for reason in row['blockers']}
            self.assertIn('known_future_column_without_inference_mapping', reasons)  # file 化但缺映射
            self.assertIn('generator_source_retired_from_active_configs', reasons)  # generator 源退出活动配置
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            destination = root / 'report.json'
            result = subprocess.run([sys.executable, 'scripts/audit_weather_configs.py', '--root', str(root), '--report', str(destination)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(destination.read_text())['blocked_count'], 2)

    def test_mapped_file_source_has_no_blockers(self):
        from scripts.audit_weather_configs import audit_weather_configs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_fixture(root)
            path = root / 'model.yaml'
            payload = yaml.safe_load(path.read_text())
            source = payload['data']['sources'][1]
            source.update(
                source_type='file',
                history_path='weather_history.csv',
                future_path='weather.csv',
                availability='forecast_origin',
            )
            source.pop('generator')
            source.pop('generator_options')
            # 每个模型面列补 inference_columns 映射 + pred 物理 ignored 声明
            known = [c for c in source['columns'] if c.get('role') == 'known_future']
            source['columns'] = known + [
                {'name': f'pred_{c["name"]}', 'role': 'ignored', 'categorical': False} for c in known
            ]
            source['inference_columns'] = {c['name']: f'pred_{c["name"]}' for c in known}
            path.write_text(yaml.safe_dump(payload))
            report = audit_weather_configs(root, repository_root=root)
            self.assertEqual(report['blocked_count'], 0)

    def test_empty_root_is_not_success(self):
        from scripts.audit_weather_configs import audit_weather_configs
        with tempfile.TemporaryDirectory() as directory:
            report = audit_weather_configs(directory, repository_root=directory)
            self.assertFalse(report['complete'])
            self.assertEqual(report['model_count'], 0)
