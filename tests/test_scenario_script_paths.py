"""场景脚本下沉后，默认路径与独立 CLI 必须仍指向仓库根。"""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROOT_MODULES = (
    ('aidc_ess_selfuse_load.scripts.build_strategy_features', 'REPO_ROOT'),
    ('aidc_load_15min_daily.scripts.derive_load_state_features', 'PROJECT_ROOT'),
    ('aidc_load_15min_daily.scripts.load_event_analysis', 'PROJECT_ROOT'),
    ('aidc_load_15min_rolling.scripts.derive_load_state_features', 'PROJECT_ROOT'),
    ('aidc_load_15min_rolling.scripts.derive_multivariate_loads', 'PROJECT_ROOT'),
    ('aidc_load_15min_short.scripts.derive_load_state_features', 'PROJECT_ROOT'),
    ('aidc_load_15min_short.scripts.derive_multivariate_loads', 'PROJECT_ROOT'),
    ('aidc_load_month.scripts.load_event_analysis', 'PROJECT_ROOT'),
    ('aidc_power_month.scripts.derive_load_state_features', 'PROJECT_ROOT'),
    ('aidc_power_month.scripts.load_event_analysis_1day', 'PROJECT_ROOT'),
)


class ScenarioScriptPathsTest(unittest.TestCase):
    def test_relocated_scripts_resolve_project_root(self):
        for name, attribute in ROOT_MODULES:
            with self.subTest(module=name):
                module = importlib.import_module('config.' + name)
                self.assertEqual(getattr(module, attribute), ROOT)
        pipeline = importlib.import_module('config.aidc_ess_selfuse_load.scripts.strategy_features.pipeline')
        self.assertEqual(pipeline.DEFAULT_DATA_ROOT, ROOT / 'dataset/aidc_ess_selfuse_load')

    def test_cli_help_without_pythonpath_from_external_cwd(self):
        entries = (
            'aidc_ess_selfuse_load/scripts/build_strategy_features.py',
            'aidc_load_15min_daily/scripts/load_event_analysis.py',
            'aidc_load_month/scripts/load_event_analysis.py',
            'aidc_power_month/scripts/load_event_analysis_1day.py',
        )
        environment = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
        with tempfile.TemporaryDirectory() as directory:
            for entry in entries:
                with self.subTest(entry=entry):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / 'config' / entry), '--help'],
                        cwd=directory, env=environment, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_multivariate_default_inputs_are_read_from_real_dataset(self):
        for scenario in ('aidc_load_15min_short', 'aidc_load_15min_rolling'):
            with self.subTest(scenario=scenario):
                module = importlib.import_module(f'config.{scenario}.scripts.derive_multivariate_loads')
                frame = module.build_multivariate_loads()
                self.assertEqual(list(frame.columns), ['time', 'A_load', 'B_load'])
                self.assertGreater(len(frame), 0)
                self.assertEqual(module.DATA_DIR, ROOT / 'dataset' / scenario)
