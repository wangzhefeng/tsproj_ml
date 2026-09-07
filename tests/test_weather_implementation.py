"""在隔离源码副本修改实现，真实计算生成器/设计/OOF 身份；不 mock 哈希链。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class WeatherImplementationTest(unittest.TestCase):
    def test_copied_implementation_change_invalidates_design_and_oof_identity(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(repo / 'data_loading', root / 'data_loading', ignore=shutil.ignore_patterns('__pycache__'))
            script = '''
import json, sys
from pathlib import Path
sys.path.extend([sys.argv[1]+'/tests', sys.argv[1]])
from test_weather_compiler import config_fixture
from data_loading import SourceRegistry
from data_loading.weather_generator.generator import weather_implementation_hash
from feature_engineering.cache import compute_raw_design_fingerprint, cache_dir
from model_ensemble.cache import member_source_hashes
root = Path.cwd()
data = root/'assets'
data.mkdir()
config = config_fixture(data)
registry = SourceRegistry(config.data, data)
import data_loading.weather_generator.generator as module
assert Path(module.__file__).is_relative_to(root)
def identity():
    return {'implementation': weather_implementation_hash(), 'design': compute_raw_design_fingerprint(config, base_dir=data, origin='2026-01-01T00:00Z', generators=registry.generators), 'oof': member_source_hashes('member', config.data, data, registry.generators)}
before = identity()
path = root/'data_loading/weather_generator/resampling.py'
path.write_text(path.read_text()+'\\n# isolated implementation revision\\n')
after = identity()
assert before['implementation'] != after['implementation']
assert before['design'] != after['design']
assert before['oof'] != after['oof']
assert cache_dir(root, before['design']) != cache_dir(root, after['design'])
assert after == identity()
print(json.dumps({'changed': True, 'stable_repeat': True}))
'''
            result = subprocess.run([sys.executable, '-c', script, str(repo)], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {'changed': True, 'stable_repeat': True})
