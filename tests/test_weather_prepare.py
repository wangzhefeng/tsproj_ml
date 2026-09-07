"""prepare_weather 真 CLI：临时目录原样归档、幂等与拒绝覆盖。"""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest

from test_weather_assets import asset_fixture


class WeatherPrepareTest(unittest.TestCase):
    def test_register_historical_json_with_hashed_availability_csv(self):
        from data_loading.weather_generator.assets import WeatherAssetStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = asset_fixture(root, kind='reanalysis')
            source = root / 'response.json'
            source.write_text(json.dumps({'latitude':30., 'longitude':120., 'utc_offset_seconds':0, 'hourly_units':{'time':'iso8601','temperature_2m':'°C'}, 'hourly':{'time':['2026-01-01T01:00'],'temperature_2m':[10.]}}))
            releases = root / 'releases.csv'
            releases.write_text('time,available_at\n2026-01-01T01:00Z,2026-01-03T01:00Z\n')
            meta = manifest['snapshots'][0]
            meta['evidence_class'] = 'historical_release_contract'
            meta['raw'] += [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (source,releases)]
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps(meta))
            command = [sys.executable,'scripts/prepare_weather.py','register','--adapter','open_meteo','--input',str(source),'--metadata',str(metadata),'--availability-csv',str(releases),'--base-dir',str(root),'--output-root',str(root/'shared')]
            for flags in ([], ['--write'], ['--write']):
                result = subprocess.run(command+flags,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                if not flags:
                    self.assertFalse((root/'shared').exists())
            snapshot = WeatherAssetStore(root).load(json.loads(result.stdout)['reference'])[0]
            self.assertEqual(snapshot.frame.available_at.iloc[0].isoformat(),'2026-01-03T01:00:00+00:00')
            releases.write_text('tampered')
            result = subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)

    def test_register_vendor_snapshot_and_read_manifest(self):
        import hashlib
        from data_loading.weather_generator.assets import WeatherAssetStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = asset_fixture(root)
            source = root / 'raw.csv'
            source.write_text('ts,temperature_2m\n2026-01-01T01:00:00Z,10\n2026-01-01T02:00:00Z,12\n')
            meta = manifest['snapshots'][0]
            meta['raw'][0]['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps(meta))
            command = [sys.executable, 'scripts/prepare_weather.py', 'register', '--input', str(source), '--metadata', str(metadata), '--adapter', 'vendor', '--time-col', 'ts', '--base-dir', str(root), '--output-root', str(root / 'shared')]
            preview = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertFalse((root / 'shared').exists())
            completed = subprocess.run(command + ['--write'], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            reference = json.loads(completed.stdout)['reference']
            snapshot = WeatherAssetStore(root).load(reference)[0]
            self.assertEqual(snapshot.frame.value.tolist(), [10., 12.])
            self.assertTrue(all(str(root / 'shared') in str(root / r['path']) for r in snapshot.metadata['raw']))
            self.assertFalse(Path(reference['manifest']).is_absolute())
            self.assertTrue(all(not Path(r['path']).is_absolute() for r in snapshot.metadata['raw']))
            self.assertIn(metadata.read_bytes(), [(root / r['path']).read_bytes() for r in snapshot.metadata['raw']])
            indexes = list((root / 'shared/raw/by-source/synthetic/fixture-site/run-1').glob('*.json'))
            self.assertEqual(len(indexes),1)
            with tempfile.TemporaryDirectory() as restored:
                recovered = Path(restored)
                shutil.copytree(root / 'shared', recovered / 'shared')
                restored_snapshot = WeatherAssetStore(recovered).load(reference)[0]
                self.assertEqual(restored_snapshot.frame.value.tolist(), [10.,12.])

    def test_archive_dry_run_and_idempotent_real_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'vendor.csv'
            source.write_bytes(b'ts,unknown_unit\n2026-01-01,7\n')
            output = root / 'shared'
            command = [sys.executable, 'scripts/prepare_weather.py', 'archive', '--input', str(source), '--output-root', str(output), '--source-id', 'unverified-fixture']
            preview = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertFalse(output.exists())
            first = subprocess.run(command + ['--write'], capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            payload = json.loads(first.stdout)
            self.assertEqual(Path(payload['path']).read_bytes(), source.read_bytes())
            second = subprocess.run(command + ['--write'], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(second.stdout)['sha256'], payload['sha256'])
            self.assertTrue(source.exists())


if __name__ == '__main__':
    unittest.main()
