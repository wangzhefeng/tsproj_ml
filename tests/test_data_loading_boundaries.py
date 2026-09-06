"""数据层职责迁移：迁移前冻结值、读取/可见性、公开边界回归。"""
import json
import ast
import importlib.util
import pickle
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

import data_loading
from data_loading import BUILTIN_GENERATORS, InformationSetRequest, SourceRegistry
from feature_engineering import FeatureCompiler
from forecasting_core.specs import ColumnSpec, DataSourceSpec, DataSpec
import test_feature_visibility_compiler as fixtures
from test_package_layering import PROJECT_PACKAGES

GOLDEN = Path(__file__).parent / 'fixtures' / 'data_loading_materialization.json'


def snapshot_frame(frame):
    return {'csv': frame.to_csv(index=False), 'dtypes': [str(t) for t in frame.dtypes]}


def snapshot_info(info, request):
    payload: dict[str, Any] = {
        role: {name: snapshot_frame(frame) for name, frame in getattr(info, role).items()}
        for role in ('target_history', 'observed_past', 'known_future', 'static')
    }
    payload['lineage'] = [asdict(value) for value in info.lineage]
    payload['providers'] = {
        repr(identity): {
            'values': [provider.value_at('humidity', step) for step in range(request.H)],
            'methods': [provider.provider_name('humidity') for step in range(request.H)],
            'available_at': [str(provider.available_at('humidity', step)) for step in range(request.H)],
        }
        for identity, provider in info.observed_future_providers.items()
    }
    return payload


def frozen_cases():
    """合成输入仅为测试；期望文件在生产迁移前生成，不随测试更新。"""
    fixture = fixtures.FeatureVisibilityCompilerTest()
    fixture.setUp()
    try:
        result = {}
        for global_scope in (False, True):
            for method in ('persistence', 'provided_scenario'):
                fixture.write_fixture(global_scope=global_scope)
                config = fixture.build_config(global_scope=global_scope, observed_lags={'humidity': (1,)})
                sources = list(config.data.sources)
                sources[1] = replace(sources[1], provider=method, availability='column',
                                     available_at_col='issued_at', backtest_path='sensor_forecast.csv' if method != 'persistence' else None)
                targets = pd.read_csv(fixture.base_dir / 'targets.csv')
                labels = targets.groupby('site_id', sort=False).tail(2).copy() if global_scope else targets.tail(2).copy()
                labels['ts'] = pd.to_datetime(labels['ts']) + pd.Timedelta(hours=2)
                pd.concat([targets, labels], ignore_index=True).to_csv(fixture.base_dir / 'targets.csv', index=False)
                rows = []
                for identity, offset in ((('A', 0), ('B', 1000)) if global_scope else ((None, 0),)):
                    for hour in (4, 5):
                        for issued, adjustment in (('02:00', 0), ('03:00', 1), ('06:00', 999)):
                            rows.append({**({'site_id': identity} if global_scope else {}),
                                         'ts': f'2026-01-01 0{hour}:00', 'issued_at': f'2026-01-01 {issued}',
                                         'humidity': offset + hour * 100 + adjustment})
                fixture.write_csv('sensor_forecast.csv', rows)
                config = replace(config, data=DataSpec(tuple(sources)))
                for access in ('history_only', 'supervised_labels'):
                    request = fixture.request(global_scope=global_scope)
                    request = InformationSetRequest(request.forecast_origin, request.forecast_times,
                                                    request.series_ids, target_access=access)
                    info = SourceRegistry(config.data, fixture.base_dir).materialize(request)
                    payload = snapshot_info(info, request)
                    if access == 'history_only':
                        compiled = FeatureCompiler(config).compile(
                            info, request, observed_future_providers=info.observed_future_providers,
                        )
                        payload['compiled'] = snapshot_frame(compiled.frame)
                        payload['schema'] = list(compiled.schema.feature_names)
                        payload['proofs'] = json.loads(json.dumps([asdict(p) for p in compiled.visibility_proof], default=str))
                    result[f'global={global_scope},provider={method},access={access}'] = payload
        calendar = DataSourceSpec(
            name='calendar', source_type='generated', generator='chinese_holiday',
            columns=(ColumnSpec('is_holiday', 'known_future'), ColumnSpec('holiday_name', 'known_future', categorical=True),
                     ColumnSpec('next_holiday_days', 'known_future'), ColumnSpec('solar_term', 'known_future', categorical=True)),
            time_col='time', availability='generator_defined',
        )
        request = InformationSetRequest('2026-04-03', pd.date_range('2026-04-04', periods=9, freq='12h'), ())
        fixture.write_fixture()
        target = fixture.build_config().data.sources[0]
        result['calendar'] = snapshot_info(SourceRegistry(DataSpec((target, calendar)), fixture.base_dir,
                                                         generators=BUILTIN_GENERATORS).materialize(request), request)
        # 错误类型及消息同样冻结，防止检查顺序在迁移中变化。
        fixture.write_fixture()
        config = fixture.build_config()
        for case in ('duplicate_vintage', 'missing', 'nonfinite', 'empty'):
            fixture.write_fixture()
            path = fixture.base_dir / ('weather_future.csv' if case == 'duplicate_vintage' else 'targets.csv')
            frame = pd.read_csv(path)
            if case == 'duplicate_vintage':
                frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
            elif case == 'missing':
                frame.loc[0, 'load'] = float('nan')
            elif case == 'nonfinite':
                frame.loc[0, 'load'] = float('inf')
            else:
                frame = frame.iloc[:0]
            frame.to_csv(path, index=False)
            try:
                SourceRegistry(config.data, fixture.base_dir).materialize(fixture.request())
            except (ValueError, TypeError) as exc:
                result[case] = {'type': type(exc).__name__, 'message': str(exc)}
            else:
                raise AssertionError(f'{case} must fail')
        return json.loads(json.dumps(result, default=str).replace(str(fixture.base_dir), '<fixture>'))
    finally:
        fixture.tearDown()


def registry_private_accesses(source):
    """保守 AST 门禁：覆盖构造/注解/赋值别名、registry 属性链和 getattr。"""
    nodes = list(ast.walk(ast.parse(source)))
    constructors = {'SourceRegistry'}
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('data_loading'):
            constructors.update(alias.asname or alias.name for alias in node.names if alias.name == 'SourceRegistry')
    receivers = {'registry'}

    def constructor(node):
        return (isinstance(node, ast.Name) and node.id in constructors) or (
            isinstance(node, ast.Attribute) and node.attr == 'SourceRegistry'
        )

    def registry(node):
        return ast.unparse(node) in receivers or (
            isinstance(node, ast.Attribute) and node.attr == 'registry'
        ) or (isinstance(node, ast.Call) and constructor(node.func))

    for node in nodes:
        if isinstance(node, ast.arg) and node.annotation is not None and constructor(node.annotation):
            receivers.add(node.arg)
        elif isinstance(node, ast.AnnAssign) and constructor(node.annotation):
            receivers.add(ast.unparse(node.target))
    changed = True
    while changed:
        previous = len(receivers)
        for node in nodes:
            if isinstance(node, ast.Assign) and registry(node.value):
                receivers.update(ast.unparse(target) for target in node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None and registry(node.value):
                receivers.add(ast.unparse(node.target))
        changed = len(receivers) != previous
    violations = []
    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr.startswith('_') and registry(node.value):
            violations.append((node.lineno, node.attr))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in {'getattr', 'setattr', 'hasattr'} and len(node.args) >= 2
              and registry(node.args[0]) and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str) and node.args[1].value.startswith('_')):
            violations.append((node.lineno, node.args[1].value))
    return sorted(violations)


class DataLoadingBehaviorTest(unittest.TestCase):
    def test_registry_boundary_gate_rejects_aliases_and_attribute_chains(self):
        self.assertTrue(callable(globals().get('registry_private_accesses')))
        cases = (
            'from data_loading import SourceRegistry as R\nx = R(spec, root)\ny = x\ny._read_path("x")',
            'r = self.runner.registry\ngetattr(r, "_base_dir")',
            'from data_loading import SourceRegistry\ndef use(loader: SourceRegistry):\n    return loader._generators',
            'self.loader = data_loading.SourceRegistry(spec, root)\nself.loader._filter_identity(s, f, i)',
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertTrue(registry_private_accesses(source))
        self.assertEqual(registry_private_accesses('self.registry.base_dir\nself.registry.materialize(request)'), [])

    def test_production_has_no_registry_private_access(self):
        self.assertTrue(callable(globals().get('registry_private_accesses')))
        root = Path(__file__).resolve().parents[1]
        paths = list(root.glob('*.py'))
        for package in (PROJECT_PACKAGES - {'data_loading'}) | {'scripts', 'config'}:
            paths.extend((root / package).rglob('*.py'))
        violations = [(str(path.relative_to(root)), item) for path in paths
                      for item in registry_private_accesses(path.read_text())]
        self.assertEqual(violations, [])

    def test_discovery_reports_facts_and_context_is_read_only(self):
        fixture = fixtures.FeatureVisibilityCompilerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.write_fixture(global_scope=True)
        registry = SourceRegistry(fixture.build_config(global_scope=True).data, fixture.base_dir,
                                  generators=BUILTIN_GENERATORS)
        self.assertTrue(callable(getattr(registry, 'target_history_coverage', None)))
        coverage = registry.target_history_coverage()
        self.assertEqual(len(coverage), 1)
        self.assertEqual(coverage[0].series_ids, ('A', 'B'))
        self.assertTrue(coverage[0].times.equals(pd.date_range('2026-01-01 01:00', periods=3, freq='1h')))
        self.assertTrue(coverage[0].times_by_series['B'].equals(coverage[0].times))
        coverage[0].times_by_series.clear()
        self.assertEqual(set(registry.target_history_coverage()[0].times_by_series), {'A', 'B'})
        self.assertEqual(registry.latest_target_time(), pd.Timestamp('2026-01-01 03:00'))
        self.assertEqual(registry.base_dir, fixture.base_dir)
        with self.assertRaises(AttributeError):
            registry.base_dir = Path('elsewhere')
        registry.generators.clear()
        self.assertEqual(registry.generators, BUILTIN_GENERATORS)

    def test_calendar_registration_and_legacy_provider_paths(self):
        self.assertIsNotNone(importlib.util.find_spec('data_loading.information.indexing'))
        self.assertEqual(data_loading.chinese_holiday_frame.__module__, 'data_loading.calendar_generator.calendar_features')
        self.assertEqual(BUILTIN_GENERATORS['chinese_holiday'].__module__, 'data_loading.calendar_generator.chinese_holiday')
        generators = importlib.import_module('data_loading.calendar_generator')
        self.assertIs(generators.BUILTIN_GENERATORS, BUILTIN_GENERATORS)
        legacy = json.loads((GOLDEN.parent / 'data_loading_legacy_provider.json').read_text())
        with self.assertRaises((AttributeError, ModuleNotFoundError)):
            pickle.loads(bytes.fromhex(legacy['pickle_hex']))

    def test_reading_rules_have_independent_owners(self):
        for name in ('sources.source_io', 'processing.validation', 'processing.visibility', 'processing.alignment'):
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.util.find_spec(f'data_loading.{name}'))

    def test_frozen_materialization_and_compilation(self):
        self.assertEqual(frozen_cases(), json.loads(GOLDEN.read_text()))

    def test_reads_and_defensive_frames_are_registry_local(self):
        fixture = fixtures.FeatureVisibilityCompilerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.write_fixture()
        config = fixture.build_config()
        calls = []

        def reader(path):
            calls.append(path)
            return pd.read_csv(path)

        registry = SourceRegistry(config.data, fixture.base_dir, reader=reader)
        first = registry.materialize(fixture.request())
        expected = snapshot_info(first, fixture.request())
        frame = first.target_history['targets']
        frame.iloc[0, 1] = -999
        second = registry.materialize(fixture.request())
        self.assertEqual(snapshot_info(second, fixture.request()), expected)
        self.assertEqual(len(calls), 4)
        SourceRegistry(config.data, fixture.base_dir, reader=reader).materialize(fixture.request())
        self.assertEqual(len(calls), 8)

    def test_index_domain_duplicate_and_pickle_contract(self):
        fixture = fixtures.FeatureVisibilityCompilerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.write_fixture()
        config = fixture.build_config()
        sources = list(config.data.sources)
        sources[1] = replace(sources[1], provider='persistence')
        info = SourceRegistry(DataSpec(tuple(sources)), fixture.base_dir).materialize(fixture.request())
        frame = info.target_history['targets']
        info.register_row_position_lookup('target:targets', {'targets': frame}, 'ts', frame_name='targets')
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        info.register_row_position_lookup('observed:targets', {'targets': duplicate}, 'ts', frame_name='targets')
        _, lookup = info.row_position_lookup('observed:targets')
        self.assertEqual(lookup[pd.Timestamp(frame.iloc[0]['ts']).value], -1)
        old = info.row_position_lookup('target:targets')
        info.register_row_position_lookup('target:targets', {'targets': duplicate}, 'ts', frame_name='targets')
        self.assertIs(info.row_position_lookup('target:targets'), old)
        # 迁移前信息集本身不支持 unpickle；本次不得顺手修正这个既有行为。
        with self.assertRaisesRegex(AttributeError, 'MaterializedInformationSet is immutable'):
            pickle.loads(pickle.dumps(info))
        provider: Any = info.observed_future_providers[()]
        restored = pickle.loads(pickle.dumps(provider))
        self.assertEqual(restored.value_at('humidity', 0), provider.value_at('humidity', 0))
        self.assertEqual(restored.available_at('humidity', 0), provider.available_at('humidity', 0))


if __name__ == '__main__':
    unittest.main()
