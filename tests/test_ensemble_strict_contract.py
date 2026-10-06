"""融合配置必须在模型调用前拒绝未消费参数与歧义输入。"""
import copy
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock
from pathlib import Path

from config.config_loader import load_yaml_document
from model_ensemble.configuration.loader import load_raw_yaml, parse_ensemble_document, validate_member_sources
from model_ensemble.configuration.specs import EnsembleSpecError, MethodSpec
from test_ensemble_runtime import _ensemble_doc
from test_ensemble_runtime import EnsembleRuntimeTestBase, RUNTIME_SERVICES
import yaml
from test_weather_registry import weather_data


class EnsembleStrictContractTest(unittest.TestCase):
    def test_method_params_are_method_specific(self):
        for name, params in (("stacking", {"alpha": 999}), ("averaging", {"typo": True}),
                             ("linear_blending", {"metric": "mae"}), ("weighted", {"metric": "typo"})):
            with self.subTest(name=name), self.assertRaisesRegex(EnsembleSpecError, "params|metric"):
                MethodSpec(name, params)

    def test_unconsumed_top_level_options_are_rejected(self):
        variants = (("training", {"sample_weight": {"method": "exponential_decay", "halflife_days": 1}}),
                    ("train_outlier", {"method": "none"}), ("refit_every", 0))
        for key, value in variants:
            document = _ensemble_doc("averaging")
            document["validation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(EnsembleSpecError, key):
                parse_ensemble_document(document)
        document = _ensemble_doc("averaging")
        document["output"]["overlay"] = {"path": "unused.csv", "column": "y"}
        with self.assertRaisesRegex(EnsembleSpecError, "overlay"):
            parse_ensemble_document(document)

    def test_duplicate_keys_have_identical_strict_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("estimator:\n  params:\n    alpha: 1\n    alpha: 2\n")
            for load in (load_yaml_document, load_raw_yaml):
                with self.subTest(loader=load.__name__), self.assertRaisesRegex(ValueError, "Duplicate YAML key 'alpha'"):
                    load(path)

    def test_generated_source_contract_includes_generator_options(self):
        with tempfile.TemporaryDirectory() as directory:
            data = weather_data(Path(directory))
            document = _ensemble_doc("averaging")
            document["data"] = data.canonical_payload()
            config = parse_ensemble_document(document)
            members = {ref.name: {"data": copy.deepcopy(document["data"])} for ref in config.members}
            options = members[config.members[0].name]["data"]["sources"][1]["generator_options"]
            options["semantics_version"] = "weather_research_v1"
            options["research"] = {"release_delay": "8h", "rationale": "different source semantics"}
            with self.assertRaisesRegex(EnsembleSpecError, "identical.*subset"):
                validate_member_sources(config, members)

    def test_nonstring_member_name_is_not_silently_coerced(self):
        document = _ensemble_doc("averaging")
        document["ensemble"]["members"][0]["name"] = 123
        with self.assertRaisesRegex(EnsembleSpecError, "member.name"):
            parse_ensemble_document(document)


class EnsembleMemberPreflightTest(EnsembleRuntimeTestBase):
    def test_unused_member_training_option_fails_before_runner_construction(self):
        path = self.root / "member_recursive.yaml"
        document = yaml.safe_load(path.read_text())
        document["validation"]["training"] = {"blend_weight_windows": 2}
        path.write_text(yaml.safe_dump(document))
        factory = Mock(side_effect=AssertionError("runner must not be constructed"))
        with self.assertRaisesRegex(EnsembleSpecError, "blend_weight_windows"):
            self._run("averaging", services=replace(RUNTIME_SERVICES, runner_factory=factory))
        factory.assert_not_called()
