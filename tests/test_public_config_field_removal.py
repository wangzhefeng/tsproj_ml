# -*- coding: utf-8 -*-
"""公共配置不暴露内部固定值和无运行语义字段。"""

import unittest

from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from forecasting_core.specs.output import OutputSpec
from forecasting_core.specs.probabilistic import ProbabilisticConfigSpec


class PublicConfigFieldRemovalTest(unittest.TestCase):
    def test_output_rejects_setting_suffix_at_both_public_locations(self):
        cases = (
            {"setting_suffix": "-unused"},
            {"identity": {"scenario_subpath": "demo", "setting_suffix": "-unused"}},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "Unknown fields in output"):
                    OutputSpec.from_mapping(payload, source="inline.yaml")

    def test_yaml_probability_contract_rejects_internal_fixed_fields(self):
        base = {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }
        for field, value in (
            ("recursive_propagation", "median_path"),
            ("schema_version", 1),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Unknown fields in probabilistic"):
                    ProbabilisticConfigSpec.from_mapping(
                        {**base, field: value},
                        source="inline.yaml",
                    )

    def test_deployment_mapping_rejects_public_override_of_internal_fixed_fields(self):
        base = {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }
        for field, value in (
            ("recursive_propagation", "median_path"),
            ("schema_version", 1),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Unknown probabilistic key"):
                    probabilistic_spec_from_mapping({**base, field: value})

    def test_deployment_mapping_keeps_internal_fixed_values(self):
        spec = probabilistic_spec_from_mapping(
            {
                "mode": "quantile",
                "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5,
            }
        )

        self.assertEqual(spec.recursive_propagation, "median_path")
        self.assertEqual(spec.schema_version, 1)


if __name__ == "__main__":
    unittest.main()
