# -*- coding: utf-8 -*-
"""活动 canonical YAML 必须使用生产 FeatureCompiler 的唯一 transformation grammar。"""

import unittest
from pathlib import Path
from typing import Any, cast

from config.config_loader import is_model_yaml, load_yaml_config
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    FixedStepBacktestSpec,
    ForecastConfigSpec,
)
from forecasting_core.specs.config import parse_model_config
from scripts.check_model_configs import check_model_yaml


ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_TRANSFORMATIONS = {
    "direct",
    "advanced",
    "feature_scaling",
    "target",
    "datetime_categorical",
    "interactions",
}
_ALLOWED_ADVANCED = {
    "rolling",
    "expanding",
    "difference",
    "percent_change",
    "time_since",
    "cyclical",
    "interaction",
    "polynomial",
}


class ActiveConfigRuntimeContractTest(unittest.TestCase):
    def test_all_base_configs_use_runtime_transformation_grammar(self):
        incompatible = []
        base_count = 0
        for path in sorted((ROOT / "config").rglob("*.yaml")):
            if not is_model_yaml(path):
                continue
            config = load_yaml_config(path)
            if not isinstance(config, ForecastConfigSpec):
                continue
            base_count += 1
            transformations = config.features.transformations
            unknown = sorted(set(transformations) - _ALLOWED_TRANSFORMATIONS)
            advanced = transformations.get("advanced", {})
            unknown_advanced = (
                sorted(set(advanced) - _ALLOWED_ADVANCED)
                if hasattr(advanced, "keys")
                else ["<not-a-mapping>"]
            )
            if unknown or unknown_advanced:
                incompatible.append(
                    (str(path.relative_to(ROOT)), unknown, unknown_advanced)
                )

        self.assertEqual(base_count, 821)
        self.assertEqual(
            incompatible,
            [],
            f"{len(incompatible)} active base configs diverge from runtime grammar; "
            f"first={incompatible[:3]}",
        )

    def test_fixed_step_configs_store_explicit_origin_step_geometry(self):
        cases = (
            (
                "config/aidc_load_15min_short/route_A/baseline/st_usmr_mean.yaml",
                (1536, 1424, 7, 16),
            ),
            (
                "config/aidc_load_15min_daily/route_A/baseline/enet_usmd_mean.yaml",
                (3264, 2784, 5, 96),
            ),
            (
                "config/aidc_electricity_computility/electricity_computility/"
                "add_training_inference_pod/lgbm_usmr_a.yaml",
                (8928, 4032, 17, 288),
            ),
            (
                "config/aidc_power_month/route_A/freq_1month/window_length_8/"
                "lgbm_usmd_prob_mean.yaml",
                (10, 7, 3, 1),
            ),
        )
        for relative, expected in cases:
            with self.subTest(config=relative):
                config = load_yaml_config(ROOT / relative)
                validation = config.validation
                self.assertIsInstance(validation.backtest, FixedStepBacktestSpec)
                geometry = cast(FixedStepBacktestSpec, validation.backtest)
                actual = (
                    geometry.history_steps,
                    geometry.train_window_steps,
                    geometry.fold_count,
                    geometry.stride_steps,
                )
                self.assertEqual(actual, expected)

    def test_calendar_month_config_uses_typed_day_month_geometry(self):
        path = (
            ROOT
            / "config/aidc_power_month/route_A/freq_1day/baseline/"
            / "lgbm_usmd_prob_mean_conformal.yaml"
        )
        config = load_yaml_config(path)
        geometry = config.validation.backtest
        self.assertIsInstance(geometry, CalendarMonthBacktestSpec)
        self.assertEqual(
            (
                geometry.train_window_days,
                geometry.fold_count,
                geometry.stride_months,
            ),
            (120, 6, 1),
        )

    def test_legacy_geometry_fields_fail_during_parse(self):
        path = (
            ROOT
            / "config/aidc_load_15min_short/route_A/baseline/st_usmr_mean.yaml"
        )
        config = load_yaml_config(path)
        payload = config.canonical_payload()
        validation = cast(dict[str, Any], payload["validation"])
        validation.update(
            {
                "history_length": 1536,
                "window_length": 1424,
                "max_test_windows": 7,
                "test_window_stride": 16,
            }
        )
        for key in ("history_steps", "train_window_steps", "fold_count", "stride_steps"):
            validation.pop(key, None)
        with self.assertRaisesRegex(ValueError, "Unknown fields"):
            parse_model_config(payload, source="<legacy-geometry-probe>")

    def test_unknown_runtime_sections_fail_during_parse(self):
        path = (
            ROOT
            / "config/aidc_load_15min_short/route_A/baseline/st_usmr_mean.yaml"
        )
        config = load_yaml_config(path)
        self.assertIsInstance(config, ForecastConfigSpec)
        for section in ("validation", "output", "probabilistic"):
            with self.subTest(section=section):
                payload = config.canonical_payload()
                section_payload = cast(dict[str, Any], payload[section])
                section_payload["totally_unknown_field"] = 1
                with self.assertRaisesRegex(ValueError, "Unknown fields"):
                    parse_model_config(payload, source=f"<{section}-probe>")

        nested_cases = (
            ("validation", "eval_mask"),
            ("output", "identity"),
            ("probabilistic", "conformal"),
        )
        for section, nested in nested_cases:
            with self.subTest(section=section, nested=nested):
                payload = config.canonical_payload()
                section_payload = cast(dict[str, Any], payload[section])
                section_payload[nested] = {"totally_unknown_field": 1}
                with self.assertRaisesRegex(ValueError, "Unknown fields"):
                    parse_model_config(payload, source=f"<{section}.{nested}-probe>")

    def test_checker_accepts_visible_target_history_transformations(self):
        path = (
            ROOT
            / "config/aidc_load_15min_short/route_A/baseline/enet_usmd_mean.yaml"
        )
        _, problems = check_model_yaml(str(path))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
