# -*- coding: utf-8 -*-
"""多步策略重构 Phase 0-3 的表驱动契约测试。"""

import unittest
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

import config.config_sections as config_sections
from models.multistep.contracts import (
    require_direct_output,
    require_dirrec_output,
    require_future_horizon,
    require_pointwise_output,
    require_recursive_output,
)
from models.multistep.executors import EXECUTOR_CATALOG
from models.multistep.resolve import resolve_strategy
from models.multistep.spec import InputScope, RolloutFamily, STRATEGY_SPECS
from models.multistep.state import RecursiveState
from models.ModelForecasting import Forecaster


METHOD_CASES = (
    ("univariate-single-multistep-direct-pointwise", "usmdp", InputScope.TARGET_ONLY, RolloutFamily.POINTWISE, (0,)),
    ("univariate-single-multistep-direct", "usmd", InputScope.TARGET_ONLY, RolloutFamily.DIRECT, (1, 2, 3, 4)),
    ("univariate-single-multistep-recursive", "usmr", InputScope.TARGET_ONLY, RolloutFamily.RECURSIVE, (0,)),
    ("univariate-single-multistep-direct-recursive", "usmdr", InputScope.TARGET_ONLY, RolloutFamily.DIRREC, (1, 2)),
    ("multivariate-single-multistep-direct", "msmd", InputScope.ALL_ENDOGENOUS, RolloutFamily.DIRECT, (1, 2, 3, 4)),
    ("multivariate-single-multistep-recursive", "msmr", InputScope.ALL_ENDOGENOUS, RolloutFamily.RECURSIVE, (0,)),
    ("multivariate-single-multistep-direct-recursive", "msmdr", InputScope.ALL_ENDOGENOUS, RolloutFamily.DIRREC, (1, 2)),
    ("univariate-single-multistep-blend-direct-recursive", "usbr", InputScope.TARGET_ONLY, RolloutFamily.BLEND, (1, 2, 3, 4, 0)),
    ("multivariate-single-multistep-blend-direct-recursive", "msbr", InputScope.ALL_ENDOGENOUS, RolloutFamily.BLEND, (1, 2, 3, 4, 0)),
)


def make_args(method, **overrides):
    values = dict(
        pred_method=method,
        lags=[2, 4],
        block_size=0,
        direct_strategy="multioutput",
        align_direct_features_to_target=False,
        use_horizon_exogenous_for_direct=False,
        endogenous_backfill_strategy="persistence",
        blend_weight_strategy="fixed",
        blend_weights=[0.5, 0.5],
        enable_feature_cache=False,
        enable_global_training=False,
        enable_lags_features=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class StrategyResolutionContractTest(unittest.TestCase):
    def test_config_method_codes_are_derived_from_strategy_catalog(self):
        self.assertFalse(hasattr(config_sections, "_PRED_METHODS"))
        self.assertEqual(
            config_sections.PRED_METHOD_CODE,
            {method: spec.code for method, spec in STRATEGY_SPECS.items()},
        )
        self.assertTrue(all(spec.description for spec in STRATEGY_SPECS.values()))

    def test_nine_external_methods_resolve_to_plan_table(self):
        for method, code, scope, rollout, label_steps in METHOD_CASES:
            with self.subTest(method=method):
                resolved = resolve_strategy(make_args(method), horizon=4, target_feature="y")
                self.assertEqual(resolved.spec.method, method)
                self.assertEqual(resolved.spec.code, code)
                self.assertEqual(resolved.spec.input_scope, scope)
                self.assertEqual(resolved.spec.rollout, rollout)
                self.assertEqual(resolved.target_plan.label_steps, label_steps)
                self.assertEqual(resolved.runtime_plan.forecast_length, 4)
                self.assertEqual(
                    resolved.training_plan.output_width,
                    resolved.runtime_plan.model_output_width,
                )

    def test_short_codes_remain_supported(self):
        for method, code, *_ in METHOD_CASES:
            with self.subTest(code=code):
                self.assertEqual(resolve_strategy(make_args(code), 4).spec.method, method)

    def test_dirrec_effective_block_controls_training_targets(self):
        resolved = resolve_strategy(make_args("usmdr", lags=[3, 9]), horizon=8)
        self.assertEqual(resolved.runtime_plan.block_size, 3)
        self.assertEqual(resolved.target_plan.column_names, ("y_shift_1", "y_shift_2", "y_shift_3"))
        self.assertEqual(resolved.training_plan.output_width, 3)

    def test_dirrec_empty_lags_defaults_to_one(self):
        resolved = resolve_strategy(make_args("msmdr", lags=[]), horizon=8)
        self.assertEqual(resolved.runtime_plan.block_size, 1)

    def test_dirrec_explicit_block_outside_horizon_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "block_size.*horizon"):
            resolve_strategy(make_args("usmdr", block_size=9), horizon=8)
        with self.assertRaisesRegex(ValueError, "block_size.*>= 0"):
            resolve_strategy(make_args("usmdr", block_size=-1), horizon=8)

    def test_catalog_has_one_executor_per_rollout_family(self):
        self.assertEqual(set(EXECUTOR_CATALOG), set(RolloutFamily))
        self.assertEqual(len(set(EXECUTOR_CATALOG.values())), 5)


class ShapeAndStateContractTest(unittest.TestCase):
    def test_execution_summary_has_stable_auditable_fields(self):
        from models.multistep import runtime

        formatter = cast(Any, getattr(runtime, "format_execution_summary", None))
        self.assertIsNotNone(formatter)
        self.assertEqual(
            formatter(
                "dirrec",
                model_calls=3,
                block_size=2,
                backfill_source="auxiliary",
            ),
            "multistep execution family=dirrec model_calls=3 "
            "backfill_source=auxiliary block_size=2",
        )

    def test_future_frame_must_equal_horizon(self):
        future = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=2, freq="h")})
        with self.assertRaisesRegex(ValueError, "future frame length mismatch"):
            require_future_horizon(future, horizon=3)

    def test_family_specific_shapes_are_strict(self):
        np.testing.assert_array_equal(require_pointwise_output([[1], [2]], 2), [1, 2])
        np.testing.assert_array_equal(require_direct_output([[1, 2]], 2), [1, 2])
        self.assertEqual(require_recursive_output([[1.5]]), 1.5)
        np.testing.assert_array_equal(require_dirrec_output([[1, 2]], 2), [1, 2])

        with self.assertRaisesRegex(ValueError, "pointwise"):
            require_pointwise_output([[1, 2]], 2)
        with self.assertRaisesRegex(ValueError, "direct"):
            require_direct_output([1], 2)
        with self.assertRaisesRegex(ValueError, "recursive"):
            require_recursive_output([1, 2])
        with self.assertRaisesRegex(ValueError, "dirrec"):
            require_dirrec_output([], 2)

    def test_recursive_state_does_not_modify_source_history(self):
        history = pd.DataFrame({"time": [1, 2], "y": [10.0, 20.0]})
        original = history.copy(deep=True)
        state = RecursiveState.from_history(history, max_length=2)
        state.append(pd.DataFrame({"time": [3], "y": [30.0]}))
        pd.testing.assert_frame_equal(history, original)
        self.assertEqual(state.history["y"].tolist(), [20.0, 30.0])


class ExecutorMigrationContractTest(unittest.TestCase):
    def test_forecaster_dispatches_through_catalog(self):
        forecaster = Forecaster.__new__(Forecaster)
        forecaster.args = make_args("usmd")
        forecaster.horizon = 2
        forecaster.target_feature = "y"
        forecaster.df_future = pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=2, freq="h")}
        )
        forecaster._quantile_outputs = None
        forecaster.log_prefix = "[test]"
        forecaster.model = object()
        forecaster.target_transform = None
        forecaster.target_decomposer = None
        forecaster._restore_target_transform = lambda values: np.asarray(values, dtype=float)
        forecaster.univariate_single_multi_step_direct_forecast = lambda: (_ for _ in ()).throw(
            AssertionError("legacy method dispatch must not be used")
        )

        class CatalogExecutor:
            def execute(self, context):
                self.context = context
                return np.asarray([1.0, 2.0])

        with patch.dict(EXECUTOR_CATALOG, {RolloutFamily.DIRECT: CatalogExecutor}):
            result = forecaster._predict_by_method()
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_dirrec_block_ranges_cover_horizon_without_empty_blocks(self):
        from models.multistep.executors import dirrec

        self.assertEqual(
            list(dirrec.iter_block_ranges(horizon=5, block_size=2)),
            [(0, 2), (2, 4), (4, 5)],
        )
        with self.assertRaisesRegex(ValueError, "block_size"):
            list(dirrec.iter_block_ranges(horizon=5, block_size=0))

        resolve_runtime_block_size = cast(
            Any,
            getattr(dirrec, "resolve_runtime_block_size", None),
        )
        self.assertIsNotNone(resolve_runtime_block_size)
        resolved = resolve_strategy(make_args("usmdr", lags=[2, 4]), horizon=5)
        self.assertEqual(resolve_runtime_block_size(SimpleNamespace(), resolved), 2)
        self.assertEqual(
            resolve_runtime_block_size(SimpleNamespace(model_output_width=5), resolved),
            5,
        )

    def test_blend_requires_two_exact_horizon_paths(self):
        from models.multistep.executors.blend import combine_exact_paths

        np.testing.assert_allclose(
            combine_exact_paths([1, 2], [3, 4], [0.25, 0.75], horizon=2),
            [2.5, 3.5],
        )
        with self.assertRaisesRegex(ValueError, "direct"):
            combine_exact_paths([1], [3, 4], [0.5, 0.5], horizon=2)
        with self.assertRaisesRegex(ValueError, "recursive"):
            combine_exact_paths([1, 2], [3], [0.5, 0.5], horizon=2)

    def test_blend_execution_preserves_shared_model_and_history(self):
        from models.multistep.executors.blend import BlendExecutor

        history = pd.DataFrame({"time": [1, 2], "y": [10.0, 20.0]})
        context = SimpleNamespace(
            horizon=2,
            model={"bundle_type": "blend_direct_recursive"},
            blend_direct_model="direct-model",
            blend_recursive_model="recursive-model",
            df_history_for_lags=history.copy(deep=True),
            _recursive_schema_cache={"existing": {"columns": ["x"]}},
            _quantile_outputs=None,
            args=make_args("usbr", predict_type="point"),
            _resolve_blend_weights=lambda: [0.5, 0.5],
        )
        original_model = context.model
        original_history = context.df_history_for_lags.copy(deep=True)
        original_schema = deepcopy(context._recursive_schema_cache)

        class Child:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=float)
                self._quantile_outputs = None

        children = iter((Child([1, 2]), Child([3, 4])))
        context._fork_for_model = lambda model: next(children)

        with patch("models.multistep.executors.blend.DirectExecutor.execute", return_value=np.asarray([1, 2])), \
             patch("models.multistep.executors.blend.RecursiveExecutor.execute", return_value=np.asarray([3, 4])):
            result = BlendExecutor().execute(context)

        np.testing.assert_allclose(result, [2, 3])
        self.assertIs(context.model, original_model)
        pd.testing.assert_frame_equal(context.df_history_for_lags, original_history)
        self.assertEqual(context._recursive_schema_cache, original_schema)


if __name__ == "__main__":
    unittest.main()
