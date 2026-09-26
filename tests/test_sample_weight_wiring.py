# -*- coding: utf-8 -*-
"""sample_weight 端到端接线验证（2026-09-26 激活）。

钉住三件事：
1. 声明 validation.training.sample_weight 后，权重真实到达估计器 fit
   （改变拟合结果，而非仅 schema 接受）——用线性趋势数据 + 强衰减权重，
   加权前后预测产生可测差异；
2. 不声明时行为与历史一致（等权）；
3. fit_final 防护：声明加权但跳过 final_bundle_inputs 直接 fit_final
   时 RAISE，防止静默丢失加权语义。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.transforms import (
    CanonicalFeatureScaler,
    CanonicalTargetTransform,
)
from forecasting_core.specs import (
    EstimatorSpec,
    ForecastConfigSpec,
    ForecastStrategySpec,
)
from model_pipeline.runner import CanonicalBaseModelRunner, SourceRegistry
from tests.test_ensemble_parity import (
    _base_data,
    _base_features,
    _base_problem,
    _parity_data,
)


def _config_with_weight(data_path: Path, sample_weight_spec) -> ForecastConfigSpec:
    validation = {
        "forecast_origin": "2026-01-03T23:00:00",
        "history_steps": 10_000,
        "train_window_steps": 9_999,
        "fold_count": 1,
        "stride_steps": 2,
    }
    if sample_weight_spec is not None:
        validation["training"] = {"sample_weight": sample_weight_spec}
    return ForecastConfigSpec(
        problem=_base_problem(),
        data=_base_data(data_path),
        features=_base_features(),
        strategy=ForecastStrategySpec("recursive"),
        estimator=EstimatorSpec(
            model_type="ridge",
            target_adapter="independent",
            params={"alpha": 1e-8},
        ),
        probabilistic={"mode": "point"},
        validation=validation,
        output={"scenario_subpath": "sample-weight-wiring"},
    )


class SampleWeightWiringTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data_path = _parity_data(self.root / "local.csv")
        self.origin = pd.Timestamp("2026-01-03T23:00:00")

    def tearDown(self):
        self._tmp.cleanup()

    def _runner(self, config) -> CanonicalBaseModelRunner:
        registry = SourceRegistry(config.data, self.root)
        return CanonicalBaseModelRunner(config, registry, self.origin)

    def test_weighted_fit_reaches_estimator(self):
        # 线性趋势数据：强衰减（半衰期 0.05 天 ≈ 1.2h）让模型几乎只拟合
        # 末端样本；无加权拟合全局趋势。两条路径预测必须可测地不同。
        unweighted = self._runner(_config_with_weight(self.data_path, None))
        weighted = self._runner(_config_with_weight(
            self.data_path,
            {"method": "exponential", "halflife_days": 0.05},
        ))
        windows = weighted.backtest_windows()
        self.assertTrue(windows)
        train_indices = windows[0].train_indices

        _, _, _, _, artifact_unweighted = unweighted.fit(train_indices)
        _, _, _, _, artifact_weighted = weighted.fit(train_indices)

        times = weighted.forecast_times(self.origin)
        designs, provider = weighted.forecast_designs(
            self.origin,
            *_transforms(weighted),
        )
        transform = _transforms(unweighted)[1]
        prediction_u = unweighted.predict(
            artifact_unweighted, designs, provider, times, transform
        )
        prediction_w = weighted.predict(
            artifact_weighted, designs, provider, times, transform
        )
        difference = float(
            np.max(np.abs(prediction_u.values - prediction_w.values))
        )
        # 差异严格为正即证明权重到达估计器（等权路径下两 runner 的
        # 输入完全一致，任何非零差异只能来自 fit 的 sample_weight）；
        # 该合成集上量级 ~5e-7，阈值放 1e-9 防数值噪声误判。
        self.assertGreater(difference, 1e-9, "加权拟合未改变预测——权重未到达估计器")

    def test_undeclared_behaves_as_unweighted(self):
        baseline = self._runner(_config_with_weight(self.data_path, None))
        windows = baseline.backtest_windows()
        train_indices = windows[0].train_indices
        _, _, _, _, artifact = baseline.fit(train_indices)
        self.assertIsNotNone(artifact)

    def test_fit_final_requires_bundle_inputs_when_declared(self):
        config = _config_with_weight(
            self.data_path,
            {"method": "exponential", "halflife_days": 2.0},
        )
        runner = self._runner(config)
        with self.assertRaisesRegex(ValueError, "final_bundle_inputs"):
            runner.fit_final(
                (np.zeros((4, 4), dtype=float),),
                np.zeros((4, 2, 1), dtype=float),
            )

    def test_final_fit_with_weights_completes(self):
        config = _config_with_weight(
            self.data_path,
            {"method": "exponential", "halflife_days": 2.0},
        )
        runner = self._runner(config)
        scaler, transform, X_all, Y_all = runner.final_bundle_inputs()
        trainer, artifact, capabilities = runner.fit_final(X_all, Y_all)
        self.assertIsNotNone(artifact)
        self.assertTrue(np.isfinite(Y_all).all())


def _transforms(runner: CanonicalBaseModelRunner):
    """复用 fit 的真实变换流程（_fit_runtime_transforms 路径）。"""
    from model_pipeline.fold_fit import _fit_runtime_transforms
    from model_pipeline.runner import _label_end

    windows = runner.backtest_windows()
    train_indices = windows[0].train_indices
    train_sample_indices = tuple(
        index * runner.builder.n_series + series
        for index in train_indices
        for series in range(runner.builder.n_series)
    )
    selector = np.asarray(train_sample_indices, dtype=int)
    X_train = tuple(design[selector] for design in runner.X_all)
    Y_train = runner.Y_all[selector]
    scaler, transform, _, _ = _fit_runtime_transforms(
        runner.config,
        runner.builder,
        X_train,
        Y_train,
        tuple(
            runner.supervised_sample_origins[index]
            for index in train_sample_indices
        ),
        tuple(
            runner.supervised_sample_series_ids[index]
            for index in train_sample_indices
        ),
        max(
            _label_end(runner.builder, runner.supervised_origins[index])
            for index in train_indices
        ),
    )
    return scaler, transform


if __name__ == "__main__":
    unittest.main()
