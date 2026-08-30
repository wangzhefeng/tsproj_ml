# models 模块说明

`models/` 提供 estimator factory 和模型持久化；canonical Trainer/Forecaster 实现位于 `model_training/trainer.py` / `model_forecasting/forecaster.py`（2026-08-29 架构收敛迁入 model_forecasting/，2026-08-30 流水线阶段重排迁为顶层阶段包）；source/feature/strategy/result 编排位于 `model_forecasting/runtime.py`。legacy 只读层已于 2026-08-29 全部删除。

## 文件职责

| 文件 | 职责 |
|---|---|
| `ModelFactory.py` | 统一创建 10 类回归器封装（LightGBM、XGBoost、CatBoost、RandomForest、HistGradientBoosting、Ridge、ElasticNet、Lasso、QuantileRegressor、SeasonalTemplate） |
| `model_training/trainer.py` | CanonicalTrainer / DirectMultiOutputRegressor / HorizonAlignedDirectRegressor |
| `model_forecasting/forecaster.py` | CanonicalForecaster，严格返回 `(N,H,K)` |
| `model_forecasting/runtime.py` | registry/compiler、滚动回测、变换、训练、预测和持久化编排 |
| `model_training/strategies/` | Recursive / Direct / MIMO / RecMO / DirRec / DirMO / DirRecMO 七个标准 executor |
| `model_forecasting/results.py` | canonical long CSV 读写；指标计算在 `model_evaluation/` |
| `ModelSaveLoad.py` | pickle 模型和目标缩放器保存/加载 |
| `learning_rate.py` | 固定/自动学习率解析 |
| `losses.py` | 模型损失名称和调参 scorer 推断 |

## Canonical 训练、回测与预测合同

1. `SourceRegistry` 按预测原点物化 target / observed-past / known-future / static；`FeatureCompiler` 同时生成 visibility proof 与固定 feature schema。
2. `CanonicalTrainer` 按 strategy 的标准 model count 和 dependency plan 训练；independent / regressor-chain / native adapter 必须通过显式 capability gate，禁止静默降级。
3. rolling backtest 保证 `training_label_end < holdout_label_start`；每个窗口写 long `cv_plot_df.csv`、per-target + aggregate `test_scores_df.csv` 和 source/holdout metadata；quantile 模式追加 `test_scores_probabilistic_df.csv`（tidy long：pinball/central 区间指标，eval_mask 口径与点评估一致）。
4. final forecast 复用同一 registry/compiler/transform 状态，写 long `prediction.csv`、`resolved_config.json`、`model.pkl`、`resolved_model.json`。
5. point shape 固定 `(N,H,K)`；quantile shape 固定 `(N,H,K,Q)`，recursive quantile 只用 median path 推进。joint samples 明确 unsupported。

canonical 结果目录：

```text
results/<scenario>/<result_identity>/
├── pretrained_models/model.pkl
├── pretrained_models/resolved_model.json
├── results_test/cv_plot_df.csv
├── results_test/test_scores_df.csv
├── results_test/test_scores_probabilistic_df.csv  # 仅 quantile 模式
├── results_test/result_metadata.json
├── results_forecast/prediction.csv
└── results_forecast/resolved_config.json
```

`prediction.csv` 的唯一键为 `(series_id,time,target)`；quantile 模式追加 `predict_q*`。canonical 当前不包含 CQR runtime，因此 canonical 结果当前不会写 `predict_pi*`。迁移配置中的 `probabilistic.conformal` 仅保留 legacy 意图。

## Legacy 清理结论

legacy 只读适配层（`LegacyArtifactAdapter`、`LegacyResultReader`）与迁移器已删除；`main.py` / `run.py` 只执行 canonical `ForecastConfigSpec`。

## 模型产物

canonical 新训练模型统一保存 schema-2 `ForecastModelBundle`：`pred_method=None`，必须包含 canonical problem、strategy 或 ensemble spec、estimator capabilities、固定 output order、series order、feature/source lineage、transform/scaler 状态和 config fingerprint。普通策略保存 `CanonicalStrategyArtifact`，融合保存 `CanonicalEnsembleArtifact`；quantile bundle 保存每个 level 的 canonical strategy artifact。

`ModelSaveLoad.py::load_model()` 只加载 schema-2 bundle 与 decomposition bundle；旧裸估计器/legacy dict pkl 不再被适配（加载结果原样返回，调用方自行判型）。

旧结果与 legacy 产物已废弃：`CanonicalResultReader` 只接受 canonical long schema。

## 与入口配置的关系

- `main.py` / `run.py` 只构造 `CanonicalModel`，且只接受 `ForecastConfigSpec`。
- canonical estimator 能力与参数由 `model_training/estimators/capabilities.py` 和 `ModelFactory.py` 解析。

## 验证

模型层已有 unittest 覆盖训练窗口异常报告和测试窗口边界。改动模型逻辑后至少运行：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run \
  python -m unittest discover -s tests -p "test_*.py"
```

最新精确测试/配置计数只在 [`docs/multistep_forecasting_redesign.md`](../docs/multistep_forecasting_redesign.md) 的实施完成记录维护。
