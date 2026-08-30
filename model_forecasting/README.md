# forecasting 模块说明

`model_forecasting/` 是 canonical 预测架构的**核心合同 + 编排**层（2026-08-30 流水线阶段重排后收缩）：不可变 specs、统一预测张量、目标/特征变换、结果读写 schema、运行时编排。流水线各阶段已拆为顶层包：`data_loading/`（数据构造与组织）、`feature_engineering/`（特征编译）、`model_training/`（训练+策略+estimator 适配）、`model_testing/`（回测折与原语）、`model_evaluation/`（评估指标）、`prediction/`（推理执行）。权威设计文档：[`docs/multistep_forecasting_redesign.md`](../docs/multistep_forecasting_redesign.md)。

## 文件职责

| 文件 | 职责 |
|---|---|
| `specs/` | schema-2 配置语义类型（problem/data/feature/strategy/estimator/config），严格解析、未知字段 RAISE |
| `tensors.py` | 统一张量合同：point `(N,H,K)`、边际 quantile `(N,H,K,Q)`、samples `(N,S,H,K)` 类型边界；`require_matching_point_axes` 轴对齐校验 |
| `transforms.py` | 目标变换（calendar normalization → decomposition → scaling，严格逆序）+ `CanonicalFeatureScaler`；按 `(series_id,target)` 隔离、状态随 bundle 保存 |
| `results.py` | 结果读写：long schema 转换（`backtest_tensors_to_long`/`distribution_to_long`）、`write_backtest_results`/`write_forecast_results`、回测绘图、`CanonicalResultReader` |
| `runtime.py` | 唯一编排器：`run_canonical_config`（回测 → 最终训练 → 预测）+ `CanonicalBaseModelRunner`（ensemble 成员门面） |
| `forecaster.py` | `CanonicalForecaster`：推理合同校验 + 执行（严格 `(N,H,K)`；2026-08-30 自 prediction/ 收编——预测阶段的重量在 executor 与编排，薄壳不独立成包） |

## 边界

- 核心合同文件（specs/tensors/transforms/results）不得 import 任何阶段包或 ensemble；`runtime.py` 与 `forecaster.py` 是编排/执行豁免文件（仍不得 import ensemble）；
- 分层门禁：`tests/test_package_layering.py`（v2，13 项全包扫描）。
