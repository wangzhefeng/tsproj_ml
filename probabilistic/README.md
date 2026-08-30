# probabilistic 模块说明

`probabilistic/` 提供概率预测的语义契约、训练编排、后处理、校准与评估。当前 canonical 范围为 **point + 每目标边际 quantile `(N,H,K,Q)`**；CQR 与 joint samples 不在当前验收范围。

## 文件职责

| 文件 | 职责 |
|---|---|
| `types.py` | 强类型数据契约：`QuantileGrid`、`MarginalForecastDistribution`（point 与 marginal quantile 同轴校验、`dependence_model=None`）、schema-2 `ForecastModelBundle`（模型产物唯一载体，含 canonical problem/strategy/estimator/lineage/fingerprint） |
| `spec.py` | 概率配置归一化与严格校验：quantile grid 合法性、q50 锚点、recursive median path 等 |
| `objectives.py` | 模型类型到原生 quantile objective 的显式能力映射（LightGBM/XGBoost/CatBoost 等），不支持即 RAISE |
| `model_training.py` | 按 quantile grid 编排训练并构造 bundle；recursive quantile 只用 median path 推进 |
| `pipeline.py` | 预测后处理流水线（quantile 输出整形与恢复） |
| `postprocessing.py` | 分位数 crossing 诊断与 q50 锚定修复（`repair_quantile_crossing`），crossing 修复按 target 独立 |
| `metrics.py` | **已迁入 `model_evaluation/metrics.py`**（2026-08-30 evaluation 模块化） |
| `model_evaluation.py` | 从滑窗宽表生成概率预测长表、窗口指标和 horizon 指标（legacy 宽表工具，仅测试消费）；生产评估通路 `evaluate_marginal_distribution` 已迁入 `model_evaluation/marginal.py` |
| `calibration.py` | CQR 严格数学内核（nonconformity score、`calibrate_quantile_band`、`attach_cqr_interval_columns`）；**当前 canonical runtime 不执行**，保留为后续接线的实现基础 |

## 边界

- 边际 quantile 不是联合分布：metadata 恒为 `dependence_model=None`，samples 生成器 unsupported。
- CQR 输出（如启用时）是独立 prediction interval（`predict_pi*`），不回写 `predict_q*`；当前 canonical 结果不含 `predict_pi*`。
- bundle 唯一写入口是 schema-2 `ForecastModelBundle`；schema-1 `ProbabilisticModelBundle` 与 `BlendQuantileModel` 已删除。

## 验证

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python -m unittest \
  tests.test_probabilistic_types tests.test_probabilistic_contracts \
  tests.test_probabilistic_objectives tests.test_probabilistic_training \
  tests.test_probabilistic_pipeline tests.test_probabilistic_postprocessing \
  tests.test_probabilistic_metrics tests.test_probabilistic_evaluation \
  tests.test_probabilistic_prequential tests.test_multitarget_probabilistic \
  tests.test_conformal -v
```
