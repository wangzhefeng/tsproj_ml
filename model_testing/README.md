# testing 模块说明

`model_testing/` 是**测试/回测层**（流水线第 4 阶段）：滚动回测的时间合同与公开原语。2026-08-30 自 `model_forecasting/`（validation.py + backtest.py）迁为顶层包。

## 文件职责

| 文件 | 职责 |
|---|---|
| `validation.py` | `TimeGeometry` + `rolling_origin_folds`：滚动滑窗折生成（单模型外层回测与 ensemble OOF 共享同一时间合同；扩展窗口 = window_length 取全长） |
| `backtest.py` | 回测公开原语：`seasonal_naive_tensor`（基线）、`actual_tensor`（holdout 实际值）、`resolve_origin`（预测原点解析）、`positive_validation_int` |

## 边界

- 只依赖 `model_forecasting.tensors`（合同）；
- 评估指标计算在 `model_evaluation/`（掩码 `build_eval_mask`、点评估、概率评估），本包只做时间/原语；
- OOF 特化折生成在 `model_ensemble/oof.py`（gap/cutoff/静默跳过差异为有意设计，见 redesign 文档声明）。
