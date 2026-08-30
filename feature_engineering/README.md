# feature_engineering 模块说明

`feature_engineering/` 是**特征工程层**（流水线第 2 阶段）：canonical 唯一特征通路。2026-08-30 自 `model_forecasting/features/` 迁为顶层包。

## 文件职责

| 文件 | 职责 |
|---|---|
| `compiler.py` | `FeatureCompiler`：列角色（target/observed_past/known_future/static/key/ignored）显式编译，lag/datetime/交互特征生成，**可见性证明**（每个特征在预测原点必须可得）+ lineage 审计 |
| `selection.py` | 监督特征选择（`features.selection` 配置，默认关闭）：SelectKBest（f_regression/mutual_info）+ force_keep；挂在训练 fit 边界逐窗重拟合（无泄漏），选中集写入 artifact.feature_schema，预测端按同名子集对齐 |

## 边界

- 依赖 `data_loading`（信息集）、`model_forecasting.specs/tensors/transforms`（合同与特征缩放器）；
- 实验性特征脚本（fft/wavelet/holiday 等）在 `docs/feature_engineering/` 归档，未接入主流程。
