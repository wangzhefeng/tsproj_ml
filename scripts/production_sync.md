# Production Sync

## 边界

`tsproj_ml` 是时间序列预测核心库，生产包只作为 adapter。核心算法、特征工程、评估、损失函数和通用数据质量逻辑应先在本仓库优化，再按需迁移到生产包。

生产包中的部署入口、平台父类、生产输入输出字段、算力数据预处理、dataset 和 results 文件不回迁到本仓库。

## 可迁移核心模块

- `models/ModelTesting.py`
- `models/ModelForecasting.py`
- `features/FeatureEngineering.py`
- `features/FeatureScalering.py`
- `models/losses.py`
- `models/learning_rate.py`
- `data_provider/outlier_handling.py`
- `utils/eval_mask.py`（MAPE 评估掩码与绘图掩码，被 `ModelTesting`/`ModelForecasting` 依赖）
- `utils/frequency.py`（采样频率解析，被 `FeatureEngineering` 依赖）

## 禁止直接回迁内容

- 生产 API main class
- `BaseModelMainClass`
- 算力数据预处理（`config/aidc_electricity_computility/electricity/2026-06-11/scripts/*`——属算力（AIDC）场景特定，不是通用数据质量逻辑，不回迁）
- 生产输出字段
- 生产路径 import
- dataset 和 results 文件

## 同步记录模板

```markdown
### YYYY-MM-DD change-id

- 变更摘要：
- 是否需要迁移项目 2：
- 项目 2 适配点：
- 验证结果：
```

## 当前同步项

### 2026-06-12 train-window-outlier-handling

- 变更摘要：将滑窗测试训练段目标异常处理抽成通用模块，支持短连续高值和短时下探回弹两类规则，并输出 `train_outlier_report.csv`。
- 是否需要迁移项目 2：需要时从本仓库核心模块同步；项目 2 已有相近逻辑。
- 项目 2 适配点：保持生产输入输出、算力预处理和部署父类不变，仅同步核心异常处理接口或 `ModelTesting` 调用点。
- 验证结果：以本仓库 `unittest` 和轻量真实配置测试结果为准。

### 2026-06-14 train-outlier-low-rise-rules

- 变更摘要：`TrainOutlierConfig` 扩展为高/低 × 绝对/相对 四象限——新增「绝对低值」(`low_outlier_threshold`/`low_outlier_max_run_points`，默认关) 和「骤升-回弹」(`rise_outlier_max_run_points`/`rise_rebound_min_abs_diff`，默认开) 两类规则；`outlier_handling.py` 新增 `_detect_rise_rebound_positions`。
- 是否需要迁移项目 2：需要时同步 `data_provider/outlier_handling.py` 与 `config/config_sections.py` 的 `TrainOutlierConfig`；新规则按需启用。
- 项目 2 适配点：保持生产输入输出、算力预处理、部署父类不变；注意 low 默认关、rise 默认开。
- 验证结果：本仓库 `tests/test_outlier_handling.py` 6 用例通过。

### 2026-06-14 eval-mask-configurable

- 变更摘要：`EvalMaskConfig`（mode/percentile/min_value/max_value）取代硬编码 P5 分位；MAPE 指标与绘图掩码统一到 `utils/eval_mask.build_eval_mask`，移除 `_build_relative_mape_mask`/`_build_plot_mask` 重复实现；`test_scores_df.csv` 新增 `MAPE Upper Threshold` 列；72 个 aidc electricity yaml 填入 eval_mask 默认值。
- 是否需要迁移项目 2：需要时同步 `utils/eval_mask.py`、`config/config_sections.py` 的 `EvalMaskConfig`，以及 `ModelTesting.py`/`ModelForecasting.py` 的 `build_eval_mask` 调用点；yaml 默认值按生产场景重新填写。
- 项目 2 适配点：保持生产输入输出、算力预处理、部署父类不变；仅替换 MAPE/掩码计算入口与对应配置节。
- 验证结果：以本仓库 `unittest` 与 aidc electricity 真实配置滑窗测试为准。

### 2026-06-15 feature-lags-and-drift-fix

- 变更摘要：`features/FeatureEngineering.py` 新增 `_filter_supported_lags`——丢弃超出当前样本数的 lag，避免 `shift(lag)` 产出全 NaN 列导致模型无声退化；修复 `pred_method in "univariate-single-multistep-direct-pointwise"` 字符串子串误匹配为 `==` 精确判断。
- 是否需要迁移项目 2：需要时同步 `features/FeatureEngineering.py` 的 lag 过滤逻辑与 `pred_method` 判断修复。
- 项目 2 适配点：保持生产输入输出、算力预处理、部署父类不变；仅同步特征工程的 lag 处理。
- 验证结果：以本仓库真实配置（含启用 lags 的 usmd 配置）滑窗测试为准。
