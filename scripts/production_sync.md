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

## 禁止直接回迁内容

- 生产 API main class
- `BaseModelMainClass`
- 算力数据预处理
- 生产输出字段
- 生产路径 import
- dataset/results 文件

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
