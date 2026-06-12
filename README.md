# tsproj-ml

基于机器学习模型（LightGBM / XGBoost / CatBoost）的时间序列预测项目，支持单变量/多变量、多步直接/递归/混合递归预测。

项目核心目标：
- 用统一的数据与特征工程流程，支持多种预测策略快速切换。
- 同时支持离线测试（滑窗验证）和未来区间预测（forecast）。
- 支持日期特征、天气特征、滞后特征和可选高级统计特征。

## 1. 项目能力概览

当前实现的预测方法（`pred_method`）共 7 种：

1. `univariate-single-multistep-direct-output` (USMDO)
2. `univariate-single-multistep-direct` (USMD)
3. `univariate-single-multistep-recursive` (USMR)
4. `univariate-single-multistep-direct-recursive` (USMDR)
5. `multivariate-single-multistep-direct` (MSMD)
6. `multivariate-single-multistep-recursive` (MSMR)
7. `multivariate-single-multistep-direct-recursive` (MSMDR)

支持模型：
- LightGBM（默认）
- XGBoost
- CatBoost
- 可选集成（stacking/averaging/weighted/blending）

## 2. 项目结构

```text
tsproj_ml/
├── main.py                      # 本地开发入口（VSCode 直接运行）
├── run.py                       # CLI 入口（支持 config + 参数覆盖）
├── pyproject.toml               # 项目依赖定义（uv）
├── config/
│   ├── model_config.py          # 通用配置
│   └── model_config_*           # 不同模型/策略配置变体
├── data_provider/
│   ├── data_loader.py           # 数据加载与历史/未来时间轴对齐
│   └── outlier_process.py       # 异常值处理工具
├── features/
│   ├── FeatureEngineering.py    # 外生/内生/高级特征工程
│   ├── FeatureScalering.py      # 数值缩放与类别特征处理
│   ├── FeatureSelection.py      # 特征选择工具
│   └── DataAugment.py           # 数据增强占位
├── models/
│   ├── ModelFactory.py          # 模型工厂
│   ├── ModelTraining.py         # 训练与可选调参
│   ├── ModelTesting.py          # 滑窗评估与指标计算
│   ├── ModelForecasting.py      # 多策略预测逻辑
│   ├── ModelEnsemble*.py        # 集成模型逻辑
│   └── ModelSaveLoad.py         # 模型保存/加载
├── utils/
│   └── log_util.py              # 日志工具
└── saved_results/               # 输出目录（模型、测试、预测结果）
```

## 3. 数据要求

在配置中由以下参数定义输入：
- `data_dir`: 数据目录
- `data_path`: 目标序列文件
- `target_ts_feat`: 时间列名
- `target`: 预测目标列名
- `freq`: 时间频率，使用 pandas 固定频率字符串，例如 `5min`、`15min`、`1h`、`1D`

可选外生数据：
- 日期类型数据：`date_history_path`, `date_future_path`
- 气象数据：`weather_history_path`, `weather_future_path`

## 4. 核心流程

`main.py` 中主流程：

1. 读取配置（`config/model_config.py` 或变体）
2. 加载并对齐历史/未来数据（`DataLoader`）
3. 特征工程（`FeatureEngineer`）
4. 训练集特征/目标拆分
5. 可选滑窗测试（`is_testing=True`）
6. 可选未来预测（`is_forecasting=True`）
7. 输出结果到 `saved_results/`

## 5. 快速开始（推荐使用 uv）

### 5.1 安装依赖并创建虚拟环境

```bash
uv sync
```

> `uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 同步当前项目环境。

### 5.2 运行方式

方式 A：本地开发（VSCode）

```bash
uv run python main.py
```

方式 B：CLI 运行（基于配置模块）

```bash
uv run python run.py \
  --config-module config.model_config_lgbm_usmd_A \
  --config-class ModelConfig_univariate \
  --model-type lightgbm \
  --pred-method univariate-single-multistep-direct \
  --is-testing 1 \
  --is-forecasting 0 \
  --now-time 2025-12-27T00:00:00
```

方式 C：脚本运行（模型测试）

```bash
bash scripts/run_model_test.sh
```

切换模型/配置示例：

```bash
CONFIG_MODULE=config.model_config_xgb_usmd_A MODEL_TYPE=xgboost bash scripts/run_model_test.sh
```

## 6. 关键配置说明（`config/model_config.py`）

重点字段：
- 时间与窗口：`history_days`, `predict_days`, `window_days`
- 预测策略：`pred_method`
- 模型类型：`model_type`（`lightgbm`/`xgboost`/`catboost`）
- 特征开关：
  - `enable_datetime_features`
  - `enable_date_features`
  - `enable_weather_features`
  - `enable_lags_features`
  - `enable_advanced_features`
- 训练模式：
  - `is_testing`
  - `is_forecasting`
- 滑窗测试训练集异常处理：
  - `enable_train_outlier_handling`：默认 `False`，开启后只清洗每个滑窗的训练段目标列。
  - `high_outlier_threshold` / `high_outlier_max_run_points`：识别短连续高值异常。
  - `drop_outlier_max_run_points` / `drop_rebound_min_abs_diff`：识别短时下探后快速回弹异常。

训练集异常处理不会修改测试段真实 `y`，因此 `test_scores_df.csv` 的指标仍基于原始测试目标。

示例（常用组合）：
- 只做滑窗验证：`is_testing=True`, `is_forecasting=False`
- 只做未来预测：`is_testing=False`, `is_forecasting=True`
- 都执行：`is_testing=True`, `is_forecasting=True`

## 7. 输出说明

默认输出目录：
- 模型：`saved_results/pretrained_models/<setting>/model.pkl`
- 测试：`saved_results/results_test/<setting>/`
  - `test_scores_df.csv`
  - `cv_plot_df.csv`
  - `train_outlier_report.csv`
  - `test_prediction.png`
- 预测：`saved_results/results_forecast/<setting>/`
  - `prediction.csv`
  - `prediction.png`

其中 `<setting>` 由 `model_type-data-pred_method` 组成。

## 8. 常见问题

1. `main.py` 和 `run.py` 有什么区别？
- `main.py`：本地开发入口，适合在 VSCode 里直接修改配置后运行。
- `run.py`：CLI 入口，适合脚本化运行和参数覆盖。

2. 如何切换模型？
- 在配置里改 `model_type` 为 `lightgbm`、`xgboost` 或 `catboost`。

3. 如何切换预测策略？
- 在配置里修改 `pred_method` 为上述 7 种之一。

4. 如何启用天气/日期特征？
- 打开相应开关，并配置好对应数据路径与时间列。

## 9. 版本与依赖

- Python: `>=3.10`
- 依赖以 `pyproject.toml` 和 `uv.lock` 为准。
- 推荐统一使用 `uv` 管理环境，避免 `pip` 与锁文件漂移。
