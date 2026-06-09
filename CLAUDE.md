# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 构建与运行

```bash
# 安装依赖并创建虚拟环境
uv sync

# 本地开发（VSCode — 直接修改 main.py 中的 config 导入）
uv run python main.py

# CLI 运行（配置 + 参数覆盖）
uv run python run.py \
  --config-module config.templates.univariate_config \
  --config-class ModelConfig \
  --model-type lightgbm \
  --pred-method univariate-single-multistep-direct \
  --is-testing 1 --is-forecasting 0 \
  --now-time 2025-12-27T00:00:00

# Shell 脚本封装（支持环境变量覆盖）
MODEL_TYPE=xgboost bash scripts/run_model_test.sh

# 模板脚本（完整参数化）
MODEL_TYPE=catboost PRED_METHOD=univariate-single-multistep-recursive bash scripts/template.sh
```

Python 版本锁定为 `3.10.11`（见 `pyproject.toml`）。

## 架构概览

基于树模型（LightGBM / XGBoost / CatBoost）的时间序列预测框架，支持 7 种预测方法和可配置的特征工程。

### 入口文件

- **`main.py`** — VSCode 开发入口。导入配置模块（`DEFAULT_CONFIG_MODULE`），实例化 `Model` 类，调用 `model.run()`。
- **`run.py`** — CLI 入口。按模块名+类名加载配置（如 `config.templates.univariate_config` + `ModelConfig`），应用 CLI 参数覆盖，然后委托给 `main.Model`。

### 核心流程（`Model.run()` in `main.py`）

1. `DataLoader.load_data()` — 加载所有原始数据
2. `DataLoader.process_history_data()` — 对齐历史时间轴
3. `FeatureEngineer.create_features()` — 生成日期时间/滞后/气象/高级特征，构建目标输出
4. 预测特征与目标输出特征分离
5. **可选：测试** — `Tester._window_test()` 逐滑动窗口执行（通过 ProcessPoolExecutor 并行）
6. **可选：预测** — `Trainer.train()` → `Forecaster._predict_by_method()` 对未来数据进行推理

### 关键目录

| 目录 | 职责 |
|---|---|
| `config/` | `@dataclass` 配置类。`templates/` 存放基础模板；`aidc_electricitys/` 存放按项目复制的配置（含真实数据路径和特征开关） |
| `data_provider/` | `DataLoader` — 加载并对齐历史/未来数据的时间轴，处理日期/气象外生输入 |
| `features/` | `FeatureEngineer`（滞后/外生/日期时间/高级特征生成）、`FeatureScaler`/`TargetScaler`（缩放）、`FeatureSelector`（特征选择）、`DataAugment`（数据增强） |
| `models/` | `ModelFactory`（工厂+抽象基类）、`ModelTraining`（训练+Optuna 调参）、`ModelTesting`（滑窗评估）、`ModelForecasting`（7 种预测方法）、`ModelEnsemble`（模型融合）、`ModelSaveLoad`（模型持久化） |
| `utils/` | `log_util.py`（带 TimedRotatingFileHandler 的日志器）、`frequency.py`（频率字符串 → 每日样本数） |

### 配置系统

配置采用 Python `@dataclass` 类（非 JSON/YAML）。命名规范：
- 配置文件：`config/<project>/model_config_<model>_<method>_<variant>.py`
- 类名：统一使用 `ModelConfig`（通过 `importlib` + `getattr` 加载）

切换配置模块：修改 `main.py` 中的 `DEFAULT_CONFIG_MODULE`，或通过 `run.py` 的 `--config-module` 参数传入。

### 预测方法（均在 `ModelForecasting.py` 中实现）

| 缩写 | 含义 |
|---|---|
| `USMDO` | 单变量多步直接输出预测 |
| `USMD` | 单变量多步直接预测 |
| `USMR` | 单变量多步递归预测 |
| `USMDR` | 单变量多步直接递归预测 |
| `MSMD` | 多变量多步直接预测 |
| `MSMR` | 多变量多步递归预测 |
| `MSMDR` | 多变量多步直接递归预测 |

单变量 = 仅使用目标序列自身的特征。多变量 = 使用目标序列 + 其他内生序列的特征。

### 日志系统

日志使用 `utils/log_util.py`。调用方在导入 logger 之前必须设置 `os.environ['LOG_NAME']` 为调用模块名（不含扩展名）。这会在 `logs/<LOG_NAME>/` 下创建按模块隔离的日志目录。日志级别通过环境变量 `SERVICE_LOG_LEVEL` 设置（默认 `INFO`）。

### 输出结构

```
saved_results/
  pretrained_models/<setting>/model.pkl
  results_test/<setting>/test_scores_df.csv, cv_plot_df.csv, test_prediction.png
  results_forecast/<setting>/prediction.csv, prediction.png
```

`<setting>` = `{model_type}-{pred_method_code}-{data_name}`。

### 重要注意事项

- 没有单元测试（`tests/` 已在 `.gitignore` 中忽略）。
- `.gitignore` 同时忽略了 `dataset/`、`saved_results/`、`logs/`、`data/`，以及以 `gaoweichao` 或 `aidc` 开头的配置目录。
- 添加新配置时，从 `config/templates/univariate_config.py`（单目标序列）或 `multivariate_config.py`（多内生序列）复制修改。
- 配置中的 `now_time` 是"当前"时间戳 — 历史数据 = `[now_time - history_days, now_time)`，未来区间 = `[now_time, now_time + predict_days]`。
- 配置使用条件字段赋值（`if enable_date_features: ... else: ...`），这虽然不常见但是有意为之 — 让 dataclass 类体保持自文档化。
- 模型训练使用分离的验证集，而非随机划分 — 时序交叉验证通过 `TimeSeriesSplit` 实现。
