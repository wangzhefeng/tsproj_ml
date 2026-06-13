# tsproj-ml

基于机器学习回归器的时间序列多步预测项目。主链路围绕
LightGBM / XGBoost / CatBoost 构建，模型工厂层额外保留
RandomForest 支持；支持单变量/多变量、多步 direct、recursive、
direct-recursive 和 direct-output 预测。

本文档按当前代码和当前目录整理，不以 `docs/`、`logs/` 中的历史材料作为事实来源。

## 入口

推荐使用 `run.py` 作为正式 CLI 入口：

```bash
uv run python run.py \
  --config-module config.templates.univariate_config \
  --config-class ModelConfig \
  --model-type lightgbm \
  --pred-method univariate-single-multistep-direct \
  --is-testing 1 \
  --is-forecasting 0 \
  --now-time 2025-12-27T00:00:00
```

`main.py` 是轻量本地入口，只负责加载一个配置类并执行主流程：

```bash
uv run python main.py \
  --config-module config.templates.univariate_config \
  --config-class ModelConfig
```

配置加载边界：

- `config/config_loader.py` 只提供无副作用加载函数，导入时不解析命令行。
- `run.py` 负责完整 CLI 覆盖，包括模型、数据、时间窗口、开关和结果目录。
- 所有模板和当前生成配置的类名统一为 `ModelConfig`。

## 项目结构

```text
tsproj_ml/
├── main.py                 # Model 主流程和轻量直跑入口
├── run.py                  # 推荐 CLI 入口
├── config/                 # 配置模板、生成配置和配置加载器
├── dataset/                # 示例和项目数据
├── data_provider/          # CSV 读取、时间轴切片、异常处理
├── features/               # 特征工程、缩放、特征选择、数据增强
├── models/                 # 训练、测试、预测、融合和持久化
├── scripts/                # 配置生成、运行模板、维护脚本
├── tests/                  # unittest 回归测试
└── utils/                  # 日志、频率、运行环境辅助工具
```

运行产物默认写入 `saved_results/`，日志默认写入 `logs/`。

## 预测策略

| 缩写 | `pred_method` | 说明 |
|---|---|---|
| USMDO | `univariate-single-multistep-direct-output` | 单变量多步直接输出 |
| USMD | `univariate-single-multistep-direct` | 单变量多步 direct |
| USMR | `univariate-single-multistep-recursive` | 单变量递归预测 |
| USMDR | `univariate-single-multistep-direct-recursive` | 单变量分块 direct-recursive |
| MSMD | `multivariate-single-multistep-direct` | 多变量 direct 预测目标 |
| MSMR | `multivariate-single-multistep-recursive` | 多变量递归预测目标 |
| MSMDR | `multivariate-single-multistep-direct-recursive` | 多变量分块 direct-recursive |

主要能力：

- 点预测与分位数预测：`predict_type="point"` / `"quantile"`。
- 多输出训练策略：`multioutput`、`regressor_chain` 和 direct 专用逐 horizon 训练。
- 可选模型融合：`averaging`、`weighted`、`stacking`、`blending`。
- 可选特征选择、数据增强、目标/特征缩放、自动学习率和超参数搜索。
- 滑窗测试可启用训练段异常值局部修复，测试真实值不被清洗。

## 数据与配置

核心配置字段：

| 字段 | 含义 |
|---|---|
| `data_dir` | 数据目录，运行时转为 `Path` |
| `data_path` | 目标序列 CSV 文件名 |
| `target_ts_feat` | 目标序列时间列 |
| `target` | 目标值列，进入主链路后映射为 `y` |
| `freq` | pandas 固定频率，如 `5min`、`15min`、`1h` |
| `now_time` | 预测基准时间 |
| `history_days` | 历史窗口天数 |
| `predict_days` | 预测未来天数 |
| `window_days` | 滑窗测试窗口天数 |

可选外生输入：

- 日期类型：`df_date.csv`、`df_date_future.csv`
- 天气：`df_weather.csv`、`df_weather_future.csv`
- 其他内生变量：由 `target_series_numeric_features`、
  `target_series_categorical_features`、`target_series_drop_features` 控制。

更多映射规则见 `config/README.md` 和 `dataset/README.md`。

## 现有配置

当前主要配置树：

- `config/templates/`：单变量和多变量模板。
- `config/ETT-small/ETTm1`：21 个 ETTm1 配置。
- `config/aidc_electricity_computility/electricity/2026-06-11/`：6 个叶子数据集，
  每组 12 个配置。
- `config/aidc_electricity_computility/electricity/{2026-01-01,2026-01-07,2026-03-12}/`：
  每个日期包含 `route_A` / `route_B`，每组 12 个配置。
- `config/aidc_electricity_computility/electricity_computility/`：两个 7 策略配置目录。
- `config/aidc_electricity_computility/electricity/gaoweichao_compare`：4 个 CatBoost 单变量配置。

`run.py` 使用 `importlib.import_module()` 加载配置模块，因此 `--config-module`
必须传完整模块路径字符串。例如：

```bash
uv run python run.py \
  --config-module config.aidc_electricity_computility.electricity.2026-06-11.A1_201.model_config_lgbm_usmd_A1_201 \
  --config-class ModelConfig \
  --is-testing 1 \
  --is-forecasting 0
```

## 配置生成

`scripts/generate_configs.py` 根据一个 dataset 叶子目录自动检测目标文件、时间列、
频率、预测基准时间和外生文件，然后按模型和策略矩阵生成 Python 配置文件。

示例预览：

```bash
uv run python scripts/generate_configs.py \
  --dataset dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_201 \
  --target h_total_use \
  --models lightgbm,xgboost,catboost \
  --strategies usmdo,usmd,usmr,usmdr \
  --variant A1_201 \
  --config-dir config/aidc_electricity_computility/electricity/2026-06-11/A1_201 \
  --dry-run
```

确认输出路径后去掉 `--dry-run` 写入配置。

## 开发验证

```bash
uv run python -m py_compile main.py run.py config/config_loader.py scripts/generate_configs.py
uv run python -m unittest discover -s tests -p "test_*.py"
```

入口回归检查：

```bash
uv run python -c "import sys; sys.argv=['run.py','--config-module','config.templates.univariate_config','--config-class','ModelConfig','--model-type','lightgbm']; import main; print('imported')"
```
