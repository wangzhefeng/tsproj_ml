# tsproj-ml

基于机器学习回归器的时间序列多步预测项目。当前主链路围绕 LightGBM / XGBoost / CatBoost 构建，同时在模型工厂层保留 RandomForest 支持；支持单变量/多变量、多步逐点 direct、direct、recursive 和 direct-recursive 分块预测。

本文档按当前代码和当前目录整理，未使用 `docs/` 和 `logs/` 目录中的历史材料作为事实来源。目录级细节可继续查阅 `config/README.md`、`dataset/README.md`、`data_provider/README.md`、`models/README.md`、`scripts/README.md` 和 `utils/README.md`。

## 当前能力

预测策略由 `pred_method` 控制：

| 缩写 | `pred_method` | 说明 |
|---|---|---|
| USMDO | `univariate-single-multistep-direct-pointwise` | 单变量，按未来时点逐点 direct 预测 |
| USMD | `univariate-single-multistep-direct` | 单变量，一个多输出模型直接预测整个 horizon |
| USMR | `univariate-single-multistep-recursive` | 单变量，一步模型递归回填 |
| USMDR | `univariate-single-multistep-direct-recursive` | 单变量，按块 direct，块间递归回填 |
| MSMD | `multivariate-single-multistep-direct` | 多变量，一个多输出模型直接预测目标 horizon |
| MSMR | `multivariate-single-multistep-recursive` | 多变量递归预测目标，其他内生变量默认持久性回填 |
| MSMDR | `multivariate-single-multistep-direct-recursive` | 多变量按块 direct，块间递归回填 |

主要能力：

- 树模型主链路：`lightgbm` / `xgboost` / `catboost`。
- 工厂层额外支持：`randomforest` / `rf`。
- 点预测与分位数预测：`predict_type="point"` 或 `"quantile"`。
- 多输出训练策略：`multioutput`、`regressor_chain`，以及 direct 专用的逐 horizon 训练器。
- 可选模型融合：`averaging`、`weighted`、`stacking`、`blending`。
- 可选数据增强、特征选择、目标/特征缩放、自动学习率、`RandomizedSearchCV` + `TimeSeriesSplit` 调参。
- 滑窗测试支持训练窗口异常值局部修复，测试真实值不被清洗。
- 滑窗测试中的 `MAPE Accuracy = 1 - MAPE` 按窗口内正样本 `P5` 相对阈值过滤后计算，近零异常点不再制造失真指标。
- 预测输出保存历史上下文和预测绘图拼接数据，便于生产问题定位。
- 未来预测图只对历史上下文复用 `P5` 相对阈值断线显示；未来预测值本身不做裁剪。

## 项目结构

```text
tsproj_ml/
├── main.py                      # 主流程入口，本地修改 YAML 后直接运行
├── run.py                       # 保留的兼容入口，不作为本地直跑推荐方式
├── pyproject.toml               # uv 依赖定义，Python == 3.10.11
├── config/
│   ├── config_sections.py       # 共享分组 dataclass 和预测策略说明
│   ├── config_loader.py         # 无 import-time argparse 副作用的配置加载器
│   ├── generate_configs.py      # 根据 dataset 自动检测字段并批量生成模型配置
│   ├── univariate_config.py     # ETTm1 单变量示例配置
│   ├── multivariate_config.py   # ETTm1 多变量示例配置
│   ├── aidc_electricity_computility/
│   └── ETT-small/
├── dataset/                     # ETT-small 和 AIDC electricity 数据
├── data_provider/
│   ├── data_loader.py           # 数据读取、时间轴构造、历史/未来切片
│   ├── outlier_handling.py      # 滑窗训练段异常处理
│   └── outlier_process.py       # 原始负荷数据异常标记、清洗与可视化
├── features/
│   ├── FeatureEngineering.py    # 日期时间、日期类型、天气、滞后、高级特征
│   ├── FeatureScalering.py      # 特征/目标缩放与逆变换
│   ├── FeatureSelection.py      # 训练集特征选择
│   └── DataAugment.py           # 训练集数据增强
├── models/
│   ├── ModelFactory.py          # 模型工厂和模型封装
│   ├── ModelTraining.py         # 训练、调参、分位数、融合
│   ├── ModelTesting.py          # 滑窗测试与指标保存
│   ├── ModelForecasting.py      # 7 种预测策略
│   ├── ModelEnsemble.py         # 融合模型
│   ├── ModelSaveLoad.py         # pickle 保存/加载
│   └── losses.py, learning_rate.py
├── scripts/                     # main.py 直跑脚本和维护说明
│   ├── production_sync.md       # 生产包同步边界、可迁移模块和同步记录
│   ├── run_model_test.sh        # main.py 直跑测试脚本
│   └── template.sh              # main.py 直跑模板脚本
├── tests/                       # unittest 回归测试
└── utils/
    ├── frequency.py             # pandas 频率解析
    ├── runtime_env.py           # 运行期环境变量辅助
    └── log_util.py              # 日志器
```

运行产物默认写入 `results/`，日志默认写入 `logs/`；这两个目录通常不作为源码事实来源。

## 核心流程

`main.Model.run()` 是当前主线：

1. `DataLoader.load_data()` 读取目标序列、日期外生、天气外生。
2. `DataLoader.process_history_data()` 根据 `now_time/history_days/freq` 构造历史时间轴，并把目标列统一映射为 `y`。
3. 单变量策略只保留 `time + y`；多变量策略保留配置中的其他内生变量。
4. `FeatureEngineer.create_features()` 生成外生、滞后和可选高级特征，并构造多步目标列。
5. 若 `is_testing=True`，`Tester._window_test()` 做滑窗验证，可按窗口并行。
6. 若 `is_forecasting=True`，`Trainer.train()` 用全历史训练，`Forecaster._predict_by_method()` 推理未来区间。
7. 测试、模型、预测结果分别保存到配置指定目录。

未来预测阶段只构造未来时间模板和未来外生特征，不读取未来真实目标值；递归类策略需要的目标值由预测结果逐步回填。

## 快速开始

安装依赖：

```bash
uv sync
```

本地直跑入口：

```bash
uv run python main.py
```

使用方式：

1. 打开 [main.py](/Users/wangzf/projects/tsproj_ml/main.py)。
2. 修改 `CONFIG_YAML`，指向要测试的本地 YAML 配置文件。
3. 如需调整数据路径、模型类型、预测策略、测试与预测开关，直接修改对应 YAML。
4. 运行 `uv run python main.py`。

`main.py` 不提供命令行参数覆盖；本地开发和测试统一采用“手动改 YAML + 直接运行”的方式。

## 配置生成

`config/generate_configs.py` 是批量生成分组 YAML 配置文件的工具，不是模型运行入口。它从一个 dataset 叶子目录读取目标 CSV，自动检测数据文件、时间列、频率、预测基准时间、内生特征列和外生文件，然后按模型 × 策略矩阵写入 `config/`。

YAML 按 `config_sections.py` 的分组语义保存覆盖项，例如 `target_series`、`exogenous_features`、`time_lag_features`、`model_strategy`。标准默认值由 YAML 内的 `base_config` 字段指定（通常为 `config.univariate_config` 或 `config.multivariate_config`）。

默认模型矩阵是 3 个树模型：

- `lightgbm` -> 文件名缩写 `lgbm`
- `xgboost` -> 文件名缩写 `xgb`
- `catboost` -> 文件名缩写 `cab`

默认单变量策略矩阵是 4 种方法：

- `usmdo`: `univariate-single-multistep-direct-pointwise`
- `usmd`: `univariate-single-multistep-direct`
- `usmr`: `univariate-single-multistep-recursive`
- `usmdr`: `univariate-single-multistep-direct-recursive`

脚本的关键输入：

| 参数 | 说明 |
|---|---|
| `--dataset` | 一个具体数据集叶子目录，目录内应包含目标 CSV |
| `--target` | 目标列名，例如 `h_total_use` |
| `--data-file` | 可选；不传时选目录中第一个非外生 CSV |
| `--target-ts-feat` | 可选；不传时使用目标 CSV 第一列 |
| `--models` | 逗号分隔模型列表，默认 `lightgbm,xgboost,catboost` |
| `--strategies` | 逗号分隔策略列表，默认 `usmdo,usmd,usmr,usmdr` |
| `--variant` | 输出子目录后缀，例如 `A1_201`、`A`、`B`；用于区分同数据集不同场景的配置目录，不写入文件名 |
| `--config-dir` | 输出目录；建议显式传入，避免默认路径和项目约定漂移 |
| `--base-config` | 可选；不传时按策略和内生特征自动选择 `config.univariate_config` 或 `config.multivariate_config` |
| `--dry-run` | 只打印将生成的文件，不写入磁盘 |

自动检测规则：

- 目标数据文件默认排除 `df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` 后取第一个 CSV。
- `target_ts_feat` 默认取目标 CSV 第一列。
- `freq` 优先用 `pandas.infer_freq` 推断，失败时按前两个时间戳间隔回退到 `1min`、`5min`、`15min`、`30min`、`1h`、`1D` 等固定频率。
- `now_time` 默认取目标 CSV 最后一行时间戳。
- 同时存在 `df_date.csv` 和 `df_date_future.csv` 时启用日期外生特征。
- 同时存在 `df_weather.csv` 和 `df_weather_future.csv` 时启用天气外生特征。
- 日期时间特征和滞后特征默认启用，分别可用 `--no-datetime`、`--no-lags` 关闭。

例如为 `2026-06-11` 的一个 demand load 数据集预览配置：

```bash
uv run python config/generate_configs.py \
  --dataset dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_201 \
  --target h_total_use \
  --models lightgbm,xgboost,catboost \
  --strategies usmdo,usmd,usmr,usmdr \
  --config-dir config/aidc_electricity_computility/electricity/2026-06-11/A1_201 \
  --dry-run
```

确认输出路径后去掉 `--dry-run` 即可生成实际配置。当前 `2026-06-11/demand_load` 已按 6 个叶子数据集生成了 72 个 YAML 配置文件：

| 数据集 | 配置目录 | 文件数 |
|---|---|---:|
| `A1_201` | `config/aidc_electricity_computility/electricity/2026-06-11/A1_201` | 12 |
| `A1_01a` | `config/aidc_electricity_computility/electricity/2026-06-11/A1_01a` | 12 |
| `A1_IT` | `config/aidc_electricity_computility/electricity/2026-06-11/A1_IT` | 12 |
| `A3_01e` | `config/aidc_electricity_computility/electricity/2026-06-11/A3_01e` | 12 |
| `AIDC/route_A` | `config/aidc_electricity_computility/electricity/2026-06-11/AIDC/route_A` | 12 |
| `AIDC/route_B` | `config/aidc_electricity_computility/electricity/2026-06-11/AIDC/route_B` | 12 |

这些配置的目标文件均为 `df_power.csv`，时间列为 `count_data_time`，目标列为 `h_total_use`，频率为 `5min`，`now_time` 为 `2026-06-11T23:55:00`。所有 YAML 均以 `config.univariate_config` 作为 `base_config`。

## 数据契约

核心配置字段：

| 字段 | 含义 |
|---|---|
| `data_dir` | 数据目录，会在运行时转为 `Path` |
| `data_path` | 目标序列 CSV 文件名 |
| `target_ts_feat` | 目标序列时间列 |
| `target` | 目标值列，进入主链路后映射为 `y` |
| `freq` | pandas 固定频率，如 `5min`、`15min`、`1h` |
| `now_time` | 预测基准时间 |
| `history_days` | 历史窗口天数 |
| `predict_days` | 预测未来天数 |
| `window_days` | 滑窗测试的训练+测试窗口天数 |

可选外生输入：

- 日期类型：`date_history_path`、`date_future_path`、`date_ts_feat`、`datetype_features`
- 天气：`weather_history_path`、`weather_future_path`、`weather_ts_feat`、`weather_features`
- 其他内生变量：由 `target_series_numeric_features` / `target_series_categorical_features` / `target_series_drop_features` 控制；历史处理时会自动排除时间列、目标列和 drop 列。

当前模板说明：

- `config/univariate_config.py` 和 `config/multivariate_config.py` 均默认指向 `dataset/ETT-small/ETTm1.csv`，目标列为 `OT`，使用 `dataset/ETT-small/ETTm1_exogenous/` 下的仿真外生数据。单变量策略共用同一数据源，区别在于只使用目标列自身历史。
- `config/multivariate_config.py` 默认指向 `dataset/ETT-small/ETTm1.csv`，目标列为 `OT`，多变量输入列为 `HUFL`、`HULL`、`MUFL`、`MULL`、`LUFL`、`LULL`。
- `config/multivariate_config.py` 默认启用 `dataset/ETT-small/ETTm1_exogenous/` 下的仿真日期类型和天气外生数据。
- 单变量默认 `pred_method` 为 `univariate-single-multistep-direct`；多变量默认 `pred_method` 为 `multivariate-single-multistep-direct-recursive`。

## 关键配置

运行模式：

- `is_testing`: 是否执行滑窗测试。
- `is_forecasting`: 是否训练全历史并预测未来。
- `max_test_windows`: 可选，只跑前 N 个测试窗口做快速验证。
- `test_window_stride`: 可选，滑窗抽样步长。

特征与预处理：

- `enable_datetime_features`
- `enable_date_features`
- `enable_weather_features`
- `enable_lags_features`
- `enable_advanced_features`
- `scale_features` / `feature_scaler_type`
- `scale_target` / `target_scaler_type` / `inverse_target`
- `encode_categorical_features`

训练增强：

- `enable_data_augmentation`
- `enable_feature_selection`
- `enable_auto_learning_rate`
- `perform_tuning`
- `enable_ensemble`
- `predict_type="quantile"` + `quantiles`
- `enable_global_training` + `series_id_feature`

并行与性能：

- `window_parallel_workers`
- `multi_output_n_jobs`
- `quantile_parallel_workers`
- `ensemble_parallel_workers`
- `model_thread_count`
- `block_size`

配置加载：

- `config/config_loader.py` 提供 `load_yaml_config(config_yaml)`，按 YAML 的 `base_config` 加载对应的 Python 模板并应用分组 `overrides`。
- `load_model_config()` 为内部工具函数，由 `load_yaml_config` 调用以实例化 `base_config` 指定的模板，不对外直接使用。
- 该加载器导入时不会解析 `sys.argv`；`main.py` 直接读取本地固定的 YAML 路径。

## 输出结构

`<setting>` 当前由 `{model_type}-{data_name}-{pred_method_code}` 组成，例如 `lightgbm-df_power-usmd`。

```text
results/
  pretrained_models/<setting>/
    model.pkl
    target_scaler.pkl              # 仅目标缩放器已 fit 时保存
  results_test/<setting>/
    test_scores_df.csv
    cv_plot_df.csv
    train_outlier_report.csv
    test_prediction.png            # 有有效预测列时保存
  results_forecast/<setting>/
    prediction.csv
    prediction.png
    history_context.csv
    prediction_plot_concat.csv
```

`prediction.csv` 基础列为 `time,predict_value`；分位数预测会额外输出 `predict_q10`、`predict_q50`、`predict_q90` 这类列，实际列由 `quantiles` 决定。

滑窗测试与未来预测图的可视化契约：

- `test_scores_df.csv` 额外记录 `MAPE Threshold`、`MAPE Valid Points`、`MAPE Excluded Points`、`MAPE Excluded Ratio`。
- `cv_plot_df.csv` 额外记录测试图的有效性标记；低于窗口正样本 `P5` 阈值的点在 `test_prediction.png` 中断线显示。
- `prediction_plot_concat.csv` 保留历史与未来拼接后的原始值 `raw_value`，并额外输出 `plot_value`、`plot_valid`、`series_type`；`prediction.png` 用 `plot_value` 绘制历史上下文，用原始 `predict_value` 绘制未来预测。

离线原始负荷异常处理由 `data_provider/outlier_process.py` 提供，输出默认保存到源数据目录：

```text
df_power_outlier_detection.csv
df_power_remove_outlier.csv
df_power_anomalies.png
```

## 常见边界

- 根 README 只做项目总览和主流程说明；配置映射、数据目录、目录职责请以对应子目录 README 为准。
- 标准默认值由 Python dataclass 模板提供；新增具体数据配置优先用分组 YAML 管理。
- `main.py` 是本地开发与测试的主入口；切换任务时直接修改 `CONFIG_YAML` 和对应 YAML 内容。
- `now_time` 会被规整到整点作为历史结束/预测开始，历史区间为 `[now_time - history_days, now_time)`，预测区间为 `[now_time, now_time + predict_days)`。
- 分位数预测训练多个子模型；点预测融合只在 `predict_type="point"` 时生效。
- 多变量递归类方法对非目标内生变量目前主要采用持久性回填，不等同于为每个内生变量单独建模。
- `.DS_Store`、`__pycache__/`、`logs/`、`results/` 不是源码或数据契约。

## 生产同步边界

`scripts/production_sync.md` 是生产包同步边界和同步记录文档，不是运行脚本。本仓库 `tsproj_ml` 是时间序列预测核心库，生产包只作为 adapter。核心算法、特征工程、评估、损失函数和通用数据质量逻辑应先在本仓库优化，再按需同步到生产包。

允许从本仓库向生产包迁移的核心模块包括：

- `models/ModelTesting.py`
- `models/ModelForecasting.py`
- `features/FeatureEngineering.py`
- `features/FeatureScalering.py`
- `models/losses.py`
- `models/learning_rate.py`
- `data_provider/outlier_handling.py`

禁止直接回迁或反向污染本仓库的内容包括：

- 生产 API main class
- `BaseModelMainClass`
- 算力数据预处理
- 生产输出字段
- 生产路径 import
- dataset 和 results 文件

维护方式：

- 涉及生产同步时先更新或查阅 `scripts/production_sync.md`，不要靠临时复制代码。
- 每次同步记录使用文档中的模板，说明变更摘要、是否需要迁移项目 2、项目 2 适配点和验证结果。
- 如果生产包中只有部署入口、平台父类、生产字段或路径适配变化，不应回迁到本仓库。
- 如果生产包中出现可复用的预测、评估、特征或数据质量逻辑，应先抽象成本仓库核心模块，再由生产包做最小 adapter 适配。

## 开发与维护注意事项

- 项目事实以当前代码为准；更新文档时不要把 `docs/` 或 `logs/` 中的历史内容当作当前实现。
- 运行入口、配置路径、输出目录和脚本说明容易漂移，修改前应重新核对 `main.py`、`config/`、`scripts/`。
- 配置字段、模型类型和预测策略名保持英文；中文用于解释语义和边界。
- 如果发现脚本、README、配置和代码不一致，优先按代码写清楚现状，不要为了文档一致性假定代码已经修好。
- 修改 README 时避免重复目录级 README 的全部细节；根 README 只保留全局流程、关键契约和常用命令。

## 日志

日志由 `utils/log_util.py` 创建。调用方通常先设置 `LOG_NAME`，日志写入 `logs/<LOG_NAME>/`；日志级别由 `SERVICE_LOG_LEVEL` 控制，默认 `INFO`。

`main.py` 会设置 `LOG_NAME=main`。脚本也可通过环境变量覆盖 `LOG_NAME`。

## 验证建议

文档更新后可用一致性搜索确认没有恢复旧入口模块名、旧配置类名或旧测试目录说明。

轻量入口运行检查：

```bash
uv run python main.py
```

静态和测试检查：

```bash
uv run python -m py_compile main.py run.py config/config_loader.py config/config_sections.py config/univariate_config.py config/multivariate_config.py config/generate_configs.py
uv run python -m unittest discover -s tests -p "test_*.py"
```

如果改了 shell 脚本，再运行：

```bash
bash -n scripts/run_model_test.sh scripts/template.sh
```

## 版本与依赖

- Python: `==3.10.11`
- 依赖来源：`pyproject.toml` 和 `uv.lock`
- 推荐使用 `uv sync` / `uv run`，避免手工 `pip install` 造成锁文件漂移。
