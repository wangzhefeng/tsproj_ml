# Config 说明

`config/` 保存模型运行配置、共享配置分组、YAML 配置、配置加载器和配置生成器。标准默认值由 Python `@dataclass`（模板模块）提供；具体数据和任务参数通过 YAML 覆盖。

## 目录结构

```text
config/
├── config_sections.py      # 共享配置分组 dataclass 定义（13 个分组 + BaseModelConfig）
├── config_loader.py        # YAML 配置加载器
├── generate_configs.py     # 从数据集目录批量生成分组 YAML 配置
├── univariate_config.py    # 单变量场景模板模块（ModelConfig 子类）
├── multivariate_config.py  # 多变量场景模板模块（ModelConfig 子类）
└── <dataset_scenario>/     # 各数据场景的 YAML 配置目录（模型配置 / 聚合 / 异常检测）
```

## 模板模块

`config_sections.py` 定义扁平组合配置基类 `BaseModelConfig`，由 13 个分组 dataclass 继承组合：

- `RuntimeConfig`：运行模式（测试/预测）、时间窗口（`history_length`、`predict_steps`、`window_length`、`now_time`）、跨度口径（`horizon_mode`、`train_window_length`）、`schedule_mode`（`daily`/`intraday`）和信息口径（`forecast_mode`）
- `TargetSeriesConfig`：目标序列数据路径、频率、时间列、目标列、内生数值/类别/剔除列
- `ExogenousFeatureConfig`：日期类型外生、气象历史/回测/未来三段信息集及来源声明、自定义外生注册表（`custom_features`）、日期时间派生特征
- `TimeLagFeatureConfig`：滞后特征开关与滞后步数
- `AdvancedFeatureConfig`：滚动/扩展窗口统计、差分、百分比变化、距事件、周期三角编码、交互、多项式特征
- `PreprocessingConfig`：特征/目标缩放、月总量按日历天数归一化、窗口内因果目标分解（`linear`/`stl`/`mstl`）、分组缩放、类别编码
- `ModelStrategyConfig`：模型类型、融合模式（含成员级 `ensemble_model_specs`）、预测方法、多输出策略、分位数、Direct 目标时点对齐、`direct_strategy`、内生回填策略、blend 权重策略
- `TrainingEnhancementConfig`：早停、时间衰减样本权重、超参调优、数据增强、特征选择、学习率策略
- `TrainOutlierConfig`：滑窗测试训练窗口异常清洗
- `EvalMaskConfig`：评估/绘图掩码（`percentile`/`absolute`/`combined`）
- `ConformalConfig`：分位数预测的 CQR conformal 校准（校准窗口、α、最少 score 数）
- `PerformanceConfig`：窗口/多输出/分位数/融合并行度、日志
- `OutputConfig`：结果输出目录与场景子路径、setting 后缀、测试图叠加参考序列

组合后的 `BaseModelConfig` 仍暴露扁平字段，运行代码、YAML 覆盖和 CLI 覆盖统一读写 `cfg.data_path`、`cfg.pred_method` 等属性。

`univariate_config.py` 和 `multivariate_config.py` 各定义 `ModelConfig(BaseModelConfig)` 子类，覆盖典型单变量/多变量数据集的默认值（数据路径、频率、外生文件、滞后步数、预测方法）。YAML 通过 `base_config` 指向模板模块，例如 `config.univariate_config` 或 `config.multivariate_config`。

频率相关的滞后步数由 `config_sections.default_lags_for_freq(freq, days=7)` 按标准频率→每日样本数映射生成（如 `15min` → `[96, 192, ..., 672]`），无法识别时回退 5min 基准。

## 加载规则

`config/config_loader.py` 提供 `load_yaml_config()`；`load_model_config()` 为内部工具函数，供 YAML 加载器读取 `base_config` 指定的 Python 模板。

加载流程：

1. 读取 YAML，取 `base_config`（必填，指向模板模块）、`overrides`（映射）。
2. 实例化模板模块中的 `ModelConfig`（`config_class` 不写入 YAML，加载器默认使用 `ModelConfig`）。
3. `overrides` 按 `config_sections.py` 的分组语义扁平化（兼容旧平铺写法），嵌套分组仅支持一层。
4. 逐个覆盖字段；叶子字段必须存在于模板中，未知字段报错，避免拼写错误静默失效。
5. 已废弃的 `start_time`/`future_time` 覆盖字段被静默忽略（运行时由 `now_time` 与窗口参数推导）。

YAML 基本结构：

```yaml
base_config: config.multivariate_config
overrides:
  target_series:
    data_dir: ./dataset/<scenario>/
    data_path: <file>.csv
    freq: 15min
    target_ts_feat: <time_col>
    target: <target_col>
  exogenous_features:
    enable_date_features: true
    enable_weather_features: true
  model_strategy:
    model_type: lightgbm
    pred_method: multivariate-single-multistep-recursive
```

## scenario_subpath（结果输出路径）

每个 YAML 配置通过 `overrides.output.scenario_subpath` 显式指定结果输出路径中的 `<scenario>` 段，使 `results/` 下的目录结构与 `config/` 对齐。多组配置共用同一 `data_dir` 时必须显式指定，否则结果会混入同一目录；未设置时由 `data_dir` 自动推导。

## 配置分组参考

YAML 的 `overrides` 按 `config_sections.py` 的 dataclass 分组，字段名即各分组内的属性名。所有字段均有 dataclass 默认值，YAML 只需写需要覆盖的字段。

### runtime：自然月动态 horizon

`horizon_mode: fixed_steps`（默认）保持 `horizon = predict_steps`。日频预测完整自然月时使用：

```yaml
runtime:
  horizon_mode: calendar_month
  train_window_length: 120
  schedule_mode: daily
```

`calendar_month` 当前仅支持 `freq: 1D`：forecast_start 必须为月初，最终 horizon 自动取目标月的 28/29/30/31 天，forecast_end 为下月月初；滑窗测试按完整自然月切分，每个 fold 使用自己的 horizon，同时固定最近 `train_window_length` 天作为训练段。setting 编码真实 `train_window_length` 并追加 `-calendar-month`（不再使用遗留 `window_length` 数字），与历史固定步长结果隔离。测试汇总额外包含 `Calendar Month`、`Forecast Steps`、月实际/预测总量和 `Monthly Total MAPE`。未来气象文件必须覆盖目标自然月最后一天。

### exogenous_features：严格气象信息集

月初预测完整自然月时，天气必须区分历史实测、回测 ex-ante 代理/预报和正式未来代理/预报：

```yaml
exogenous_features:
  strict_weather_information_set: true
  weather_history_source: actual
  weather_backtest_path: weather_daily_stats_backtest_proxy_*.csv
  weather_backtest_source: proxy
  weather_future_source: proxy
```

strict 模式要求 backtest 每行 `available_at` 早于目标月、future 每行不晚于 `now_time`，且目标时间戳全覆盖；不允许把测试月实测天气当未来天气。Direct/DirRec 使用天气时必须启用 `use_horizon_exogenous_for_direct: true`，训练侧 `*_h1..*_hH` 按目标日构造；horizon-feature melt 按 h 折叠回基础列名，与推理端未来逐日外生一致。USBR 的 Direct/Recursive 共享 X 暂不支持目标日天气，保留为无天气 control。

USMDP 当前不生成目标 lag，配置必须显式 `enable_lags_features: false`、`lags: []`，语义是日历/天气逐点模板；需要目标历史递归回填时使用 USMR。

### preprocessing：目标分解

目标分解必须遵守时间因果边界：滑窗测试在每个训练段内独立拟合，测试标签始终保留原始电平；最终预测才使用全部已知历史拟合。`decomposition_method` 是唯一入口：`none` 关闭，`linear`/`stl`/`mstl` 启用对应方法。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `decomposition_method` | `none` | `none` / `linear` / `stl` / `mstl` |
| `decomposition_periods` | `[]` | STL 一个周期；MSTL 至少两个周期；每个训练窗须覆盖至少两个完整周期 |
| `decomposition_robust` | `true` | STL robust 拟合开关 |
| `decomposition_trend_degree` | `1` | 趋势外推多项式阶数，只允许 1 或 2 |
| `decomposition_trend_forecast` | `polynomial` | `polynomial` / `damped` |
| `decomposition_damping` | `0.98` | damped 趋势的阻尼系数，范围 `(0, 1]` |
| `decomposition_trend_lookback` | `28` | damped 趋势末端斜率估计使用的最近样本数 |
| `decomposition_seasonal_cycles` | `4` | 未来季节模板取最近周期的相位均值 |

`scale_target`、USBR/MSBR blend、CQR conformal 暂不与目标分解组合。point 与全部 quantile 会加回同一确定性趋势/季节分量；启用分解的最终训练会额外保存 `target_decomposer.pkl`。

```yaml
  preprocessing:
    decomposition_method: mstl
    decomposition_periods: [96, 672]
    decomposition_trend_degree: 1
    decomposition_trend_forecast: polynomial
```

### training_enhancement

训练阶段的调参、增强与损失配置：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `patience` | int | `100` | 早停轮数 |
| `enable_time_decay_sample_weight` | bool | `false` | 时间衰减样本权重：近期样本权重更高，抑制概念漂移导致的远期噪声 |
| `decay_halflife_days` | float | `14.0` | 半衰期（天）：样本权重衰减到一半所需的样本年龄；仅在 `enable_time_decay_sample_weight: true` 时生效 |
| `perform_tuning` | bool | `false` | 是否执行超参数随机搜索 |
| `enable_data_augmentation` | bool | `false` | 是否启用数据增强 |
| `enable_feature_selection` | bool | `false` | 是否启用特征选择 |
| `enable_auto_learning_rate` | bool | `false` | 是否自动估算学习率 |

> 时间衰减权重兼容性：默认配置下（`multi_output_strategy: multioutput` + `predict_type: point` + `enable_ensemble: false`）所有 9 种预测方法均支持（多输出走项目自定义 `DirectMultiOutputRegressor`，逐输出转发 `sample_weight`）；`regressor_chain` 多输出、`ensemble`、多输出分位数路径不支持，开启时会 warning 跳过（不阻塞运行）。

启用示例：

```yaml
  training_enhancement:
    enable_time_decay_sample_weight: true
    decay_halflife_days: 14
```

其余分组（runtime、target_series、exogenous_features、time_lag_features、advanced_features、preprocessing、model_strategy、train_outlier、eval_mask、conformal、performance、output）的字段与默认值以 `config_sections.py` 为准。

新增字段速查（`config_sections.py` 为权威定义）：

| 分组 | 字段 | 说明 |
|---|---|---|
| `runtime` | `schedule_mode` | `daily`（默认，日界对齐预测下一完整自然日）/ `intraday`（保留调度时刻从 `now_time` 起预测） |
| `runtime` | `horizon_mode` | `fixed_steps`（默认，固定 `predict_steps`）/ `calendar_month`（当前仅 `1D + daily`，自动预测完整目标月并按自然月回测） |
| `runtime` | `train_window_length` | `calendar_month` 必填；每个自然月 fold 固定训练天数，与 28～31 天动态测试跨度解耦 |
| `exogenous_features` | `custom_features` | 自定义外生特征注册表；`future_strategy` 可设 `explicit` 或 `freeze_last_observation`。后者可配 `availability: end_of_period`：Direct/DirRec 保留原点状态且不做 horizon shift，Recursive/Pointwise 训练历史向后对齐 1 个 freq 步，future 继续冻结 cutoff 前最后状态 |
| `model_strategy` | `ensemble_model_specs` | 融合成员级规格（非空时取代 `ensemble_models`），每项 `{model, params?, scale?, impute?}` |
| `model_strategy` | `direct_strategy` | `multioutput`（默认，每 horizon 独立模型）/ `horizon_feature`（horizon 索引作特征，仅 USMD/MSMD，训练成本 H×→1×） |
| `model_strategy` | `endogenous_backfill_strategy` | MSMR/MSMDR 非目标内生变量回填：`persistence`（默认）/ `auxiliary`（独立递归辅助模型） |
| `model_strategy` | `blend_weight_strategy` | USBR/MSBR 融合权重：`fixed` / `ridge_stacking`（滑窗测试集上 Ridge 学权重） |
| `conformal` | `enable_conformal_calibration` 等 | CQR 分位数校准：`conformal_alpha` / `conformal_calibration_windows` / `conformal_min_scores` |
| `output` | `setting_suffix` | 结果目录 setting 后缀（如 `-intraday`），同配置不同语义版本结果隔离 |
| `output` | `plot_overlay_path` / `plot_overlay_col` | 测试图叠加参考序列（次坐标轴），路径相对 `data_dir` |

## XGBoost 参数速查

`model_strategy.model_params` 覆盖 XGBoost 参数时的经验注释（本项目默认值见 `models/ModelFactory.py` 的 `XGBoostModel.DEFAULT_PARAMS`）：

| 参数 | 说明 |
|---|---|
| `booster` | 弱评估器类型，默认 `gbtree` |
| `objective` | 训练目标；本项目默认 `reg:absoluteerror`（MAE 口径，抗异常值），sklearn 默认 `reg:squarederror` |
| `tree_method` | 训练算法：`auto`/`exact`/`approx`/`hist`/`gpu_hist` |
| `eval_metric` | 验证集评估指标；本项目默认 `mae` |
| `max_depth` | 树深度，默认 6，常用 [3,10]；越大偏差越小、方差越大 |
| `min_child_weight` | 叶子节点样本权重和的最小分裂阈值，默认 1；越大越欠拟合、越小越过拟合 |
| `gamma`（`min_split_loss`） | 节点分裂所需的最小损失下降值，越大越保守，常用 [0,1,5] |
| `subsample` | 行采样比例，默认 1，常用 [0.5,1]，防过拟合 |
| `colsample_bytree` | 每棵树的列采样比例，默认 1，常用 [0.5,1]，防过拟合 |
| `alpha`（`reg_alpha`） | L1 正则；高维下可加速，兼有特征选择效果 |
| `lambda`（`reg_lambda`） | L2 正则，默认 1；增大可减少过拟合 |
| `learning_rate`（`eta`） | 学习率，越小越稳但训练越慢，常与 `n_estimators` 联动调 |

## 配置生成

`config/generate_configs.py` 从数据集叶子目录自动检测数据特征（数据文件、时间列、频率、`now_time`、外生文件），批量生成分组 YAML 配置。生成前建议使用 `--dry-run` 检查输出路径和文件名。

```bash
uv run python config/generate_configs.py \
  --dataset dataset/<scenario> \
  --data-file <file>.csv \
  --target <target_col> \
  --dry-run
```

参数要点：

- `--dataset`、`--target` 必填；`--data-file` 缺省时自动取目录下第一个非外生 CSV；时间列缺省取 CSV 第一列；`freq` 缺省自动推断（`pd.infer_freq`，失败则按首两行时间差回退）；`now_time` 缺省取数据最后时间戳。
- `--no-date` / `--no-weather` / `--no-datetime` / `--no-lags` 分别关闭日期、气象、日期时间、滞后特征。
- `--models` 默认 `lightgbm,xgboost,catboost`；`--strategies` 默认 `usmdp,usmd,usmr,usmdr`（多变量策略 `msmd`/`msmr`/`msmdr` 可显式传入）。
- `--config-dir` 指定输出目录，缺省由 `dataset/` 路径推导到 `config/` 下同结构目录。
- `--history-days` / `--predict-steps`（以 `freq` 为单位，缺省为 1 天对应步数）/ `--window-days` 控制运行窗口。
- `--base-config` 缺省按策略与内生特征自动选择 `config.univariate_config` 或 `config.multivariate_config`。
