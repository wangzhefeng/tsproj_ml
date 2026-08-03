# Config 说明

`config/` 保存模型运行配置、共享配置分组、YAML 配置、配置加载器和配置生成器。标准默认值由 Python `@dataclass`（模板模块）提供；具体数据和任务参数通过 YAML 覆盖。

## 目录结构

```text
config/
├── config_sections.py      # 共享配置分组 dataclass 定义（12 个分组 + BaseModelConfig）
├── config_loader.py        # YAML 配置加载器
├── generate_configs.py     # 从数据集目录批量生成分组 YAML 配置
├── univariate_config.py    # 单变量场景模板模块（ModelConfig 子类）
├── multivariate_config.py  # 多变量场景模板模块（ModelConfig 子类）
└── <dataset_scenario>/     # 各数据场景的 YAML 配置目录（模型配置 / 聚合 / 异常检测）
```

## 模板模块

`config_sections.py` 定义扁平组合配置基类 `BaseModelConfig`，由 12 个分组 dataclass 继承组合：

- `RuntimeConfig`：运行模式（测试/预测）、时间窗口（`history_days`、`predict_steps`、`window_days`、`now_time`）
- `TargetSeriesConfig`：目标序列数据路径、频率、时间列、目标列、内生数值/类别/剔除列
- `ExogenousFeatureConfig`：日期类型外生、气象外生、日期时间派生特征
- `TimeLagFeatureConfig`：滞后特征开关与滞后步数
- `AdvancedFeatureConfig`：滚动/扩展窗口统计、差分、百分比变化、距事件、周期三角编码、交互、多项式特征
- `PreprocessingConfig`：特征/目标缩放、去趋势、分组缩放、类别编码
- `ModelStrategyConfig`：模型类型、融合模式、预测方法、多输出策略、分位数
- `TrainingEnhancementConfig`：早停、时间衰减样本权重、超参调优、数据增强、特征选择、学习率策略
- `TrainOutlierConfig`：滑窗测试训练窗口异常清洗
- `EvalMaskConfig`：评估/绘图掩码（`percentile`/`absolute`/`combined`）
- `PerformanceConfig`：窗口/多输出/分位数/融合并行度、日志
- `OutputConfig`：结果输出目录与场景子路径

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

> 时间衰减权重兼容性：默认配置下（`multi_output_strategy: multioutput` + `predict_type: point` + `enable_ensemble: false`）所有 7 种预测方法均支持；`regressor_chain` 多输出、`ensemble`、多输出分位数路径不支持，开启时会 warning 跳过（不阻塞运行）。

启用示例：

```yaml
  training_enhancement:
    enable_time_decay_sample_weight: true
    decay_halflife_days: 14
```

其余分组（runtime、target_series、exogenous_features、time_lag_features、advanced_features、preprocessing、model_strategy、train_outlier、eval_mask、performance、output）的字段与默认值以 `config_sections.py` 为准。

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
