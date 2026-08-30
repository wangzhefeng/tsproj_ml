# Config 说明

`config/` 保存 canonical schema 模型 YAML、独立数据工具 YAML 和配置加载器。canonical 默认与校验由 `model_forecasting/specs/` 的不可变 specs 定义。

## Canonical schema 当前状态

仓库内 **824 个现役模型 YAML 全部为 canonical schema（`schema_version: 2`）**。`config_loader.load_yaml_config()` 对 canonical 配置返回严格 `ForecastConfigSpec`，由 `main.py::CanonicalModel` 进入 `model_forecasting.runtime.run_canonical_config()`；七种标准 strategy、targets 列表、六种列角色、source registry、long result 和 schema-2 bundle 都是当前实现事实。

legacy 配置体系（`base_config + overrides`、扁平 `BaseModelConfig`、九种 `pred_method` 及其只读模板）已于 2026-08-29 删除；`load_yaml_config()` 只解析 canonical schema，`main.py` / `run.py` 拒绝一切非 canonical 配置。

## 目录结构

```text
config/
├── config_loader.py        # canonical schema 严格 parser（非 canonical RAISE）
└── <dataset_scenario>/     # 各数据场景的 YAML 配置目录（模型配置 / 聚合 / 异常检测）

model_forecasting/specs/config.py # ForecastConfigSpec、strict parser、fingerprint/identity
scripts/migrate_forecast_configs.py
scripts/audit_forecast_configs.py
scripts/check_model_configs.py
```

## Canonical YAML 合同

顶层只允许：`schema_version/problem/data/features/strategy|model_ensemble/estimator/probabilistic/validation/output`。`strategy` 与 `ensemble` 必须且只能出现一个；horizon 只在 `problem` 声明；targets 与 target-role 列的顺序必须完全一致。

```yaml
schema_version: 2
problem:
  time_col: time
  freq: 15min
  horizon: 96
  targets: [load]
  information_mode: forecast
  training_scope: local
  series_id_cols: []
data:
  sources:
    - name: target_history
      source_type: file
      history_path: dataset/<scenario>/load.csv
      time_col: time
      availability: source_time
      columns:
        - {name: load, role: target, categorical: false}
features:
  target_lags: {load: [96, 192, 672]}
  observed_past_lags: {}
  datetime_features: [hour, day_of_week]
  transformations: {}
strategy: {name: recursive}
estimator:
  model_type: lightgbm
  target_adapter: independent
  params: {}
probabilistic: {mode: point}
validation:
  forecast_origin: "2026-07-31T23:45:00"
  history_length: 304
  window_length: 30
output:
  scenario_subpath: <scenario>/baseline
  results_root: results
```

加载器额外执行：重复 YAML key 检测、unknown nested field 检测、六角色唯一性、source 类型/availability/provider 约束、七策略 chunk 约束、Global key 一致性、transform 构造校验和 canonical fingerprint 计算。

## 概率输出边界

canonical 当前支持 point 与每目标边际 quantile：

```yaml
probabilistic:
  mode: quantile
  quantiles: [0.1, 0.5, 0.9]
  point_quantile: 0.5
  recursive_propagation: median_path
```

输出保留 `predict_q*`，point 锚定 q50；recursive quantile 使用 median path。joint samples 明确 unsupported。迁移 YAML 中可见的 `probabilistic.conformal` 只保留 legacy CQR 意图，**canonical runtime 当前不执行 CQR，也不输出 `predict_pi*`**；旧 CQR 说明不能当作当前已实现能力。

## 全量审计

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/check_model_configs.py

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/audit_forecast_configs.py --root config --output /tmp/forecast-config-audit.json
```

当前口径：865 个 config YAML，其中 824 个模型 YAML（全部 canonical schema）、41 个独立聚合/检测 YAML。18 个非整除 RecMO 曾于 2026-08-29 获 wangzf 批准映射为 `recursive`（`aidc_power_month` 14 个、`aidc_load_month` 4 个）；溯源字段已随 legacy 清理删除，记录见 `docs/multistep_forecasting_redesign.md`。

## Legacy 配置参考（历史记录）

以下说明对应已删除的 grouped override 机制，仅作历史解释：canonical YAML 现通过 `output.scenario_subpath` 显式指定结果输出路径中的 `<scenario>` 段，使 `results/` 下的目录结构与 `config/` 对齐。多组配置共用同一 `data_dir` 时必须显式指定，否则结果会混入同一目录。

## 配置分组参考

canonical YAML 按 `problem/data/features/strategy|model_ensemble/estimator/probabilistic/validation/output` 分组；字段校验由 `model_forecasting/specs/` 的不可变 specs 定义。

### runtime：自然月动态 horizon

`horizon_mode: fixed_steps`（默认）保持 `horizon = predict_steps`。日频预测完整自然月时使用：

```yaml
validation:
  horizon_mode: calendar_month
  train_window_length: 120
  schedule_mode: daily
```

`calendar_month` 当前仅支持 `freq: 1D`：forecast_start 必须为月初，最终 horizon 自动取目标月的 28/29/30/31 天，forecast_end 为下月月初；滑窗测试按完整自然月切分，每个 fold 使用自己的 horizon，同时固定最近 `train_window_length` 天作为训练段。setting 编码真实 `train_window_length` 并追加 `-calendar-month`（不再使用遗留 `window_length` 数字），与历史固定步长结果隔离。测试汇总额外包含 `Calendar Month`、`Forecast Steps`、月实际/预测总量和 `Monthly Total MAPE`。未来气象文件必须覆盖目标自然月最后一天。

### exogenous_features：严格气象信息集

月初预测完整自然月时，天气必须区分历史实测、回测 ex-ante 代理/预报和正式未来代理/预报：

```yaml
data:
  sources:
    - name: weather
      strict_weather_information_set: true
      weather_history_source: actual
      weather_backtest_path: weather_daily_stats_backtest_proxy_*.csv
      weather_backtest_source: proxy
      weather_future_source: proxy
```

strict 模式要求 backtest 每行 `available_at` 早于目标月、future 每行不晚于 `now_time`，且目标时间戳全覆盖；不允许把测试月实测天气当未来天气。Direct/DirRec 使用天气时必须启用 `use_horizon_exogenous_for_direct: true`，训练侧 `*_h1..*_hH` 按目标日构造；horizon-feature melt 按 h 折叠回基础列名，与推理端未来逐日外生一致。USBR 的 Direct/Recursive 共享 X 暂不支持目标日天气，保留为无天气 control。

USMDP 默认不生成目标 lag，适合日历/天气逐点模板；如需使用已知历史 lag，必须同时设置 `align_direct_features_to_target: true`、`enable_lags_features: true` 且 `min(lags) >= horizon`。该 safe-lag 路径只读取已知历史，不递归消费预测值；需要逐步回填自身预测时使用 USMR。

DirRec（USMDR/MSMDR）的 `block_size: 0` 表示从最短 lag 推导有效块长，显式正值则直接指定；最终均截到 horizon。训练目标宽度、模型输出宽度和推理块长统一使用解析后的 B。

Global panel 配置使用 `enable_global_training: true` 与 `series_id_feature`；历史/未来按 `(series_id, time)` 校验，每条序列必须覆盖完整 horizon。`global_incomplete_series_policy` 支持 `raise`/`drop`，未知 series 当前只允许 `raise`。

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

USBR/MSBR blend 暂不与目标分解组合；point、全部 quantile、target scaling 与 CQR 共用 `TargetTransformPipeline`，按严格逆序恢复到目标空间。启用分解的最终训练状态随统一模型 bundle 保存。

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

> 时间衰减权重兼容性：默认 `multi_output_strategy: multioutput` 下 point/quantile 都支持；多输出统一走项目自定义 `DirectMultiOutputRegressor`，逐输出转发同一 `sample_weight`。`regressor_chain` 与 `ensemble` 路径不支持，开启时 warning 跳过（不阻塞运行）。

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
| `output` | `setting_suffix` | 结果目录 setting 后缀（如 `-control-no-weather`），同配置不同语义版本结果隔离；消融矩阵的 no-op control 用 `-control-no-<feature>` 标注 |
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

## Legacy 迁移边界

`config/generate_configs.py`、迁移器与只读解析分支均已删除：`load_yaml_config()` 只解析 canonical schema，非 canonical YAML 直接 RAISE。
