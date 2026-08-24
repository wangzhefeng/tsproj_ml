# models 模块说明

`models/` 是训练、测试、预测、融合和模型持久化层。CSV 读取由
`data_provider/` 完成，特征构造由 `features/` 完成。

## 文件职责

| 文件 | 职责 |
|---|---|
| `ModelFactory.py` | 统一创建 10 类回归器封装（LightGBM、XGBoost、CatBoost、RandomForest、HistGradientBoosting、Ridge、ElasticNet、Lasso、QuantileRegressor、SeasonalTemplate） |
| `ModelTraining.py` | 训练、调参、分位数模型、融合、blend 双子模型、时间衰减样本权重、特征选择和数据增强入口 |
| `ModelTesting.py` | 滑窗测试、窗口内训练预测、指标计算和测试结果保存 |
| `ModelForecasting.py` | 9 种多步预测策略的推理实现 |
| `ModelEnsemble.py` | averaging、weighted、stacking、blending 融合回归器（成员级 preprocessor） |
| `AuxiliaryForecaster.py` | 非目标内生变量的 reduced-form 递归辅助预测器（`endogenous_backfill_strategy: auxiliary`） |
| `ModelSaveLoad.py` | pickle 模型和目标缩放器保存/加载 |
| `learning_rate.py` | 固定/自动学习率解析 |
| `losses.py` | 模型损失名称和调参 scorer 推断 |

## 训练

`Trainer.train()` 输入已经构造好的 `X_train`、`Y_train` 和类别特征列表。
主流程包括：

1. 可选数据增强。
2. 可选特征选择。
3. 可选自动学习率。
4. 特征缩放和目标缩放。
5. 可选超参数搜索。
6. 按 `predict_type` 和 `enable_ensemble` 进入点预测、融合或分位数训练。

支持模型（`ModelFactory._models` 注册名，均带别名）：

- `lightgbm` / `lgb`
- `xgboost` / `xgb`
- `catboost` / `cat`
- `randomforest` / `rf`
- `histgb` / `histgradientboosting`
- `ridge`
- `elasticnet` / `enet`
- `lasso`
- `quantileregressor` / `qr`
- `seasonaltemplate` / `st`（工作日/周末分组建季节模板 + NNLS 学权重）

## 测试

`Tester._window_test()` 是单个滑窗任务入口。`main.Model.test()` 负责构造窗口任务，
并在 `window_parallel_workers > 1` 时并行执行。

窗口边界：

- 训练窗口和测试窗口从历史数据尾部倒推。
- 可选训练段异常处理只修改训练窗口目标列。
- 测试段真实 `y` 不会透传给预测器。
- Direct 类方法的训练标签只在训练窗口内部构造，避免跨入测试期。

测试输出：

```text
<test_results_dir>/test_scores_df.csv
<test_results_dir>/cv_plot_df.csv
<test_results_dir>/probabilistic_predictions_df.csv
<test_results_dir>/probabilistic_scores_df.csv
<test_results_dir>/horizon_scores_df.csv
<test_results_dir>/probabilistic_intervals_df.csv
<test_results_dir>/calibration_report.csv
<test_results_dir>/train_outlier_report.csv
<test_results_dir>/test_prediction.png
```

`train_outlier_report.csv` 始终写出；未启用或未发现训练异常时为空表头。

`test_scores_df.csv` 中的 `MAPE` 与 `MAPE Accuracy` 采用业务口径：按 `eval_mask`（`utils/eval_mask.build_eval_mask`）过滤后计算——默认 `mode: percentile` 即取窗口正样本的 `P5` 作为相对阈值，只对 `y_true >= threshold` 的点计算指标；`absolute`/`combined` 模式可启用 `min_value` 绝对下限，`max_value` 上限与 mode 正交。结果表会同步保存 `MAPE Threshold`、`MAPE Upper Threshold`、`MAPE Valid Points`、`MAPE Excluded Points`、`MAPE Excluded Ratio`。`cv_plot_df.csv` 也会保留对应的有效性标记，`test_prediction.png` 对无效点断线显示。

每个窗口同时输出季节 naive 对照列 `Naive MAPE` / `Naive MAPE Accuracy`：naive 值取测试点 `n_per_day` 步前（昨日同时刻）的实际值，与模型指标共用同一 eval_mask，用于判断模型是否跑赢持久性锚点；窗口历史不足一天时记 `NaN`。

全部窗口的汇总指标由 `main.py` 用 **median（中位数）** 追加为「中位数」行——单窗口 MAPE 爆炸不会拖垮汇总。

quantile 模式按 `ProbabilisticSpec` 指定 interval 在 processed target-space boundaries 上记录 `conformal_score`。历史窗口按 `label_available_at <= current_origin` 做 prequential CQR，final forecast 复用同一 as-of selector；模型 quantile 与 calibrated prediction interval 分开保存。Blend 方法额外记录 `blend_direct_pred`/`blend_recursive_pred`，供 `ridge_stacking` 学共享权重。

## 预测

`Forecaster._predict_by_method()` 根据 `args.pred_method` 分发；point 返回一维数组，quantile 返回统一的 `ForecastDistribution`。迁移期 `quantile_outputs` 仅保留只读兼容视图：

| 方法 | 说明 |
|---|---|
| USMDP | 单变量多步逐点 direct |
| USMD | 单变量多步 direct |
| USMR | 单变量递归预测 |
| USMDR | 单变量分块 direct-recursive |
| MSMD | 多变量 direct 预测目标 |
| MSMR | 多变量递归预测目标 |
| MSMDR | 多变量分块 direct-recursive |
| USBR | 单变量 Direct+Recursive 加权融合（blend） |
| MSBR | 多变量 Direct+Recursive 加权融合（blend） |

预测输出：

```text
<pred_results_dir>/prediction.csv
<pred_results_dir>/prediction.png
<pred_results_dir>/history_context.csv
<pred_results_dir>/prediction_plot_concat.csv
```

`prediction.csv` 基础列为 `time,predict_value`。启用分位数预测时追加
`predict_q10,predict_q50,predict_q90,...`，并强制 `predict_value == predict_q50`；启用 CQR 时另加 `predict_pi<coverage>_lower/upper`，不覆盖模型 quantile。

`prediction_plot_concat.csv` 用于未来预测图排障，当前除 `time,value,series_type` 外还会保存：

- `raw_value`：历史上下文真实值或未来预测原值
- `plot_value`：eval_mask 掩码后的绘图候选值；历史上下文被掩码判为异常的点写为 `NaN`
- `plot_valid`：该点是否通过 eval_mask

`prediction.png` 的历史上下文主线当前直接绘制**原始 `y`**（保证线条连续不断线），未来预测主线使用 `prediction.csv` 中的原始 `predict_value`，不做裁剪或平滑；`plot_value`/`plot_valid` 保留在 CSV 中供排障和自定义绘图使用。（滑窗测试图 `test_prediction.png` 仍按 eval_mask 对无效点断线显示。）

## 与入口配置的关系

- `models/` 不加载配置模块；配置实例由 `run.py` 或 `main.py` 创建后传入。
- `config/config_loader.py` 导入时不解析命令行，避免 `run.py` 导入 `main.Model` 时发生二次参数解析。
- 模型线程数由 `model_thread_count` 通过训练层映射到 LightGBM/XGBoost 的
  `n_jobs` 或 CatBoost 的 `thread_count`。

## 验证

模型层已有 unittest 覆盖训练窗口异常报告和测试窗口边界。改动模型逻辑后至少运行：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```
