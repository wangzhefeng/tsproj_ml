# models 模块说明

`models/` 是训练、评估、预测和模型持久化层。数据读取和特征构造不在本目录完成，
而是由 `data_provider/` 与 `features/` 提供输入。

## 文件职责

| 文件 | 职责 |
|---|---|
| `ModelFactory.py` | 模型抽象层和工厂，统一创建 LightGBM/XGBoost/CatBoost/RandomForest |
| `ModelTraining.py` | 训练主逻辑，处理数据增强、特征选择、缩放后训练、调参、融合、分位数模型 |
| `ModelTesting.py` | 滑窗测试、窗口内训练/预测、指标计算、测试结果保存 |
| `ModelForecasting.py` | 7 种多步预测策略的推理实现 |
| `ModelEnsemble.py` | 时间序列融合回归器，支持 averaging/weighted/stacking/blending |
| `ModelSaveLoad.py` | 基于 pickle 的模型保存和加载 |
| `learning_rate.py` | 固定/自动学习率解析 |
| `losses.py` | 从模型参数推断损失名称和调参 scorer |

## ModelFactory

`ModelFactory` 当前支持的 `model_type`：

| 名称 | 别名 | 封装类 |
|---|---|---|
| `lightgbm` | `lgb` | `LightGBMModel` |
| `xgboost` | `xgb` | `XGBoostModel` |
| `catboost` | `cat` | `CatBoostModel` |
| `randomforest` | `rf` | `RandomForestModel` |

配置树和脚本主要使用 LightGBM / XGBoost / CatBoost。RandomForest 是工厂层可用模型，
不是当前批量配置的主路径。

工厂会先取模型类默认参数，再用配置中的 `model_params` 覆盖。线程字段由
`Trainer._apply_model_thread_limits()` 统一映射：

- LightGBM/XGBoost: `n_jobs`
- CatBoost: `thread_count`

## Trainer

`Trainer.train()` 是训练入口，输入为已经构造好的 `X_train`、`Y_train` 和类别特征列表。

主流程：

1. 判断是否可走 FourMethods baseline 路径。
2. 可选数据增强：`TimeSeriesAugmenter`。
3. 可选特征选择：`FeatureSelector`，选择结果会返回给预测阶段复用。
4. 可选学习率解析：`resolve_learning_rate()`。
5. 特征缩放和目标缩放：`FeatureScaler`、`TargetScaler`。
6. 可选原生数据容器准备：LightGBM `Dataset`、XGBoost `DMatrix`、CatBoost `Pool`，当前仅单输出启用。
7. 可选超参数搜索：`perform_tuning=True` 时使用 `RandomizedSearchCV` + `TimeSeriesSplit`。
8. 按 `predict_type` 和 `enable_ensemble` 进入点预测、融合或分位数训练。

### Baseline 路径

四种单变量策略在以下增强能力全部关闭时，会走 `_train_fourmethods_baseline()`：

- `scale_features`
- `scale_target`
- `enable_feature_selection`
- `enable_data_augmentation`
- `enable_ensemble`
- `perform_tuning`
- `enable_auto_learning_rate`

且 `predict_type="point"`。该路径保留原始特征/目标，无早停、无调参。

### 点预测

单输出时直接训练一个模型封装。

多输出时：

- `multi_output_strategy="regressor_chain"` 使用 sklearn `RegressorChain`。
- 默认 `multioutput` 使用项目内 `DirectMultiOutputRegressor`，每个 horizon 单独训练一个回归器，并可通过 `multi_output_n_jobs` 并行。

LightGBM 单输出和 direct 多输出路径会按末尾验证集启用 early stopping；样本不足时跳过。

### 分位数预测

`predict_type="quantile"` 时，训练阶段会为每个 `quantiles` 值训练一个子模型，
并返回如下 bundle：

```python
{
    "predict_type": "quantile",
    "quantiles": [...],
    "median_quantile": ...,
    "models": {q: model_q, ...},
}
```

不同模型的分位数参数注入：

- LightGBM: `objective="quantile"`、`metric="quantile"`、`alpha=q`
- XGBoost: `objective="reg:quantileerror"`、`quantile_alpha=q`
- CatBoost: `loss_function="Quantile:alpha=q"`

`quantile_parallel_workers` 控制分位数子模型并行训练。

### 模型融合

`enable_ensemble=True` 且 `predict_type="point"` 时启用。

可选方法：

- `averaging`
- `weighted`
- `stacking`
- `blending`

基础模型来自 `ensemble_models`，常见为 `["lgb", "xgb", "cat"]`。多输出目标会用
`MultiOutputRegressor` 包装基础估计器。

### 保存

`Trainer.model_save()` 写出：

```text
<checkpoints_dir>/model.pkl
<checkpoints_dir>/target_scaler.pkl    # 仅目标缩放器已 fit 时
```

保存工具为 `ModelDeployPkl`。

## Tester

`Tester._window_test()` 是单个滑窗任务入口。`main.Model.test()` 会构造多个窗口任务，
并在 `window_parallel_workers > 1` 时用 `ProcessPoolExecutor` 并行；如果进程池不可用，
会回退到 `ThreadPoolExecutor`。

窗口逻辑：

1. `_evaluate_split()` 从历史数据尾部倒推训练/测试窗口。
2. `handle_train_outliers()` 可选清洗训练窗口目标列。
3. `_build_window_train_xy()` 只在训练窗口内部构造特征和多步标签，避免 Direct 标签跨入测试期。
4. 训练窗口模型。
5. `_build_test_future_frame()` 只把测试期 `time` 作为未来模板，不透传测试期真实 `y`。
6. `Forecaster._predict_by_method()` 对测试窗口预测。
7. 计算 `R2/MSE/RMSE/MAE/MAPE/MAPE Accuracy`。

测试输出：

```text
<test_results_dir>/test_scores_df.csv
<test_results_dir>/cv_plot_df.csv
<test_results_dir>/train_outlier_report.csv
<test_results_dir>/test_prediction.png
```

`test_prediction.png` 只有在 `cv_plot_df` 中存在有效 `Y_trues/Y_preds` 时保存。

## Forecaster

`Forecaster._predict_by_method()` 根据 `args.pred_method` 分发：

| 方法 | 实现函数 |
|---|---|
| USMDO | `univariate_single_multi_step_direct_output_forecast()` |
| USMD | `univariate_single_multi_step_direct_forecast()` |
| USMR | `univariate_single_multi_step_recursive_forecast()` |
| USMDR | `univariate_single_multi_step_direct_recursive_forecast()` |
| MSMD | `multivariate_single_multi_step_direct_forecast()` |
| MSMR | `multivariate_single_multi_step_recursive_forecast()` |
| MSMDR | `multivariate_single_multi_step_direct_recursive_forecast()` |

重要实现边界：

- 预测前会重置 `quantile_outputs`，避免复用污染。
- Direct 类方法使用 `_build_direct_forecast_input()`，基于历史 `max_lag` 和未来外生特征构造锚点输入。
- 递归类方法会缓存 schema，预测阶段按训练时的 `selected_features` 对齐特征子集。
- USMDR/MSMDR 的块大小由 `block_size` 控制；未设置时使用最小 lag，若无 lag 则为 1。
- MSMR/MSMDR 对非目标内生变量主要使用 last-value/persistence 回填，不是完整多目标预测。
- 分位数 bundle 由 `_predict_point_and_quantiles()` 统一处理，点预测默认取最接近 0.5 的分位数。

预测输出保存由 `forecast_results_save()` 完成：

```text
<pred_results_dir>/prediction.csv
<pred_results_dir>/prediction.png
<pred_results_dir>/history_context.csv
<pred_results_dir>/prediction_plot_concat.csv
```

`prediction.csv` 基础列为：

```text
time,predict_value
```

若启用分位数预测，会追加：

```text
predict_q10,predict_q50,predict_q90,...
```

实际列名由 `quantiles` 决定。

## 与其他模块的边界

- `models/` 不负责读取 CSV；读取和历史/未来时间轴由 `DataLoader` 完成。
- `models/` 不负责决定原始特征列；特征列由 `FeatureEngineer.create_features()` 返回。
- 缩放器对象由 `main.Model.train()` 创建，训练细节在 `Trainer` 内完成。
- `selected_features` 从训练阶段返回，并由 `Forecaster` 在推理阶段复用。
- 目标列在进入模型前通常已统一为 `y`；直接使用原始 `target` 字段做模型输入通常是错误的。

## 常见注意事项

- 当前没有单元测试目录。改模型逻辑时至少做 `py_compile`，并跑一个轻量配置验证真实链路。
- 多输出早停和分位数训练可能显著增加训练时间，调大并行参数前先确认机器资源。
- `perform_tuning=True` 会触发搜索，不能当作普通快速测试。
- `predict_type="quantile"` 与 `enable_ensemble=True` 同时设置时，当前训练逻辑会走分位数单模型 bundle，不走点预测融合。
- 目标缩放启用后，是否恢复原始量纲由 `inverse_target` 控制；评估和预测保存都会受此影响。
