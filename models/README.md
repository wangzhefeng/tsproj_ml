# models 模块说明

`models/` 是训练、测试、预测、融合和模型持久化层。CSV 读取由
`data_provider/` 完成，特征构造由 `features/` 完成。

## 文件职责

| 文件 | 职责 |
|---|---|
| `ModelFactory.py` | 统一创建 LightGBM、XGBoost、CatBoost、RandomForest 封装 |
| `ModelTraining.py` | 训练、调参、分位数模型、融合、特征选择和数据增强入口 |
| `ModelTesting.py` | 滑窗测试、窗口内训练预测、指标计算和测试结果保存 |
| `ModelForecasting.py` | 7 种多步预测策略的推理实现 |
| `ModelEnsemble.py` | averaging、weighted、stacking、blending 融合回归器 |
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

支持模型：

- `lightgbm` / `lgb`
- `xgboost` / `xgb`
- `catboost` / `cat`
- `randomforest` / `rf`

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
<test_results_dir>/train_outlier_report.csv
<test_results_dir>/test_prediction.png
```

`train_outlier_report.csv` 始终写出；未启用或未发现训练异常时为空表头。

`test_scores_df.csv` 中的 `MAPE` 与 `MAPE Accuracy` 采用业务口径：对每个测试窗口先取 `y_true > 0` 的点，再用这些正样本的 `P5` 作为相对阈值，只对 `y_true >= threshold` 的点计算指标。结果表会同步保存 `MAPE Threshold`、`MAPE Valid Points`、`MAPE Excluded Points`、`MAPE Excluded Ratio`。`cv_plot_df.csv` 也会保留对应的有效性标记，`test_prediction.png` 对无效点断线显示。

## 预测

`Forecaster._predict_by_method()` 根据 `args.pred_method` 分发：

| 方法 | 说明 |
|---|---|
| USMDO | 单变量多步逐点 direct |
| USMD | 单变量多步 direct |
| USMR | 单变量递归预测 |
| USMDR | 单变量分块 direct-recursive |
| MSMD | 多变量 direct 预测目标 |
| MSMR | 多变量递归预测目标 |
| MSMDR | 多变量分块 direct-recursive |

预测输出：

```text
<pred_results_dir>/prediction.csv
<pred_results_dir>/prediction.png
<pred_results_dir>/history_context.csv
<pred_results_dir>/prediction_plot_concat.csv
```

`prediction.csv` 基础列为 `time,predict_value`。启用分位数预测时追加
`predict_q10,predict_q50,predict_q90,...`。

`prediction_plot_concat.csv` 用于未来预测图排障，当前除 `time,value,series_type` 外还会保存：

- `raw_value`：历史上下文真实值或未来预测原值
- `plot_value`：用于 `prediction.png` 的绘图值；历史上下文低于正样本 `P5` 阈值的点写为 `NaN`
- `plot_valid`：该点是否参与主图连线

`prediction.png` 的历史上下文主线使用 `plot_value`，未来预测主线始终使用 `prediction.csv` 中的原始 `predict_value`，不做裁剪或平滑。

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
