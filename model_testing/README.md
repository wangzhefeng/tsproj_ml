# model_testing

`model_testing/` 是模型测试包：滑窗回测的几何、原语、逐折评分与回测产物落盘。

- `geometry.py`：fixed-step rolling-origin、完整 calendar-month folds、标签非重叠校验、`TimeGeometry/OriginTimeline` 公共时间几何。
- `primitives.py`：actual tensor、seasonal-naive、forecast origin 与正整数校验。
- `contracts.py`：`FoldScoringRunner`、`BacktestRunner`、runner factory、设计视图与窗口协议；模型和变换对象作为不透明载荷，不依赖上层实现类型。
- `scoring.py`：两种回测共用的逐折评分体 `score_holdout_fold()`——predict 后处理 → point/probabilistic 评分 → CQR apply-before-collect → 执行证据；通过 `FoldScoringRunner` 消费公开能力，本包不 import model_forecasting。
- `fixed_step.py`：固定步长回测的并行拟合、按窗口顺序评分、聚合、provider 日志与回测产物写盘；历史准备委托 runner。
- `calendar_month.py`：calendar-month 回测生命周期（折构造、动态 config、并行拟合调度）；消费 scoring 共用体。
- `decomposition_reports.py`：接收已计算的分解诊断 DataFrame，独占创建 CSV、不覆盖；不导入分解算法，不自动启用诊断。
- `reporting.py`：回测产物写盘与可视化（cv_plot_df/test_scores*/windows_results/总图）。
- 逐窗图与总图统一约定Trues=actual_value实线、Preds=predict_value点划线；调用绘图助手时真值使用显式 `y_true` 参数，防止位置参数对调。回归见 `tests/test_backtest_plot_labels.py`。
- `tensor_frames.py`：canonical 张量到 long DataFrame 的纯转换，供 scoring 与 model_forecasting/model_ensemble 结果写盘共用。

两类回测循环均在本包，逐折体经 `scoring.score_holdout_fold` 共用同一实现。runner 与自然月 factory 通过显式 Protocol 注入，本包不导入上层 runner 实现，也不拥有 final fit、最终 bundle 或预测产物。

fixed-step 显式原始历史窗口通过 runner 的 `for_backtest_window()` 取得独立有界上下文；串行、并行拟合均由同一折上下文评分，不共用可变历史起点。窗口 metadata 记录 raw_history_start/end、train_history_steps 与预热后的 training_sample_count。actual 正常评分，不按离线填充来源新增掩码。

本包依赖 `forecasting_core`、`data_loading`、`model_evaluation`、`probabilistic`（CQR tracker 类型）及 `utils` 日志；不 import model_pipeline/model_forecasting/model_ensemble。指标计算属于 `model_evaluation/`。

`model_testing` 不是 `tests/` 测试套件；不恢复旧 `ModelTesting` 类。

天气滑窗训练/测试统一使用完整 history 文件：拟合用实测列，测试预测用对应预报列；future 只用于真正未来推理，不参与回测。目标实际值仍用于训练标签和测试评分，递归测试特征使用自身目标预测。节假日/datetime 按请求时刻提供已知特征，无天气式双列映射。
