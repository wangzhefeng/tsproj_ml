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
- `tensor_frames.py`：canonical 张量到 long DataFrame 的纯转换，供 scoring 与 model_forecasting/model_ensemble 结果写盘共用。

两类回测循环均在本包，逐折体经 `scoring.score_holdout_fold` 共用同一实现。runner 与自然月 factory 通过显式 Protocol 注入，本包不导入上层 runner 实现，也不拥有 final fit、最终 bundle 或预测产物。

本包依赖 `forecasting_core`、`data_loading`、`model_evaluation`、`probabilistic`（CQR tracker 类型）及 `utils` 日志；不 import model_pipeline/model_forecasting/model_ensemble。指标计算属于 `model_evaluation/`。

`model_testing` 不是 `tests/` 测试套件；不恢复旧 `ModelTesting` 类。
