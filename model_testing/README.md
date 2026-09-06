# model_testing

`model_testing/` 是模型测试包：调参滑窗回测的几何、原语、逐折评分与回测产物落盘（2026-09-06 R5 按方案 v3 重新定位，`.hermes/plans/2026-09-06_162605`）。

- `geometry.py`：fixed-step rolling-origin、完整 calendar-month folds、标签非重叠校验、`TimeGeometry/OriginTimeline` 公共时间几何。
- `primitives.py`：actual tensor、seasonal-naive、forecast origin 与正整数校验。
- `scoring.py`：两种回测共用的逐折评分体 `score_holdout_fold()`——predict 后处理 → point/probabilistic 评分 → CQR apply-before-collect → 执行证据；runner 以 duck-typing 协议传入（fixed-step 与 calendar-month、ensemble 成员同一协议面），本包不 import model_forecasting。
- `calendar_month.py`：calendar-month 回测生命周期（折构造、动态 config、并行拟合调度）；消费 scoring 共用体。
- `reporting.py`：回测产物写盘与可视化（cv_plot_df/test_scores*/windows_results/总图）。
- `tensor_frames.py`：canonical 张量到 long DataFrame 的纯转换，供 scoring 与 model_forecasting/model_ensemble 结果写盘共用。

fixed-step 回测循环留在 `model_forecasting/runtime.py`（orchestration），逐折体经 `scoring.score_holdout_fold` 与 calendar-month 共用同一实现。

本包依赖 `forecasting_core`、`data_loading`、`model_evaluation`、`probabilistic`（CQR tracker 类型）；不 import model_forecasting/model_ensemble。指标计算属于 `model_evaluation/`。

`model_testing` 不是 `tests/` 测试套件；不恢复旧 `ModelTesting` 类。
