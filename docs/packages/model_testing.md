# model_testing

`model_testing/` 是模型测试包：滑窗回测的几何、原语、逐折评分与回测产物落盘。2026-09-27 按职责细分三个子包；2026-09-28 补齐 `__init__.py`（纯包说明，不 re-export 符号，消费方全路径导入，与 `model_training`/model_predicting 门面收口约定一致）：

- `contracts/`：公共合同与几何（包外消费面）。
  - `protocols.py`（原 `contracts.py`）：`FoldScoringRunner`、`BacktestRunner`、runner factory、设计视图与窗口协议；模型和变换对象作为不透明载荷，不依赖上层实现类型。
  - `geometry.py`：fixed-step rolling-origin（`train_window_steps=None` 即 expanding 语义）、完整 calendar-month folds、标签非重叠排除（`_holdout_training_indices`）、`TimeGeometry/OriginTimeline` 公共时间几何。
  - `windows.py`：回测窗口构造——rolling 系（fixed/sliding/expanding）折、显式原始历史（train_history_steps，仅 fixed-step）窗口与显式 temporal 窗口（training_window/forecast_window，2026-10-05 引入；按原始时钟调度、折内应用 origin_sampling）（2026-09-27 自 `pipeline.supervised_design` 下沉，折载体复用 `geometry.RollingOriginFold`；`minimum_history` 由调用方显式传入）。
  - `primitives.py`：actual tensor、seasonal-naive 与正整数校验；forecast origin 解析已迁入 `forecasting_core/origin.py`（2026-09-27，部署路径通用原语）。
- `loops/`：回测循环形态与共用逐折体。`fixed_step.py` 内含 rolling 系共用引擎 `run_rolling_backtest()`，`sliding_window.py`/`expanding_window.py` 为薄入口（sliding 重叠折不拼总图，expanding 训练集逐折扩大）。折拟合/编译调度统一走 `loops/execution.py::ordered_bounded_map`（2026-10-05 引入：最多 workers 个在途、按序消费、失败传播并取消排队，评分留在消费线程）；training_window 配置的折 evidence 附 `forecast_origin`/`lead_steps`，产物 metadata 声明 `allow_overlapping_windows` 时跳过重叠拼接总图（保留逐窗图）。
  - `scoring.py`：两种回测共用的逐折评分体 `score_holdout_fold()`——predict 后处理 → point/probabilistic 评分 → CQR apply-before-collect → 执行证据；通过 `FoldScoringRunner` 消费公开能力，本包不 import model_predicting。
  - `fixed_step.py`：rolling 回测按 `refit_every` 选择拟合折，复用整组模型/变换/selector 状态；并行仅调度拟合折，评分与校准始终按窗口顺序进行。显式周期在执行证据中记录实际 fit origin/window/metadata，计划回测窗口不冒充已拟合窗口；历史准备委托 runner。默认每折重训。
  - `sliding_window.py`：重叠滑窗回测入口（stride_steps < horizon；`stitch_overview=False`）。
  - `expanding_window.py`：扩展窗回测入口（训练集不截断；backtest-only）。
  - `calendar_month.py`：calendar-month 回测生命周期（折构造、动态 config、并行拟合调度）；消费 scoring 共用体。
- `artifacts/`：产物写盘与可视化。
  - `tensor_frames.py`：canonical 张量到 long DataFrame 的纯转换，供 scoring 与 model_predicting/model_ensemble 结果写盘共用。
  - `reporting.py`：回测产物写盘与可视化（cv_plot_df/test_scores*/windows_results/总图）。
  - `decomposition_reports.py`：接收已计算的分解诊断 DataFrame，独占创建 CSV、不覆盖；不导入分解算法，不自动启用诊断。
- 逐窗图与总图统一约定Trues=actual_value实线、Preds=predict_value点划线；调用绘图助手时真值使用显式 `y_true` 参数，防止位置参数对调。回归见 `tests/test_backtest_plot_labels.py`。

两类回测循环均在本包，逐折体经 `scoring.score_holdout_fold` 共用同一实现。runner 与自然月 factory 通过显式 Protocol 注入，本包不导入上层 runner 实现，也不拥有 final fit、最终 bundle 或预测产物。

fixed-step 显式原始历史窗口通过 runner 的 `for_backtest_window()` 取得独立有界上下文；串行、并行拟合均由同一折上下文评分，不共用可变历史起点。窗口 metadata 记录 raw_history_start/end、train_history_steps 与预热后的 training_sample_count。actual 正常评分，不按离线填充来源新增掩码。

本包依赖 `forecasting_core`、`data_loading`、`model_evaluation`、`probabilistic`（CQR tracker 类型）及 `utils` 日志；不 import pipeline/model_predicting/model_ensemble。指标计算属于 `model_evaluation/`。
point 残差区间同样在逐折体 apply-before-collect，独立输出可用性和概率评分；每组缺样本时不伪造区间。启用重训复用不冻结校准池；final fit 仍由上层单独执行。

`model_testing` 不是 `tests/` 测试套件；不恢复旧 `ModelTesting` 类。

天气滑窗训练/测试统一使用完整 history 文件：拟合用实测列，测试预测用对应预报列；future 只用于真正未来推理，不参与回测。目标实际值仍用于训练标签和测试评分，递归测试特征使用自身目标预测。节假日/datetime 按请求时刻提供已知特征，无天气式双列映射。
