"""model_testing.loops：回测循环形态与共用逐折评分体。

- `fixed_step.py`：固定步长 rolling-origin 回测；内含 rolling 系共用引擎
  `run_rolling_backtest()`（并行拟合、按窗口序评分、产物落盘）。
- `sliding_window.py`：重叠滑窗回测入口（`stride_steps < horizon`，
  重叠折不拼接总图）。
- `expanding_window.py`：扩展窗回测入口（训练集取全部合格历史候选、
  逐折扩大；无固定训练窗口语义，暂限 backtest-only）。
- `calendar_month.py`：calendar-month 回测生命周期（自然月折构造、
  动态 config、并行拟合调度）。
- `scoring.py`：逐折评分共用体 `score_holdout_fold()`——predict 后处理 →
  point/probabilistic 评分 → CQR apply-before-collect → 执行证据；
  经 `contracts.protocols.FoldScoringRunner` 消费 runner 能力。

四种循环共用同一引擎与评分体，新形态只在外壳层差异化。
"""
