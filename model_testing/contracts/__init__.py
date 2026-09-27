"""model_testing.contracts：回测包的公共合同层（包外消费面）。

- `protocols.py`：`FoldScoringRunner` / `BacktestRunner` 注入协议与
  `BacktestWindow` / `FitResult` 载体，模型与变换对象为不透明载荷。
- `geometry.py`：`TimeGeometry` / `OriginTimeline` 公共时间几何，
  rolling-origin 折（`train_window_steps=None` 即 expanding 语义）、
  calendar-month 折与标签非重叠排除。
- `windows.py`：回测窗口构造入口——rolling 系（fixed/sliding/expanding）
  与显式原始历史（train_history_steps，仅 fixed-step）。
- `primitives.py`：actual tensor、seasonal-naive 基线与正整数校验。

只含被动定义与纯函数，不执行编排；上层经 `protocols` 注入实现。
"""
