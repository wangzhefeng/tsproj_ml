"""model_testing.artifacts：回测产物写盘与可视化。

- `reporting.py`：回测 csv/图/元数据落盘；拼接总图要求折不重叠
  （sliding 形态经 `stitch_overview=False` 跳过总图，保留逐窗图）。
- `tensor_frames.py`：canonical 张量 → long DataFrame 纯转换，
  供 scoring 与 model_forecasting/model_ensemble 结果写盘共用；
  `CANONICAL_KEY_COLUMNS` / `BACKTEST_KEY_COLUMNS` 为唯一键定义。
- `decomposition_reports.py`：分解诊断报告写盘（休眠能力，按需启用）。
"""
