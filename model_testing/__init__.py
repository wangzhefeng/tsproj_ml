"""model_testing：回测专用包——回答「这个模型配置在过去表现如何」。

与 model_predicting（面向未来的预测与部署）分工：本包只做历史回测，
不写部署产物、不交付预测。

- `contracts/`：包外消费的被动合同——geometry（时间几何与 rolling/calendar
  折构造）、windows（回测窗口构造）、protocols（runner 注入协议）、
  primitives（actual/seasonal-naive 张量原语）。
- `loops/`：回测循环形态与共用逐折体——fixed_step（内含 rolling 系共用引擎
  `run_rolling_backtest()`）、sliding_window（重叠折，不拼总图）、
  expanding_window（训练集逐折扩大，backtest-only）、calendar_month
  （自然月几何）、scoring（逐折评分共用体）。
- `artifacts/`：产物写盘与可视化——reporting（csv/图/元数据）、
  tensor_frames（canonical 张量 → long DataFrame）、decomposition_reports
  （分解诊断报告）。

依赖方向：本包向下消费 forecasting_core / data_loading / model_evaluation /
probabilistic；不 import pipeline / model_predicting / model_ensemble，
上层通过 `contracts.protocols` 的 Protocol 注入 runner，门禁见
`tests/test_package_layering.py`。

无包级 re-export 门面：消费方统一走完整点路径导入（与 model_predicting
2026-09-26 门面收口约定一致）。
"""
