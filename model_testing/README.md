# model_testing

`model_testing/` 提供纯时间几何和回测原语。

- `validation.py`：fixed-step rolling-origin、完整 calendar-month folds、标签非重叠校验。
- `backtest.py`：actual tensor、seasonal-naive、forecast origin 和正整数校验。

本包只依赖 `forecasting_core.tensors`。指标计算属于 `model_evaluation/`。

## 公共时间几何

- `TimeGeometry/OriginTimeline` 统一监督 origin 与标签范围的表达。
- `scheduled_origin_indices()` 确定正式 forecast-origin stride 网格，`rolling_origin_folds()` 在该网格选取固定步回测折。
- `calendar_month_folds()` 根据真实自然月解析 horizon，不把所有月份固定为同一日数。
- `is_label_safe()` 校验训练标签结束与验证标签开始之间的 gap；gap 只隔离标签，不平移预测原点。
- `actual_tensor()` 与 `seasonal_naive_tensor()` 生成与预测轴一致的真值/基线，`resolve_origin()` 处理原点解析。月频 naive 使用月步 offset，不用固定 Timedelta 替代。

`model_testing` 不是 `tests/` 测试套件，也不负责执行整轮回测：单模型 runtime 和融合 OOF 消费这里的公开几何。指标在 `model_evaluation/`，结果与绘图在 `model_forecasting/`；不恢复旧 `ModelTesting` 类。

```bash
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_intraday_schedule_contract
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_oof_gap_contract
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_calendar_month_horizon
```
