# 新能源 2025-09—2026-08 跨年回放

本目录仅两份配置：`demand_load/freq_{15min,1day}/lgbm_direct-pointwise.yaml`。
原始 CSV 保持不变；`prepare_xinnengyuan_2026.py` 严格校验 2025-09-01 至 2026-09-01（右开）的完整15min网格、有限非负值，按自然日 mean 生成365行日均数据及源哈希审计。不将日均最大值冒充15min最大需量。

## 窗口合同

- 2025年9月真值直接接入并计分，另外报告排除该段的指标。
- 2025年10、11月从9月1日起使用截至原点的全部历史。
- 2025年12月起，历史下界为预测区间起点减三个日历月；不是90天，也不是30天。
- 15min每日预测次日96点，全年335个模型窗口；1D每月一次预测完整自然月，全年11个模型窗口。原点前历史先截断，训练与特征使用同一窗口，禁止利用预测期真值。
- 仅 direct-pointwise；继承原新能源对应频率的L1、命名日历及60天时间衰减配置，不分解、不融合、不递归。
- **2025年10月日频短样本例外**：9月只有30日，不足以训练完整31步标签。使用既有非递归短样本pointwise路径：单步标签训练共享模型，以目标星期及日历特征直接预测整月；保留原点冻结的lag/rolling特征，不将预测喂回模型；该阶段不加时间权重。审计记录 training_horizon=1、forecast_horizon=31；不能宣称这一窗接受了完整多步标签训练。11月起恢复完整多步训练与权重。此先验未通过独立效果比较验证。

## 命令

```bash
# 数据准备和两份配置生成，不训练
 env -u PYTHONPATH .venv/bin/python config/hongtaiyang_cesuan/prepare_xinnengyuan_2026.py
# 下列为正式全年回放命令，按需另行启动
 env -u PYTHONPATH MPLBACKEND=Agg .venv/bin/python config/hongtaiyang_cesuan/annual_backtest.py --config-yaml config/hongtaiyang_cesuan/xinnengyuan_2026/demand_load/freq_15min/lgbm_direct-pointwise.yaml
 env -u PYTHONPATH MPLBACKEND=Agg .venv/bin/python config/hongtaiyang_cesuan/annual_backtest.py --config-yaml config/hongtaiyang_cesuan/xinnengyuan_2026/demand_load/freq_1day/lgbm_direct-pointwise.yaml
```

`--max-windows 1`为独立身份的冒烟结果，不是全年完成；不加`--rerun`可续跑匹配身份的完整窗口。结果位于正式`results/results_test/hongtaiyang_cesuan/xinnengyuan_2026/`下，周期、滚动规则、数据、配置及代码身份写入audit。YAML validation仅描述末窗参考几何，不能用通用run.py代替变长窗口年度调度。

旧`--freq`批量入口及`verify_results.py`仍仅覆盖原2025年20份矩阵，不纳入新周期。新周期通过显式`--config-yaml`运行；跨年窗口、真值接入、图表分窗及剔除首月评分由`tests/test_hongtaiyang_2026.py`覆盖。完成状态不代表已导出部署模型。
