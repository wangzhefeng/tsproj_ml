# data_process 模块说明

`data_process/` 是**离线数据准备工具库**（L1 基础设施层）：聚合、填补回测、离线异常清洗、事件检测、峰谷检测、周期分析等，均按场景维度在「进模型之前」使用。canonical 主链的数据读取/信息集通路在 `data_loading/`（`SourceRegistry + InformationSetRequest`，2026-08-30 自 model_forecasting/data/ 迁为顶层包）。

## 与 canonical 信息集契约的关系（AGENTS.md「包间分层规则」流水线数据契约）

- 进入模型的信息集默认「**缺失/异常 = RAISE**」——这是预测性维护数据的预测性默认，registry 对列级 NaN、时间戳解析失败、序列缺失直接 RAISE，特征阶 NaN 由 lag 起点裁剪天然排除；
- 本库的填补/清洗只发生在**离线数据准备阶段**（先离线填好/清洗 → 产出 CSV → 进模型），不修改 canonical runtime 的信息集行为；
- 未来若引入 config 驱动的 compiler 前置清洗步骤，必须满足 as-of 可得性（`available_at <= forecast_origin`），且与评估掩码严格分离（前者改数据、后者只改评估口径）。

## 文件职责

| 文件 | 职责 |
|---|---|
| `data_aggregate.py` | 频率聚合（配置驱动，月频上界校验） |
| `fill_method_backtest.py` | 缺失填充方法掩码回测（离线裁决用哪种填补） |
| `outlier_process.py` | 离线异常标记/清洗/可视化 CLI |
| `outlier_handling.py` | **legacy 身份**：滑窗训练段异常清洗实现（自 `data_provider/` 迁入）。canonical runtime 不接线（`enable_train_outlier_handling` 已被 run.py 判为 canonical CLI 不支持项）；保留原因是报告 schema 仍被算力场景结果融合脚本消费。**禁止接入 canonical runtime** |
| `load_event_detection.py` | AIDC 负荷事件检测核心（shift/stress/burst/spike，15min↔日频一致，有单测） |
| `peak_valley_detection.py` | 峰谷检测与提取（配置驱动） |
| `periodicity_analysis.py` | 周期自动检测（FFT/ACF/STL；`_acf_periods/_fft_dominant_period` 将随收敛 P4 提公开名供 decomposition 使用） |

## 验证

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s tests -p "test_*.py"
```
