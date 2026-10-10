# AIDC 5min 负荷数据准备

本目录是 **5min 负荷的离线数据准备入口，不是模型实验矩阵**；当前没有 `schema_version: 2` 模型 YAML。`outlier_detect.yaml` 是数据处理配方，不能传给 `run.py --config-yaml`。运行环境约定见根目录 [AGENTS.md](../../AGENTS.md)。

## 目录内容与两条数据链

| 文件 | 用途 | 输入与输出 |
|---|---|---|
| [outlier_detect.yaml](outlier_detect.yaml) | A/B 路总负荷异常检测配方 | `dataset/aidc_load_5min/{A,B}_Loads_5min_20251001_20260731.csv` → 同数据目录下 `outlier_analysis/` |
| [point_load_aggregate.py](point_load_aggregate.py) | 点位表驱动的暖通、UPS、列头柜汇总 | `A1_A2_A3_points/` → `A1_A2_A3_data/route_{A,B}/` |

两条链相互独立：点位汇总不会自动运行总负荷异常检测，也不等于 [暖通预测场景](../aidc_hvac_load_5min/模型测试说明.md) 的严格完整总量与填补流程。`load` 表示负荷功率；点位汇总输出为 **kW**，不是按时间累加的 kWh。

## 总负荷异常处理

- 配方同时覆盖 A/B 两路；物理上下界为 `0～25000`，孤立跳飞参数为 `spike_window: 2`、`spike_z_threshold: 8`、绝对差阈值 `1500`、同伴离散阈值 `500`。
- 只自动处理物理越界和两侧稳定同伴支持的孤立跳飞；通用局部/周期统计阈值设为 `.inf`，`auto_clean_statistical: false`，不据统计异常自动抹平持续负荷变化。
- 执行器为 `data_process/outlier_process.py::run_outlier_detection`；输出异常标记、审计及 `<stem>_remove_outlier.csv`。清洗含时间插值与前后填充，属于**离线处理**，不能声称在当时严格在线可得。
- 15min、日频派生场景消费清洗后的 5min CSV，不应在聚合后重复应用本配方；日均负荷链见 [日频长跨度测试说明](../aidc_load_month/模型测试说明.md)。

## 点位聚合口径

默认参考表为 `dataset/aidc_load_5min/A1_A2_A3_points/all_ids.xlsx`，点位文件按 `<data_type>/<spot_id>.csv` 读取，必须有 `time/value` 列。参考表按 `暖通负荷`、`UPS负荷`、`列头柜负荷` 三个 sheet 选择点位，需要 `data_type/deviceId/spot_id/备注/route` 及 `spotName` 或 `spot_name`。

| 类别 | 选择规则 | 为什么需要 |
|---|---|---|
| 暖通主要设备 | 冷水机组、冷却塔、冷冻水一次泵、冷冻水二次泵 | 明确“主要设备”口径，不等于所有暖通设备 |
| UPS | 排除备注为“一楼”的点位 | 保留脚本既定统计范围，避免误认为全量 UPS |
| 列头柜 | 按楼栋、路线筛选 | 与 UPS 分开汇总，不把供电层级相加造成重复统计 |
| 总功率/分相 | 同设备有总功率时排除其 A/B/C 相功率 | 防止总量与分相重复计入 |

每类生成 A1、A2、A3 和三楼合计，分别按 A/B 路输出，共 **24 份 CSV**。默认网格为 `2025-10-01 00:00:00～2026-07-31 23:55:00`、5min；`--freq` 改变对齐网格，**不执行时间聚合**。

- `unit_spot=W` 乘 `0.001` 转为 kW，kW 原样保留；空单位按系数 1 处理，不代表单位已被证实，其他单位报错。
- 非法时间丢弃并审计；非数值/非有限值记 NaN；重复时刻保留最后一行，非网格时刻不就近取整。缺文件、重复点位及缺测均进入审计，不自动填补。
- 单楼 CSV 保留点位列并生成 `value`；三楼 CSV 为 `time/a1_value/a2_value/a3_value/value`。使用 `sum(min_count=1)`：全缺失为 NaN，部分可用时为部分和，**不保证完整总负荷**。
- 审计文件为 `A1_A2_A3_points_aggregate_audit.json`，记录点位选择、单位转换、时间问题、缺文件和输出缺失。

## 使用与副作用

安全查看参数：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_load_5min/point_load_aggregate.py --help
```

实际聚合使用同一入口，按需传入 `--reference-path/--points-root/--output-dir/--start-time/--end-time/--freq`。**脚本没有 validate-only 或覆盖确认开关**：运行会原子替换目标 CSV/审计，并删除输出目录内匹配预设旧命名的文件；先审核输入和输出范围，获准后再执行，不把 `--help` 当成数据验收。

相关回归位于 `tests/test_point_load_aggregate.py`；测试分组与命令见 [tests/README.md](../../tests/README.md)。数据覆盖、缺失率和完整总量资格必须另查实际审计；本说明不提供模型效果或部署结论。
