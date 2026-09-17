# 暖通负荷(大设备)数据提取脚本说明

本目录（`config/aidc_hvac_load_5min/scripts/`）保存 AIDC 暖通负荷预测的数据提取与处理脚本。

## 文件职责

| 文件 | 职责 |
|---|---|
| `build_hvac_tables.py` | 依据 `dataset/aidc_load_5min/A1_A2_A3_points/all_ids.xlsx`「暖通负荷(大设备)」sheet 提取 A1/A2/A3 楼暖通大设备点位数据，按设备版本 × route × 楼 输出宽表到 `dataset/aidc_hvac_load_5min/` |
| `analyze_hvac_window.py` | 对 `dataset/aidc_hvac_load_5min/` 两个版本 × A/B 路数据在窗口 2026-07-24 14:00 ~ 2026-07-31 23:55 内做缺失/异常/total_load 缺失影响分析，输出汇总 CSV 与可视化图到 `dataset/aidc_hvac_load_5min/analysis/` |

## 运行方式

从仓库根目录执行：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/build_hvac_tables.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/analyze_hvac_window.py
```

## 提取规则

- `spot_id`：从 `dataset/aidc_load_5min/A1_A2_A3_points/<data_type>/<spot_id>.csv` 读取单点位数据；
- `备注`：设备版本区分 —— `hvac_all_devices`（全部设备）/ `hvac_remove_devices`（剔除「冷冻水二次泵」）；
- `route`：A / B 两路分别成表（`route_A/`、`route_B/`）；
- `data_type`：区分 A1 / A2 / A3 楼。

## 时间网格

- A1：`2025-10-01 00:00:00` ~ `2026-08-17 15:10:00`，5min（92343 行）；
- A3：`2025-10-01 00:00:00` ~ `2026-08-17 15:05:00`，5min（92342 行）；
- A2：`2026-07-24 10:05:00` ~ `2026-08-17 15:05:00`，5min（6973 行）；
- 数据源：基础目录 `<data_type>/`（~2026-07-31）拼接增量目录 `<data_type>_20260801-20260916/`（目录名到 09-16，实际数据止于 2026-08-17 15:05/15:10；与基础数据无重叠时间戳，重复时间戳直接 RAISE）；
- 缺失不填补，保留 NaN；`data.csv` 采用 A1 完整网格，其他楼列在各自有效范围之外为 NaN。

## 输出结构

```
dataset/aidc_hvac_load_5min/
  {hvac_all_devices,hvac_remove_devices}/
    {route_A,route_B}/
      A1_data.csv / A2_data.csv / A3_data.csv   # time + 各 spot_id 列 + total_load(该楼该路点位求和)
      data.csv                                  # time + 三楼全部点位列(加 A1_/A2_/A3_ 前缀) + total_load(三楼该路总负荷)
```

- 求和规则：按行对非空值求和，全部缺失时 `total_load` 为 NaN（`min_count=1`）；
- CSV 编码 `utf-8-sig`，索引列名 `time`。

## 注意事项

- 每楼大设备 36 点：A1/A3 楼 A 路 24 点、B 路 12 点；A2 楼 A 路 17 点、B 路 19 点；剔除二次泵后 A1/A3 为 20/10、A2 为 15/15。
- 源数据中 A1 楼最后一天（2026-07-31）存在规律性缺测、末两个刻度（23:50/23:55）无数据，输出中对应为 NaN，本脚本不填补。
- **进度记录（2026-09-16）**：缺失填充方案已定但暂停实施——点级 `interpolate(method='time')` 填充 spot_id 列 → 重算 total_load → 重建 data.csv，填充位置落审计文件，A2 在 2026-07-24 10:05 之前的存在性缺失不填；待 wangzf 恢复该任务时实现。
