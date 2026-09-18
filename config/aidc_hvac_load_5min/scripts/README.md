# AIDC 负荷数据提取脚本说明

本目录（`config/aidc_hvac_load_5min/scripts/`）保存 AIDC 暖通/IT 负荷预测的数据提取与处理脚本。

## 文件职责

| 文件 | 职责 |
|---|---|
| `build_hvac_tables.py` | 依据 `dataset/aidc_load_5min/A1_A2_A3_points/all_ids.xlsx`「暖通负荷(大设备)」sheet 提取 A1/A2/A3 楼暖通大设备点位数据，按设备版本 × route × 楼 输出宽表到 `dataset/aidc_hvac_load_5min/` |
| `build_it_load_tables.py` | 依据同 xlsx「列头柜负荷」sheet 提取 A1/A2/A3 楼列头柜（IT 负荷）点位数据，按楼输出宽表到 `dataset/aidc_hvac_load_5min/IT_load/`（不分 route、不做版本过滤） |
| `analyze_hvac_window.py` | 对 `dataset/aidc_hvac_load_5min/` 两个版本 × A/B 路数据在窗口 2026-07-24 14:00 ~ 2026-07-31 23:55 内做缺失/异常/total_load 缺失影响分析，输出汇总 CSV 与可视化图到 `dataset/aidc_hvac_load_5min/analysis/` |

## 运行方式

从仓库根目录执行：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/build_hvac_tables.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/build_it_load_tables.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/analyze_hvac_window.py
```

## 提取规则

- `spot_id`：从 `dataset/aidc_load_5min/A1_A2_A3_points/<data_type>/<spot_id>.csv` 读取单点位数据；
- `备注`：设备版本区分 —— `hvac_all_devices`（全部设备）/ `hvac_remove_devices`（剔除「冷冻水二次泵」）；
- `route`：A / B 两路分别成表（`route_A/`、`route_B/`）；
- `data_type`：区分 A1 / A2 / A3 楼。

## 时间网格

- A1：`2025-10-01 00:00:00` ~ `2026-09-16 23:55:00`，5min（101088 行）；
- A3：`2025-10-01 00:00:00` ~ `2026-09-16 23:55:00`，5min（101088 行）；
- A2：`2026-07-24 10:05:00` ~ `2026-09-16 23:55:00`，5min（15719 行）；
- 数据源：基础目录 `<data_type>/`（~2026-07-31）拼接增量目录 `<data_type>_20260801-20260916/`（2026-09-17 补全至 09-16；与基础数据无重叠时间戳，重复时间戳直接 RAISE）；
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

## 列头柜负荷（IT_load）提取规则

- sheet：「列头柜负荷」，仅用 `data_type`（分楼）与 `spot_id`（点位）；不分 route、不按 `备注` 过滤；
- 时间网格：三楼统一 `2025-10-01 00:00:00` ~ `2026-09-16 23:55:00`，5min（101088 行，已核实三楼基础/增量目录范围一致）；
- 数据源拼接与 RAISE 规则同暖通脚本；
- **sheet 声明但无源文件的点位**：43 个 `spot_id` 在 A1/A3 两楼目录均无 CSV（39 个 `0_10xx` 段被同时登记在 A1 五楼与 A3 四楼、4 个 `50_0_10x` 段属 A3 二楼 YL202-RBB），输出保留为全 NaN 列并打印警告清单，不影响 `total_load` 语义；
- 输出 `dataset/aidc_hvac_load_5min/IT_load/`：`A1_data.csv`（286 点位）/ `A2_data.csv`（168）/ `A3_data.csv`（304）/ `data.csv`（758 点位，列名加 `A1_/A2_/A3_` 前缀），均为 `time + 点位列 + total_load`；
- 已知数据特征：源采样时刻不规则（非整 5min 对齐），reindex 后部分网格行所有点位同时缺测（采集侧整段缺口），`total_load` 对应为 NaN，不填补；
- 2026-09-18 经一次性补丁脚本（用后已删除）合并 `20260918/20260918_列头柜831/` 补采导出后（A1/A2 的 8/17~8/31 缺口已填；A3 该段源系统无数据仍缺），total_load 空置率：A1 6.2%、A2 4.2%、A3 8.6%（更新前为 A1 约 9.5%）。

## 注意事项

- 每楼大设备 36 点：A1/A3 楼 A 路 24 点、B 路 12 点；A2 楼 A 路 17 点、B 路 19 点；剔除二次泵后 A1/A3 为 20/10、A2 为 15/15。
- 源数据中 A1 楼最后一天（2026-07-31）存在规律性缺测、末两个刻度（23:50/23:55）无数据，输出中对应为 NaN，本脚本不填补。
- A2 楼增量数据缺口已由 20260918 补采填满：2026-08-17 15:20 ~ 08-31 23:55 覆盖 99.8%（仅余夜间 23:10 规律性单点缺失）；A1 同段覆盖 80.4%（补采文件内部有散布缺测）；A3 补采仅覆盖 08-17 15:20 ~ 08-18 15:20 一天，**08-18 15:25 ~ 08-31 仍整段缺失**，属存在性缺失，不应填充。补采合并已执行完毕（一次性脚本已删除，合并结果保留在各楼 `_20260801-20260916` 增量目录内）。
- **进度记录（2026-09-16）**：缺失填充方案已定但暂停实施——点级 `interpolate(method='time')` 填充 spot_id 列 → 重算 total_load → 重建 data.csv，填充位置落审计文件，A2 在 2026-07-24 10:05 之前的存在性缺失不填；待 wangzf 恢复该任务时实现。
