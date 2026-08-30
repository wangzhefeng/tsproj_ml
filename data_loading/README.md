# data_loading 模块说明

`data_loading/` 是**数据构造与组织层**（流水线第 1 阶段）：canonical 主链的唯一数据通路。2026-08-30 自 `model_forecasting/data/` 迁为顶层包。

## 文件职责

| 文件 | 职责 |
|---|---|
| `registry.py` | `SourceRegistry`：文件读取 + schema 校验 + 严格 as-of vintage（`available_at <= forecast_origin`，缺失即 RAISE）；验证帧缓存 |
| `information_set.py` | `InformationSetRequest` / `MaterializedInformationSet`：信息集物化与行定位（含 `row_position_lookup` O(1) 缓存） |
| `providers.py` | observed_past 显式 provider / known_future trajectory / static / endogenous future 等数据提供者 |

## 边界

- 只依赖 `model_forecasting.specs`（配置合同）与 `utils`，不依赖任何下游阶段包；
- 离线数据准备（填补/清洗/事件检测）在 `data_process/`，发生在「进模型之前」，见流水线数据契约（AGENTS.md）。
