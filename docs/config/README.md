# docs/config — 配置文档中心

> 一句话版本：只接受 canonical YAML（严格字段合同，无版本字段），未知字段一律 RAISE；活动场景仅 `aidc_load_15min_short`（仅 LightGBM，171 份单模型 YAML）。

`config/` 承载全部活动模型 YAML（canonical 严格字段合同）与数据工具 YAML。经 2026-09-25 场景收敛裁决，活动预测场景只保留 `aidc_load_15min_short`（模型测试场景），且仅保留 LightGBM 配置；其余预测场景（`aidc_load_15min_daily/rolling`、`aidc_load_month`、`aidc_power_month`、`aidc_ess_selfuse_load`、`aidc_electricity_computility`、`hongtaiyang_cesuan`）及 short 内非 LightGBM 配置、`add_ensemble/` 组整体退役，内容由 Git 保留，不计入活动集。纯数据准备目录 `aidc_load_5min`、`aidc_hvac_load_5min` 亦已删除（Git 溯源），`dataset/` 下对应数据资产保留。

## 章节

| 章节 | 内容 | 何时读 |
|---|---|---|
| [schema.md](schema.md) | 唯一 schema、顶层结构、transformation 嵌套 schema、加载严格性与 fingerprint | 写/改 YAML、改 loader |
| [geometry.md](geometry.md) | 严格原始历史窗口、fixed-step/calendar-month 字段、时间边界 | 改时间几何/回测窗口 |
| [data-roles.md](data-roles.md) | 数据角色、外生来源、低频与中国节假日约定 | 接新数据源/低频场景 |
| [weather.md](weather.md) | 气象文件与列分流、天气证据边界、天气生成合同 | 动天气特征/资产 |
| [seasonal-features.md](seasonal-features.md) | `same_slot`/`recent_state`/`block_weather`、`seasonal_baseline` 残差基线 | 用高级 as-of 变换 |

## 场景数据备注

- 模型测试场景 `aidc_load_15min_short` 的场景定位、特征分组与数据通路见 [`docs/scenarios/aidc_load_15min_short/模型测试说明.md`](../scenarios/aidc_load_15min_short/模型测试说明.md)。
