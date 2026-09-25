# docs/dataset — 数据资产文档

`dataset/` 保存模型配置引用的数据资产与离线分析产物。2026-09-25 场景收敛后，活动配置仅引用 `aidc_load_15min_short/` 与 `shared/` 下资产；其余场景目录数据按用户裁决原样保留，不随场景退役删除。模型读取路径以 schema-2 YAML 的 `data.sources` 为准，并由 `scripts/audit_runtime_assets.py` 强制检查存在性。

> 本文自 `dataset/README.md` 迁入 docs 并纳入版本控制；`dataset/` 整体仍被 Git 忽略，目录内其余分析报告类 README（hvac 分析、事件标签报告等）属数据产物一部分，保留原位；`dataset/shared/weather/README.md` 已迁入 [`shared-weather.md`](shared-weather.md)。

## 章节

| 章节 | 内容 |
|---|---|
| [shared-weather.md](shared-weather.md) | 共享天气资产（raw/normalized/manifest/extracted 各层）合同与使用边界 |

主要场景：

- `aidc_load_15min_daily/rolling/short`：15min 负荷；
- `aidc_load_month/`：1D 负荷 mean；
- `aidc_power_month/freq_1day|freq_1month`：日/月电量 sum；
- `aidc_ess_selfuse_load/`：储能站用电；
- `aidc_electricity_computility/`：电力与算力融合；
- `ETT-small/`：保留数据样例，不属于现役模型配置。

数据约束：

- 时间列必须可解析、有序并覆盖请求频率；唯一性按序列键、时间及 source 版本合同判断，不能把多序列时间重复当成重复资产；
- known-future 必须精确覆盖 horizon，并满足 available-at 合同；
- 缺失/异常修复发生在 `data_process/` 离线阶段；
- 不允许为通过资产审计伪造未来标签；经用户裁决，缺少权威日期类型的配置已移除对应 source。

## 目录与产物分类

- `aidc_load_5min/` 保存 5min 负荷及关键点位整理数据；`aidc_points/` 保存点位与空间层级来源资料。
- `shared/` 保存跨场景共享资产；是否入模仍取决于具体 YAML 的 source 声明，不自动拼接整个目录。
- `tsproj_ml-weather-backup/` 保存天气迁移前的原件及独立恢复验证副本，不属于活动模型资产，不能递归扫描后用作训练输入。快照内 manifest 保留创建时的历史位置，当前路径与搬迁核验见 `.hermes/plans/weather-backup-verification.json`；本目录与业务数据位于同一仓库，不提供仓库整体丢失时的异地容灾。
- `shared/weather/` 为正在实施的统一天气资产根；raw 保留来源/地点/版本和原字节，normalized 与 manifest 固定传递哈希，模型配方仍在 YAML。合同见 [`../packages/weather_generator.md`](../packages/weather_generator.md)。旧场景副本尚未迁移，不得提前删除。
- `shared/holidays/` 统一保存日历、节假日数据文件，包括 `scripts/export_chinese_holiday_csv.py` 导出的 CSV；生成实现仍属于 `data_loading/calendar_generator/`。导出需显式指定 `--output`，已有配置路径不随目录约定自动迁移。
- 当前完整日历导出为 `shared/holidays/chinese_holiday_20250101_20261231.csv`，日频覆盖2025-01-01至2026-12-31；旧2025-10-01起文件保留。`generator: chinese_holiday` 按请求日期计算，不依赖这些离线CSV的起止日期。
- `outlier_analysis/`、`peak_valley_analysis/`、`periodicity_analysis/`、`event_label_features/` 是离线分析或派生产物。图片和报告存在不等于对应 CSV 已通过建模可得性检查。
- `*.aggregate.json` 等 sidecar 记录聚合参数和溯源；更换输入或聚合规则时需与派生 CSV 一起核对，不能只改文件名。
- `load` 表示负荷功率 mean，`power` 表示电量 sum；`aidc_load_month/` 实际为日频，不能按目录名推断月频。

## 使用流程与边界

1. 由场景脚本或 `data_process/` 离线准备数据，保留原始资产、派生路径及审计信息。
2. 在 canonical YAML 中显式声明列角色与路径；`available_at` 必须满足预测原点可得性，允许版本记录的 source 按自身键和版本合同校验。
3. 先做资产审计，再由 registry/compiler 校验实际请求窗口。文件存在性通过不代表所有回测折或预测期覆盖通过。
4. `feat_` 为本频率 trailing 特征，`xf_`/`xr_` 分别表示跨频率/跨路由特征；`lbl_` 事件标签可能使用未来信息，只能离线分析，不直接作在线预测特征。

模型 bundle、回测与预测结果写入 `results/`，不写回本目录覆盖原始数据。当前 `dataset/` 整体被 Git 忽略。
