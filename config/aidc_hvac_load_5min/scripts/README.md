# AIDC HVAC 单版本数据准备

所有活动路径不再含数据版本层级。原始数据位于 `raw_data/`；旧版配置、数据、结果和专属流程已退役。

## 当前入口

- `imputed_data/prepare_data.py`：保留的 A1 固定窗口基线构建器（包含原有预测输入构建；本次不运行）。
- `imputed_data/gap_filling.py`：仅过去原观测候选与遮蔽评分。默认不能填补即报错；显式 `on_unfillable='keep'` 才保留 NaN 并返回逐缺口原因。
- `imputed_data/prepare_buildings.py`：A2/A3 全原始时间范围缺失处理，生成两个阶段的分楼表及三楼汇总，不处理预测输入。
- `analysis/analyze_building_events.py`：独立筛查 A2/A3 事件候选，只输出证据，不自动改值。
- `outlier_remove_data/prepare_building_events.py`：执行 `building_events.json` 中人工批准的五个夏季区间，只替换 A2/A3 暖通异常修正点位表和三楼汇总。
- `outlier_remove_data/adjust_demand_response.py`：保留 A1 已确认事件的历史构建及通用估计核心。
- `forecast_data/`：保留模型配置、水温流量与设备切换特征入口；本次仅迁移路径，不重算现有预测表。

## 处理合同

1. 历史准备批次保留了当时 raw 原字节；授权的外部观测补充见下节。下游 A1 原有点位、固定窗口及 IT 204 点子集不变。
2. A2 暖通从 2026-07-24 10:05 开始；A3 暖通和 A2/A3 IT 保留原始全时段。
3. 末值、过去完整1h均值、过去日片段为候选；只使用过去原观测，前30天已结束遮蔽块评分，至少3个共同有效块。
4. 超24天、边缘或校准不足缺口保留 NaN 并审计；不补零，不递归把旧估计作为新参考。A3全空IT列仍保留，不推断为真实零。
5. 暖通两设备口径共享点位只估计一次，去二次泵口径投影复用。
6. 主业务路径（不含hvac_switch）的imputed_data为填补基线；outlier_remove_data为已批准事件的常态反事实估计。事件外保持基线；IT不做需求响应修正。
7. 三楼 data.csv 从分楼点位重建，沿用对应 raw 总表网格；任一所需点位缺失则 total_load 为 NaN。A1 IT是固定子集，因此总IT也不是全园区完整IT口径。
8. 候选曲线不证明真实调度，批准仅授权离线反事实研究；估计不是真值，不宣称严格在线。

## 外部原始观测补充

入口：`raw_data/supplement_raw.py`，显式提供 --data、--source、--audit；默认仅准备，--apply 才发布。审计目录必须不存在。

- 范围 2025-10-01 至 2026-10-23 23:55；按楼栋＋测点匹配，不改变现有路线和列成员。
- 只补原缺失和时间轴外新观测，不插值、不覆盖已有观测。任一重叠值冲突则暂缓整个测点；比较容差 rtol=1e-9、atol=1e-6。重复外源冲突或负功率也暂缓。
- 原始汇总及分机组表同步投影；raw 的 total_load 仍为 min_count=1 部分和，不能当作全点完整负荷。
- 完整备份在 analysis/raw_supplement_20261023/before/raw_data；manifest.json 记录前后、外源和下游 SHA，points.csv 记录逐点决策及新增量。异常发布回滚；不保证进程被强杀或并发读者下的跨文件原子性。
- imputed_data、outlier_remove_data、forecast_data 不重建，仍为补充前历史快照。旧审计源 SHA 不改写；直接拿新 raw 检验旧资产来源会失败，应对照本次完整备份。不得把旧源清单更新为新 SHA 以冒充重建。

## hvac_switch 三楼点位表

入口：`imputed_data/prepare_switch_buildings.py`。只扩展清洗/填补命名空间，不构建预测特征或事件状态。

- A2/A3各新增六套设备×两设备口径的分套表，分套表仍仅time＋点位列；全设备6点，去二次泵5点。
- 两阶段的每种设备口径、每条路线均提供A1_data.csv、A2_data.csv、A3_data.csv与三楼data.csv；分楼/三楼表含严格total_load。
- 切换清洗只将负功率、无限值置NaN，保留真实启停及需求响应，不读取常态反事实修正表。填补使用过去原观测；无法可靠估计时保留NaN，不从估计值推断设备状态。
- A2/A3保留各自原始全时段；A1既有分套表原字节保留（清洗为全时段，填补仍为原固定窗口）。三楼汇总沿用raw总表网格，窗外或缺点位时严格总量为NaN。
- 本次新增80份业务CSV（每阶段24份分套＋16份分楼/三楼汇总）。审计在 `analysis/hvac_switch/building_extension/`，包含源/旧资产/产物SHA、缺口方法与资格时间、invalid/observed/imputed/unresolved掩码。

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/imputed_data/prepare_switch_buildings.py
```

## 执行

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/imputed_data/prepare_buildings.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/analysis/analyze_building_events.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/outlier_remove_data/prepare_building_events.py
```

已有发布产物拒绝重复构建。事件修正入口是显式批准后的替换步骤，发布前预检源SHA并保留临时回滚副本；不承诺并发读者跨目录原子性。

审计位于 `analysis/building_extension/`：逐缺口原因、逐点 observed/imputed/unresolved 掩码、冻结源及产物SHA、候选事件与已批准修正的评分和逐点前后值。observed/imputed/unresolved掩码对应填补基线；异常修正版的来源须另叠加event_corrections中的逐点方法与前后值。旧生成器SHA保留为生成时事实，路径迁移不伪装成重新运行。

天气使用共享 processed 资产，绑定场景名为 `aidc_hvac_load_5min`，清单在 `analysis/weather_binding/manifest.csv`，绑定两种设备口径的 `forecast_data/{devices}/route_AB/A1/data.csv`。

## A1_all 配置路径修复（2026-11-19）

- `A1_all` 是配置场景名，不是当前数据目录。现有六组配置、naive/A1_all 及生成器使用 `forecast_data/{devices}/route_AB/A1/data_dr_adjusted.csv`，naive/A1_all_old 使用同目录的 `data.csv`；窗口、目标、IT 204点子集和场景名不变。
- 历史预测表构建器及配方中的 forecast 路径同步迁移；配方 SHA 值保留。没有重建 CSV、重新训练或迁移已有模型结果。数据路径属于 canonical payload，后续运行须按新身份产出，不能假定旧 bundle 身份不变。
- `analysis/forecast_windows/{devices}/A1_all/mask.csv` 是仍存在的审计路径，不能跟随预测数据目录一起改名。历史审计清单保留生成时事实，不改写成新构建。
- 原始数据已补充且温度/流量表已改为分楼命名。旧重建流程的原始输入 SHA/文件名及历史审计不代表当前 raw；本次只修复现有预测资产的路径引用，不能据此直接重跑历史数据准备。应使用对应批次备份复核，或另行建立新数据配方。
