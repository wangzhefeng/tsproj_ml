# AIDC 暖通数据准备脚本

## 目录约定

脚本按**产物阶段**归类，再区分通用流程与版本专属流程。未出现的目录表示没有该版本专属实现，不建空壳。

```text
scripts/
  preparation_paths.py                   # 五类数据产物的版本路径合同
  raw_data/                              # 共享原始提取、历史归档迁移
    build_hvac_tables.py
    build_it_load_tables.py
    migrate_hvac_data.py
  outlier_remove_data/outlier_detection.py # 共享纯检测函数：不填补、不写数据
  imputed_data/impute_hvac_data.py         # 共享过去观测填补算法及表级处理
  forecast_data/
    forecast_schema.py                   # 双路字段与点位来源合同
    select_hvac_windows.py                # 版本显式的选窗、掩码和折审计
    build_model_configs.py               # 当前模型配置目录，绑定data_v1
  analysis/
    analyze_hvac_window.py               # 原始诊断，保存于data_v1/raw_diagnostics
    analyze_it_load.py
    analyze_forecast_data.py             # 版本显式的预测输入可视化
  weather_data/prepare_weather.py         # 共享天气源→每版逐目标天气适配
  v1/
    imputed_data/prepare_data.py          # 旧流程：先填补raw
    outlier_remove_data/
      clean_hvac_outliers.py              # 旧流程第一轮孤立点清洗及重填补
      clean_hvac_redboxes.py              # 旧流程第二轮红框修正及范围内旧估计重算
      redbox_cleaning.json
  v2/
    outlier_remove_data/
      clean_hvac_data.py                  # 新流程：合并检测，异常只置NaN
      cleaning.json                      # 冻结的孤立检测范围与原红框事件
    imputed_data/prepare_data.py          # 新流程入口：屏蔽→统一填补→发布
  v3/
    preparation.json                    # A1固定点位子集、窗口与原始输入SHA
    imputed_data/gap_filling.py          # 仅过去观测的短/长缺口遮蔽选型
    imputed_data/prepare_data.py         # A1原始→填补→双路合计固定窗口，不清洗
    outlier_remove_data/adjust_demand_response.py # 原v3不变，同目录输出需求响应修正CSV
    outlier_remove_data/demand_response.json     # 经确认的四次事件与输入SHA
    forecast_data/build_model_configs.py # 六组A1_all配置，消费data_v3，独立发布
    forecast_data/prepare_flow_temp.py   # 水温流量筛选/填补/基础特征与双版本拼接
    forecast_data/verify_flow_temp.py    # 独立回读数值、逐缺口评分与来源证据
    forecast_data/flow_temp.json         # 固定时间窗及两源表/四目标表SHA
```

## v3 套组运行与切换特征（独立新增）

入口 `v3/forecast_data/prepare_switch.py`，数值核心 `switch_features.py`。消费A1南北六套原始点位，跨A/B目录按模组识别；两设备口径的共享点位只处理一次。输出命名空间 `hvac_switch`：`outlier_remove_data/data_v3/hvac_switch/<设备口径>/<路线>/A1_<套组>_data.csv` 保留全时段清洗点位；`imputed_data/data_v3/hvac_switch/` 同构保存现有预测窗口的填补点位。窗前原观测只用于校准及历史特征，不能将窗前未填补历史称为完整观测。

- 采集异常只自动屏蔽负功率/无限值，不按幅度、低值、恒值清除真实运行状态；需求响应保留。实际无此类违规时报告零修正，不为“清洗”制造改值。填补沿用v3过去观测遮蔽选型，不足证据失败；逐点记录估计mask、资格时间与评分。
- 冷机状态使用on≥20/off≤1的迟滞阈值和连续3槽确认；这是一版显式规则，不宣称最优。缺测重置未知，恢复不记启停；补值只用于功率特征，不用于确认设备切换。状态/计数/运行小时均只取已确认历史，时长记录左截断，滚动7/30天小时数同时给共同覆盖率。
- 需求响应作为离线事件标注，保存已确认四次区间及抑制/恢复阶段，重叠启停/接替标为响应相关，不冒称常规轮换。人工事件标签不进入最终预测字段；在线可计算的通用启动/停止/接替历史保留两类事件，不能事后用人工标签改写过去特征。
- `forecast_data/data_v3/hvac_switch/features.csv` 是两口径、原版/去响应版共用的真实设备状态特征。各 `A1_all/` 新增 `data_with_flow_temp_switch.csv` 与 `data_dr_adjusted_with_flow_temp_switch.csv`，原80个业务字段逐格保持，只增加 `hvac_sw_` 列。不从反事实功率推断真实切换，不改旧CSV、模型、天气绑定。
- `analysis/data_v3/hvac_switch/` 保存冻结输入、特征字典、事件表、离线需求响应标签、逐点填补和每列资格时间。功率估计有离线闭合资格约束；下游只能作为observed-past，按预测原点检查资格并显式提供provider，绝非known-future。任何已有新目标均拒绝覆盖。

当前配方 `v3/forecast_data/switch.json` 已冻结：输出窗口46656行、144列新增特征，四份最终表各为time＋原80列＋144列。清洗和填补各12份分套表；原始全时段缺失不冒充已完整填补，填补产物仅覆盖现有预测窗口。`events.csv`包含窗前历史事件，统计目标窗口须按confirmed_at筛选；离线需求响应归因不是已知发布时间证据。

```bash
# 已有产物请只核验，不重复构建；首次输入冻结使用 --freeze-recipe，已有配方拒绝覆盖。
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v3/forecast_data/prepare_switch.py --verify-only
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_switch_features
```

## v3 水温流量特征准备

入口 `v3/forecast_data/prepare_flow_temp.py`，冻结配方 `v3/forecast_data/flow_temp.json`。只消费 `raw_data/hvac_flow_temp/{data_temp,data_flow}.csv` 的58个显式点位，生成76个带 `hvac_ft_` 前缀的基础变量；不纳入蓄冷增强、疑似华氏温度、常量异常或长恒值回水点位，不推测单位转换，不按幅度清除正常切换或负流量。

- `outlier_remove_data/data_v3/hvac_flow_temp/` 保存筛选后未填补点位（不做额外异常修正）；`imputed_data/data_v3/hvac_flow_temp/` 保存同列填补表。复用 `v3/imputed_data/gap_filling.py` 的仅过去原观测候选与历史遮蔽MAE选型；不足校准报错，不回退到全历史统计量。
- `forecast_data/data_v3/hvac_flow_temp/features.csv` 保存58原始量＋18衍生量，衍生量从填补点位重算。两设备口径 `A1_all/` 各新增 `data_with_flow_temp.csv`、`data_dr_adjusted_with_flow_temp.csv`，精确一对一拼接原版/需求响应修正版；原文件、目标、IT均不改，水系统特征不做反事实响应修正。
- `analysis/data_v3/hvac_flow_temp/` 保存点位选择/排除、逐缺口方法与评分、点位和特征的补值掩码/资格时间、来源及代码SHA、发布清单。长缺口闭合资格依赖未来，估计不是实测；CSV是离线历史资产，不能把当时水系统实测当作未来已知协变量。时间lag/rolling由后续原点编译负责，本入口不改模型或天气绑定。
- 首次构建先在临时目录生成并回读全部产物，核验原值/公式/拼接及来源SHA后发布；任一命名目标已存在则拒绝覆盖。多文件发布不保证并发原子性，发布期间不得启动消费者。`--verify-only` 独立回读正式资产，核对全部掩码及逐缺口数值/历史评分；不重写结果。

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v3/forecast_data/prepare_flow_temp.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v3/forecast_data/prepare_flow_temp.py --verify-only
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_flow_temp
```

## v3 需求响应修正派生

原 `forecast_data/data_v3/<设备口径>/A1_all/data.csv` 保持字节不变；新增同目录 `data_dr_adjusted.csv`，不创建 data_v4、不切换模型配置。需求响应是真实运行事件，修正序列是用于常态负荷研究的反事实估计，不是采集纠错或实测真值。

- 入口：`v3/outlier_remove_data/adjust_demand_response.py`；固定事件与源 SHA 配方：同目录 `demand_response.json`。仅处理已确认的 2026-07-10、07-16、07-17、08-06 四次事件，包含低谷和恢复高峰。
- 区间采用 `[start, end_exclusive)`：两设备口径 AB 同时回到各自事件前1h中位数 ±10%，并连续保持6槽，取该稳定段首槽作为结束边界。边界使用 v3 曲线（可能含旧补值），属于离线人工审核规则，不证明调度指令时刻或严格在线可得性。
- 从原始暖通点位屏蔽全部已确认事件后，只重估事件区间（包括原有估计）；方法与数值仅用事件前原观测。复用 v3 的末值、过去完整1h均值、过去日片段候选，按前30天、6h步长、与事件等长的完整真值遮蔽块评分，至少3个共同有效块，取块MAE中位数最小者；校准不足拒绝发布，不恢复响应原值。事件外旧补值不传播重算。
- 全设备点位修正一次，去二次泵口径复用共享点位，严格重算 A/B/AB；IT 及事件外全部业务数值保持不变。原点位表、原mask、天气、旧分析均不覆盖。
- `outlier_remove_data/data_v3/demand_response/<设备口径>/<路线>/A1_data.csv` 保存修正点位表；`analysis/data_v3/demand_response/` 仅保存逐格前后值、逐点方法和遮蔽分数、事件及独立来源mask、源/产物SHA、核验结果与前后对比图。新mask不自动接入现有模型评分；天气不生成新绑定，模型仍使用原文件。
- 旧 `analysis/data_v3/demand_response/point_tables/` 由同目录脚本 `v3/outlier_remove_data/relocate_point_tables.py` 原字节迁移；`analysis/data_v3/demand_response/path_migration/` 保存迁移前manifest及路径/SHA映射。历史builder SHA保持生成时事实，不改成迁移后代码身份；不重算CSV或图片。
- 天气v3入口只发现 `*/A1_all/data.csv`，不会因同目录出现具名派生而自动扩展旧天气绑定。7月10日恢复后存在持续的A/B负荷重新分配；仅过去参考不能恢复真实切换过程，分路修正末端允许保留阶跃，不为平滑修改范围外值。

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v3/outlier_remove_data/adjust_demand_response.py
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_demand_response
```

首次发布前完成全部校验；已有任一目标或审计目录则拒绝覆盖。跨目录发布不保证并发原子性，构建时不得启动消费者。

## 数据布局与版本含义

### 当前模型输入绑定

当前六组A1_all的160份物理配置已将目标及相关历史协变量路径切换到 `data_dr_adjusted.csv`；未自动使用水温流量或切换增强表，不在本次提交中重跑模型。naive的 `A1_all/` 两口径同样使用去响应版，`A1_all_old/` 两口径保留原 `data.csv` 对照。

`v3/forecast_data/build_model_configs.py` 的默认生成口径仍为原 `data.csv`，不是上述手动切换后的配置快照；不要盲目重建覆盖当前物理配置。新增增强表只生成数据，没有自动修改模型特征合同。

### v3：A1 双路合计、固定 IT 子集（构建入口见下文）

- 只消费共享 raw 的 A1；不清洗异常，不继承 v1/v2 估计，不改变旧资产。
- 用户确认固定 IT 子集：排除原39个全空点位，以及43个窗口前段无观测的晚出现点位；保留204个点位，特征命名 `it_subset_load`，不冒充完整全楼 IT。
- 两种设备口径各一份 `forecast_data/data_v3/<设备口径>/A1_all/data.csv`；目标 `hvac_total_load_AB` 严格等于 A+B，同时提供 A/B 历史与 IT 子集历史。点位级 imputed 保留来源路线，预测目录不分路线。
- 全部业务输入固定为2026-04-08 00:00至2026-09-16 23:55。仅填补选型可读取起点前30天原始历史；业务输入及天气不扩窗。
- 只补 NaN：候选为末值、过去完整1h均值、过去一天对应片段（长于一天则重复）；后者不得读取缺口内部。每个缺口使用前30天已结束遮蔽块（6h步长），至少3个共同有效块，按块MAE中位数选择。短桶验证真值完整，超过6h的桶要求至少80%原始真值，审计覆盖率；不足证据直接失败，不静默兜底。
- 共享暖通点位只填补一次，再投影至两口径。未缺失值逐格保持；IT内部长缺口估计不是实测，补值来源与依赖保留于 `analysis/data_v3/`。
- 六组配置为 `<组>/A1_all/<设备口径>/<模型>_<方法>.yaml`，原生ETS独立命名`ets.yaml`，共160份。纯目标baseline与add_context各44份，日期节假日/天气/IT/分路四组各18份；后三组不含日期节假日。普通组90天训练＋1天预测，context150天训练＋1天预测，保持相同lag。仅历史回测，不启动正式训练或覆盖结果。发布器以A1_all命名空间隔离旧576份配置，拒绝旧v3嵌套路径、清单外文件和不同内容覆盖。

v3数据及A1_all配置构建命令见[场景README](../README.md#a1_all固定数据两路合计)。旧三楼填补、自动选窗和旧可视化CLI仅接受data_v1/v2，不把这些合同套到v3；公共天气适配支持data_v3。

以 `dataset/aidc_hvac_load_5min/` 为根：

- `raw_data/`：20份共享原始基础宽表，另含36份按套组拆出的点位表；不按准备版本复制。归属修正仅按下述显式迁移流程备份、重分组，禁止常规准备入口覆盖。
- `outlier_remove_data/data_v1/{isolated_v1,redbox_past_v2}/`：旧流程的两阶段准备根，每个内部仍含自己的填补数据和审计。`redbox_past_v2`是原正式v3更名后的资产，**不等于新流程data_v2**。
- `imputed_data/data_v1/`：旧流程初始缺失填补结果，未做异常清洗。旧正式预测输入仍绑定上述红框准备根，不误用初始填补mask。
- `outlier_remove_data/data_v2/`：新流程屏蔽异常后、尚未填补的20份点位表；IT不清洗。`total_load`沿原始表部分求和合同，仅供阶段一致性核对，不代表完整总量。
- `imputed_data/data_v2/`：从屏蔽后的原观测统一填补，20份表，严格总量。
- `forecast_data/{data_v1,data_v2}/`：各版32份业务输入；不是模型预测结果。
- `analysis/{data_v1,data_v2}/`：各版的清洗/填补/选窗/可视化证据。原有分析目录、历史archive和`TOOD.md`均归入data_v1，待办内容不变。
- `weather_data/{data_v1,data_v2}/`：各版逐目标天气及路径/SHA绑定；共享处理后天气源不变。

现有576份模型YAML读取data_v1。仅迁移输入路径，不调整特征、训练窗口或预测策略，不自动切换到data_v2，不重跑或删除既有模型结果。路径属于配置payload时身份可能变化，不把旧模型结果改名冒充新配置结果。

旧资产搬迁的原始元数据及路径映射保存在 `analysis/data_v1/preparation/version_migration.json`。当前消费路径和必要哈希链已迁移；历史archive、生成时的inputs_sha256、代码SHA仍表示原执行事实，不能当成迁移后代码重新运行的证明。

## 两条流程

### v1：保留当前处理行为

`raw → 初始填补 → isolated_v1 → redbox_past_v2 → 选窗`。

孤立点处理后重算受影响表；红框阶段只覆盖选中原异常及框内旧估计，框外旧估计保持，无法替换时保留父版本值。原始及父版本不覆盖。当前资产已存在，下面首次构建入口会明确拒绝覆盖：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v1/imputed_data/prepare_data.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v1/outlier_remove_data/clean_hvac_outliers.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v1/outlier_remove_data/clean_hvac_redboxes.py
```

从空根重建v1需在初始填补后先用选窗入口 `--data-version data_v1 --preparation-root .` 建立初始窗口，再执行孤立点清洗；之后选窗显式绑定 `outlier_remove_data/data_v1/isolated_v1`，才能执行红框入口；最后绑定 `outlier_remove_data/data_v1/redbox_past_v2`。已有正式资产不得倒切到父版本覆盖结果。

### v2：异常处理前置

`raw → 孤立点标记 → 屏蔽后的原观测上红框判定 → 统一NaN缺口 → 统一填补 → 严格总量 → 选窗`。

- 孤立点检测范围从迁移前预测窗口冻结到 `v2/outlier_remove_data/cleaning.json`，运行时不依赖data_v1或预测窗口，不自动扩大原授权范围。
- 同一物理点位在全设备/去二次泵两口径中去重，统一屏蔽。判定时不读历史填充值。
- 所有历史缺口重新选型和填补，包括框外受异常污染的补值；不继承data_v1估计。
- 异常与原始缺失合并后按整段长度判定。超过72槽、首尾或验证不足，保持NaN，不恢复异常原值或旧估计；窗口可能因有效性改变而变化。
- IT不做异常清洗，按相同原始源及填补合同重新生成；应独立核验其数值与v1一致。
- 审计分别保存原始异常标记、候选/筛选证据、每个缺口的选型及来源。异常产生的估计不当作实测真值。

首次完整构建（已有data_v2拒绝覆盖）：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/v2/imputed_data/prepare_data.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/forecast_data/select_hvac_windows.py --data-version data_v2
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/weather_data/prepare_weather.py --data-version data_v2
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/analysis/analyze_forecast_data.py --data-version data_v2
```

选窗与可视化只有显式 `--replace` 才替换指定版本，禁止把替换另一版本当成默认行为。准备过程暂存完成后发布，不承诺多个目录对并发读者原子切换；真实数据重建时不要同时启动消费进程。天气适配独立于原始共享源修补，只消费已处理公共天气，不重写其他场景。

## 原始提取合同

### 套组归属修正

`raw_data/equipment_mapping.py` 读取当前 `all_ids.xlsx`，按楼栋＋“东西南北”校验18套设备，每套为冷水机组1、冷却塔2、冷却水泵1、冷冻水一次泵1、冷冻水二次泵1。保留原 `route` 作为点位电气路线，另用 `group_route` 表示套组路线：A1南2整套归B、南3整套归A（以冷水机组为准）；其他套组必须内部路线一致。分路总量此后表示套组归属总量，不再等同于电气馈线总量。

归属修正只搬移点位列及其来源证据，不重新检测异常或估计补值，不重跑模型。分套表为 `raw_data/<设备口径>/route_<A或B>/<楼栋>_<方向编号>_data.csv`，只含time和点位列，不含total_load；全设备6列、去二次泵5列。旧数据/审计保留备份；历史模型结果保留旧口径，不能直接作为修正后数据的验证结果。

迁移入口 `raw_data/regroup_hvac_equipment.py`：`build`建立独立暂存，`continue`恢复下游重分组，`finalize`在独立核验后冻结发布清单，`publish`核对源SHA并备份后发布。证据入口 `raw_data/verify_hvac_regroup.py`：`refresh`更新审计归属和预测图、`bind`刷新活动来源身份并保留旧SHA、`verify`逐点精确比对及核对36表。默认工作根 `.hermes/plans/hvac-equipment-regroup/`，已有工作根拒绝再次build；`publish`只执行一次，旧文件保留于工作根backup及backup_config。原始诊断入口 `analysis/analyze_hvac_window.py --root <暂存根>` 支持发布前重绘。不对历史archive或模型结果回写。

- 源为 `dataset/aidc_load_5min/A1_A2_A3_points/`，按清单的 `(data_type, spot_id)` 读取，基础目录与增量目录拼接；时间戳重复直接RAISE，不keep-last。
- `hvac_all_devices`/`hvac_remove_devices`区分全设备与剔除冷冻水二次泵；A/B分路、A1/A2/A3分楼，IT不分路。
- A1/A3暖通与IT网格为2025-10-01至2026-09-16；A2暖通从2026-07-24 10:05开始。5min网格，不存在的槽保留NaN。
- 原始总量使用`sum(min_count=1)`，不能以非空总量证明设备完整；填补后总量要求全部所需点位非空，三楼表从分楼点位重建，不独立填补total_load。
- `preparation.json`明确排除A1的39个、A3的43个不存在IT点位。raw保留全空列，派生排除、不补零；名单外全空列或名单内出现观测均RAISE。
- 原始CSV编码utf-8-sig，时间列time。源提取已有文件拒绝覆盖；原始诊断的双向插值仅供离线对比，不是模型填补方法。

## 检测与填补算法

- **孤立点**：前后各3槽原始观测完整稳定；汇总偏差至少max(局部中位数10%,1kW,6×1.4826×MAD)，邻域极差不超过偏差25%；恰好一个点位解释汇总偏差80%～120%。多点同步变化、持续阶跃、多槽平台不自动清除。
- **红框**：指定楼栋/路/事件区间，事件前12槽至少8个可信观测，中位数基线，阈值max(基线绝对值10%,1kW,6×1.4826×MAD)；先前异常不能作后续基线。人工圈选是授权范围，不证明传感器故障。
- **填补**：候选locf、过去1h完整观测均值、昨日同时槽；每个缺口前30天、6h步长的已结束遮蔽块，至少3段有效共同样本，按块MAE中位数最小选择，同分按候选顺序。长度桶1/3/12/36/72槽，不递归借用补值。
- **可得性**：整段不超过6h的资格在右侧恢复时才确认；孤立点判定还依赖后侧15min，其影响传播到相关填补和校准段审计。红框为人工离线选择，缺少真实决策发布时间，整个流程不可宣称严格在线/实盘可用。

## 选窗与模型输入合同

近期下界默认2026-07-14。每楼取两设备口径×两路线的严格总量完整性交集，有IT再并入对应IT；选择最长完整自然日段，同长取较新者，不拼接断档。有IT/无IT独立选窗，跨组评价需配对测试日期。

`hvac_dual_route_v1`字段合同不随数据版本变化：分楼含A/B总量，可选IT；三楼含A/B/AB总量、六个分楼暖通分量，可选三楼与分楼IT。两路线输入数值相同，目录区分预测目标。另一条路、分楼量和AB合计只可作历史信息，禁止测试时刻同刻实测泄漏。

`analysis/<data_version>/forecast_windows/`保存32份窗口、14+1折几何与逐列observed/eligibility掩码。结构折数不等于原点资格安全折数；估计目标不是实测评分真值。现有模型仍按data_v1原14/7天训练配置，数据准备不自动更改模型评分掩码。

可视化每输入三图：全量时序、日内热力图、最大相对跳变±3h；另存缺失/非有限/负值/总分不一致与统计候选。统计尖峰、跳变、低负荷、恒值仅用于复核，不自动清洗或作预测特征。每列从真实来源mask读取，跳变两端都需实测才能标为实测变化。

## 验证

```bash
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_versioned_preparation
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_data_preparation
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_weather_hvac_windows
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_model_configs
env -u PYTHONPATH .venv/bin/python tests/run_suite.py fast
```

验收必须覆盖真实两版资产、源哈希、非异常原观测保真、共享点一致、严格总量、mask、天气目标绑定及预测输入完整性；测试通过不代表模型效果提升。历史报告和模型结果不以新数据名重新标注。
