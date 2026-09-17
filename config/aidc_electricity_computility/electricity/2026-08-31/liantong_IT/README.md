# 联通 IT 2026 年 8 月功率数据准备

## 算力离线处理合同

`liantong_computility_process.py` 独立处理 `aidc_comp_liantong_5min/{training,inference}`，不修改目标、天气、模型 YAML 或原始 CSV。

- 用户确认 `uid` 为 Job ID；同 Job 同时间多个值来自不同实例且无重复抓取，全部保留（相同值也不去重），可加指标求和后不再乘副本数。
- GPU 功率单位 W，输出 sum/mean 转 kW；内存/显存保留原始单位，不宣称 Byte 或 GiB。
- 用户确认全部无记录（含部分指标无记录）按零处理；此授权只针对本批场景，不推广至其他场景。空集合利用率均值/标准差按零表示并保留计数；计数不冒充真实历史运行副本数。
- 利用率超过 1 的值在派生层截断为 1，逐样本记录原值、新值、Job、文件、时间和序列位置；负值、非有限值、非法结构/时间直接报错。原始值不改。
- 比例指标输出已上报实例的 sample_mean，不冒充容量加权利用率；无稳定实例身份不跨指标强行配对。静态元数据不广播为历史特征。
- 特征只含同时间截面聚合；lag/rolling/recent_state/same_slot 留给预测 compiler 按折截窗计算。算力是 observed_past，不是 known_future；没有实际接入模型或赋予部署资格。
- 独立 training/inference/history 特征和 quality 表均保留整月 5min 网格。分析拼接表以派生电力目标左连接、一对一验证，目标值原样保留。8.19 仍是估计目标。
- 本阶段不产生跨训练/推理功率总和及占比：不同 Job ID 不足以证明硬件计量不重叠。后续确认物理范围后再添加。

运行（默认从真实数据全量生成，路径与 cwd 无关）：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/liantong_computility_process.py
```

可指定 `--source-dir/--target-csv/--output-dir`；非默认测试网格使用 `--start/--end`。已有输出默认拒绝覆盖，明确重建用 `--overwrite`。程序验证所有输入后再发布；逐输出哈希与源哈希保存于 `computility_processing_audit.json`。不会自动补跑模型。

`--output-dir` 表示 liantong_IT 场景根目录。除拼接表 `power_computility_5min_<日期段>.csv` 留在根目录外，以下算力输出全部位于其 `aidc_comp_liantong_5min/` 子目录；日期段为 `20260801_20260831`：

| 文件 | 内容 |
| --- | --- |
| `computility_training_5min_<日期段>.csv` | time + 27 个训练侧特征 |
| `computility_inference_5min_<日期段>.csv` | time + 27 个推理侧特征 |
| `computility_history_5min_<日期段>.csv` | time + 54 个两组特征，无跨组重复计量假设 |
| `computility_quality_5min_<日期段>.csv` | time + 每项指标 sample_count/no_record，共 36 列质量数据 |
| `power_computility_5min_<日期段>.csv` | time/value + 54 个特征；仅为分析拼接表 |
| `computility_processing_audit.json` | 输入/输出哈希、处理口径、逐条修正与发布限制 |
| `training_job.csv` | 训练侧每个原始实例采样点的长表 |
| `inference_job.csv` | 推理侧每个原始实例采样点的长表 |

两份 Job 长表字段为 `time,job_id,metric,raw_value,processed_value,source_file,source_row,sample_position,is_corrected,epoch_seconds`。保留输入文件/行/序列顺序（不做全表时间排序），重复时间和相同值均保留，不补造缺失采样点。`source_file` 相对各自 training/inference 源目录；`source_row` 是含表头的 CSV 逻辑记录序号（首条数据为 2），`sample_position` 从 0 起，不代表实例 ID。`raw_value` 保留原始数值文本，`processed_value` 仅应用利用率上界处理，两列单位相同（GPU 功率仍为 W）。分批写临时文件，全部输入验证通过后发布，避免将全量长表堆入内存。长表没有稳定实例身份，不可由序列位置推断跨时间实例连续性。

每项指标包含已上报 Job 数；利用率包含 sample_mean/sample_std（样本标准差），可加量包含 sum/sample_mean。Job 数和均值只针对该指标已上报对象；无记录按零处理并不构造不存在的实例 ID，也不将其他指标的 Job 数当该指标分母。最终 state、instances、ready_replicas 不入模。

实现采用固定场景合同和 CLI 路径参数，不另建可任意切换语义的准备 YAML。更换缺失/单位/聚合合同须修改文档和处理版本，不能悄悄改变存量结果。

本次全量执行处理 18 个 CSV、15,231,457 个样本；五份聚合/质量/拼接 CSV 各为 8,928 行且数值有限。长表 `training_job.csv` 为 14,715,681 行，`inference_job.csv` 为 515,776 行，逐采样核对原始值文本、处理值、Job、时间及来源位置通过；没有去重或补造采样。修正 7 条利用率（memory_util 2 条、gpu_memory_util 5 条），全部在 training；数值解析的浮点舍入不算异常修正。独立按原始序列重算全部指标的 sum/mean/std、Job 数和采样数，与输出在浮点容差内一致；源文件、目标及原五份输出哈希不变，拼接目标逐值精确一致。定向测试 `integration --match test_liantong_computility_process` 5 项通过，`fast` 289 项通过。这是离线准备验收，不表示模型接入或预测效果改善。

本目录按八组保存 72 份物理模型 YAML（原四组 37 份、三个算力组各 9 份、`accuracy_ablation` 8 份）；`liantong_power_process.py` 为独立离线处理入口，不修改模型配置。

`accuracy_ablation/` 是以 baseline Direct pointwise 为基座的独立精度候选组：原样对照、近期状态、复杂度控制、L2/Huber 损失、特征筛选、Ridge 和 XGBoost。保持原数据与严格 14 天历史/17 折回测合同；不加入算力、测点或分解，不改原有六组。配置与实验边界见 [accuracy_ablation/README.md](accuracy_ablation/README.md)。仅配置验证不代表正式回测或精度提升。

## 执行

从项目根目录运行：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/liantong_power_process.py
```

默认输入 `dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/aidc_load_liantong_5min/`，仅读取 `data-*.xlsx` 和 `联通IT测点筛选表.xlsx`，不读取 Excel 锁文件。可用 `--source-dir` / `--output-dir` 指定测试路径；覆盖输出路径不会改变默认输入路径。

## 数据合同

- 参考表 `SignalID` 为白名单，ID 按字符串原样保留；未列入参考表的信号排除并审计。
- 按 `(RoomID, DevAssestID)` 分组，设备首次出现顺序决定 `point_1_value` 等编号。每设备只能为一个总有功功率信号，或完整的 A/B/C 三相信号定义。
- 全部功率单位为 kW；301 机房参考表单位空白按用户确认视为 kW。不计算电量、不乘时间间隔。
- 时间为本地墙上时间，向下取整到 5 分钟网格；输出从 `2026-08-01 00:00:00` 到 `2026-08-31 23:55:00`。不四舍五入、不插值、不前后填充。
- 三相和全设备总计均忽略空值求和；全部为空时保持空值（`min_count=1`），真实零值保留。部分相位缺失时设备列是可用相位之和，不代表完整三相总功率；部分设备缺失时 `value` 同样是部分和。
- 同一信号同一时间格重复、白名单信号非法时间/非数值/非有限值、参考定义异常均报错，不静默覆盖。

## 输出

场景根目录：`dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/`。`df_power.csv` 留在根目录；下列映射表和处理审计放在 `aidc_load_liantong_5min/` 子目录，与原始 Excel 同级。

- `df_power.csv`：`time,point_1_value,...,point_200_value,value`，8,928 行，空值写空字段。
- `point_mapping.csv`：每个原始信号一行，共 264 行；`SignalID → point_column` 为多对一映射，含 `RoomID,DevAssestID,DeviceName,SignalName,unit`。源表字段实际名为 `SignalID`，不是 `SingleID`。
- `processing_audit.json`：输入数量、被排除信号、逐信号缺失数量、部分相位/设备时间格数量及处理规则。

已知原始缺口：8 月 19 日整天缺失；8 月 10 日 `303-RPP-D1-B-列头柜` 缺 A 相；8 月 30 日 `303-RPP-H2-A-列头柜` 缺 A 相。按白名单复核还发现 `303-RPP-H1-B-列头柜` 的 A 相额外缺 288 格，以及 8 月 25 日 `304-RPP-E1-B-列头柜` 总功率整日缺失。表外信号不能替代这些缺口。最终缺失以审计输出为准。

此宽表是离线数据产物，不直接进入 canonical runtime；模型输入由下述准备脚本生成。

## 8 月模型数据准备（修订合同）

`liantong_august_prepare.py` 从上述原始 `df_power.csv` 和共享天气宽表
`dataset/shared/weather/extracted/actual/weather_in_20250101_20260831.csv` 生成独立派生文件，不覆盖原始文件。

- 保留 8 月全部真实时间戳。只填充 8 月 19 日整日缺口，不删除日期、不平移后续功率或天气。
- 8.19 每个 5min 时刻采用 8.01—8.18 同一时刻的算术均值，必须有完整的 18 天历史，不读取 8.20 及以后数据，不再进行候选方法选择。
- 派生目标只输出 `time,value`；填好后正常参与特征、训练和评分，不加特殊掩码或特征。估计来源、方法和范围记入 `.meta.json`，不冒充补获的实测；8.19 的误差是相对补全序列的误差。
- 严格 14 天约束准备后的模型输入；补值的上游血缘包含此前 18 天，两者不能混称。
- 天气保留真实日期；九配置统一使用辐射、温度、相对湿度、风速、气压、降雨六项，实测列 `rt_ssr/rt_tt2/cal_rh/rt_ws10/rt_ps/rt_rain` 对应 `pred_ssrd/pred_tt2/pred_rh/pred_ws10/pred_ps/pred_rain`。保留湿度派生与小时内 hold，不压缩时间。共享源中气压缺口已补齐；源旁 `.six_features_repair.json` 的哈希及再分析替代性质随天气 metadata 保存。夜间辐射缺测置零属于显式离线填补，审计记录位置。
- 新输出：`target_power_5min_20260801_20260831.csv`、`weather_history_5min_20260801_20260831.csv`，各附 `.meta.json`。天气 history 覆盖整月：训练用实测列，测试预测用预报列；当前无真正未来任务，不配置或生成 future。旧分段天气产物已停用、暂留，旧压缩目标 CSV 及 metadata 已按确认删除。

运行：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/liantong_august_prepare.py
```

## 数据归档与算力关联分析

- 根目录保留 `df_power.csv`、`target_power_5min_20260801_20260831.csv`、`power_computility_5min_20260801_20260831.csv` 和天气文件。目标 `.meta.json` 移入 `aidc_load_liantong_5min/`；天气 metadata 仍与天气 CSV 同级。迁移只改变存储位置，不重新计算数据，不改既有审计中的历史血缘。
- `liantong_computility_analysis.py` 读取拼接表及电力原表，以 `value`（kW）为唯一目标。严格核对完整 5min 时间轴、数值、特征名单和非补值目标的一致性；不把质量字段当作候选特征。
- 主分析排除原表目标缺失的标签时刻（本批为 8.19），保留全样本敏感性对照；不改变模型当前的补值/计分规则。先在完整网格 shift/diff，再掩码，禁止删日后跨缺口错配。
- 输出全部特征的同期 Pearson/Spearman、一阶差分相关、非零子集相关、日内去均值相关；输出 `corr(X[t-lag], y[t])` 及双方差分相关，lag 为 0/1/3/6/12/36/72/144/288/576 个 5min 步。同期和 288/576 步另输出分日相关及稳定性。正 lag 表示算力先于目标；288/576 步满足当前次日整日预测的历史边界，采集延迟仍待核实。短 lag 仅为关系诊断，不冒充次日可用特征。
- 输出特征之间绝对 Spearman ≥ 0.95 的冗余对、同期和 288 步排名及 Markdown 报告，保存在 `aidc_comp_liantong_5min/feature_analysis/`。常数/样本不足的相关值为空并标记，不写成零；不作独立同分布假设下的显著性宣称。全月探索不等于严格 14 天回测、因果关系或增量预测收益，不据此自动改 YAML。

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/liantong_computility_analysis.py
```

已有分析产物默认拒绝覆盖，重建使用 `--overwrite`。输出记录输入哈希与分析口径；后续特征选择应限制在每折训练窗口，再与仅负荷/天气基线做消融。

## 九策略严格历史窗口

`add_weather/lgbm_*.yaml` 为原有九配置的原样语义迁移，仅输出组路径改变。九策略分别覆盖 Direct pointwise（含 horizon 特征的独立变体）、Direct、Recursive、DirRec、DIRMO、RecMO、DirRecMO、MIMO，使用 lag、rolling/expanding、datetime、holiday 和 weather，不做目标分解。

- `validation.train_history_steps: 4032`：每折先截取连续 14 天 5min 数据，再构造所有训练和预测特征；expanding 从该折起点重置。
- 预测长度 288 点，每日滚动；回测日期 8.15—8.31，共 17 折，包含正常计分的 8.19。
- 特征预热和标签均位于窗口内。Direct/DIRMO/MIMO 的有效监督原点数为 1728，其余变体为 1729；这是 lag 锚点差异，不用扩大历史强行统一。
- 配置只支持 `--backtest-only`，完整生命周期、final fit 和 bundle 导出显式拒绝。其他场景未启用 `train_history_steps` 时保持原行为。
- 验收包括九策略合成 LightGBM 回测及窗口外扰动不变性、真实数据首折/8.19/末折特征设计探针；不表示九份正式配置已经完成大规模拟合，也不提供部署资格或实测缺失日误差证明。

## 算力消融组

`add_training_compute/`、`add_inference_compute/` 和 `add_training_inference_compute/` 各含 baseline 的九份 `lgbm_*.yaml` 对照。前两个单因素组仅添加独立算力 source、observed-past lag 和组输出路径，baseline 的目标、节假日、datetime、rolling/expanding、模型参数、策略与验证设置全部保留。ETS 不消费这些外生指标，仅保留原 baseline 对照，不生成名不副实的 ETS 加算力配置。

| 组 | 明确入模的基础列 |
| --- | --- |
| `add_training_compute` | `training_cpu_util_sample_mean`、`training_cpu_util_sample_std`、`training_gpu_power_usage_sum_kw` |
| `add_inference_compute` | `inference_gpu_memory_amount_sum_raw`、`inference_gpu_memory_util_sample_mean`、`inference_memory_amount_sum_raw`、`inference_memory_total_sample_mean_raw` |
| `add_training_inference_compute` | 上述全部 7 列，另加目标 `value` 的 `recent_state`（原点前含原点 6/12/36 点 level/mean/std/diff/slope，与 baseline_opt 同口径） |

`add_training_inference_compute` 是训练+推理算力与近期状态的组合候选：除两组算力 source、7 列 lag、`recent_state` 和组输出路径外，其余字段与 baseline 逐字段一致；不加 same_slot、seasonal_baseline 或参数调整，不能将组合效果归因于单一增量。`recent_state` 只作用于目标 `value`，不展开到算力列。

- 分别读取 `aidc_comp_liantong_5min/computility_training_5min_20260801_20260831.csv` 和 `computility_inference_5min_20260801_20260831.csv`，不读取带目标的分析拼接表或 Job 长表；未声明列不会入模。内存/显存保留原始单位，GPU 功率为 kW。
- 每列 `observed_past_lags: [288, 576]`：单因素组分别增加 6/8 个、组合组增加 14 个算力 lag 列，不增加算力 rolling、算力近期状态或特征选择器。CPU 的 std 指同刻实例间离散度，不是沿时间的波动。
- `availability: source_time` 为采集可得性假设，`provider: persistence` 满足 observed-past 显式 provider 合同；由于最小 lag 等于 horizon，当前设计只消费原点前真实历史，不使用 persistence 外推。不能据此证明实际采集零延迟。
- 保留基线的历史锚点：普通 Direct、MIMO、DIRMO 的 `align_to_target: false` 将 lag 冻结在预测原点；Direct pointwise 两变体及递归家族以目标时刻回溯。不能把原点锚定的 288 步 lag 描述成预测时刻的昨日同槽。每个变体只与同名 baseline 配对，避免把策略锚点差异归因为算力。
- 保留 5min、288 点 horizon、14 天原始历史、17 折和 backtest-only 边界。三组仅为待验证候选；全月关联分析不是独立泛化证据，不宣称预测改善，不自动运行正式回测或导出部署模型。

验收：全场景 `scripts/check_model_configs.py 'config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/**/*.yaml'` 为 72/72 通过；`tests/run_suite.py integration --match test_liantong_` 为 33 项通过。新增测试覆盖全部 27 份算力配置的真实训练设计、首折/补值标签日/末折预测第 1/144/288 步值及其 proof，并核对窗口外/未来算力扰动不变性；不代表正式回测、final fit 或部署验收。

## 四组与原生 ETS

| 目录 | 物理 YAML 数量 | 特征和模型合同 |
| --- | --- | --- |
| `baseline/` | 10 | 原九策略逐一去天气，无天气 source；另有独立 `ets.yaml` |
| `baseline_opt/` | 9 | 无天气优化候选，同槽/近期状态与残差通路 |
| `add_weather/` | 9 | 迁移前模型、特征、验证语义保持不变 |
| `add_weather_opt/` | 9 | 有天气优化候选；四个块输出策略额外使用完整块天气摘要 |

优化共有项：3/7 天同槽 mean/std；原点前含原点 6/12/36 点的 level/mean/std/diff/slope（level 为末值，mean 为窗口平均水平，std 为样本标准差，slope 为首末差除以步间隔数）。同槽特征锚定目标时刻，近期状态始终锚定原点。7 天同槽均值从每个训练原点的 as-of 历史独立计算，标签减基线、预测加回，不是只添加基线特征；不与目标变换混用。

优化配置保留 7 天预热，使用较少日期字段和保守 LightGBM 参数。所有参数仅是未验证候选，不能据此宣称误差改善。天气优化的 MIMO/DIRMO/RecMO/DirRecMO 对实际调用块内六项天气计算 mean/min/max，固定 schema；实测/预报切换与原源合同一致。本次不核实、校正或下载天气。

ETS 直接从每折完整 4032 点、5min 规则历史估计；每日周期 288，独立预测未来 288 点。默认候选 ANA/AAA/AAdA、BIC 选择、heuristic 初始化、每候选最多 300 次迭代；记录候选收敛/失败证据，全失败 RAISE，不静默换模型。仅借鉴 M5 ES_bu 自动指数平滑思想，不宣称复现零售层级方法。

四组保持同样 17 折日期、严格 14+1 天和评分规则，输出路径分别追加组名；旧结果完全保留，不自动迁移或清除。原始目标和 8.19 同槽填补逻辑未修改。所有 YAML 仍限定 `--backtest-only`，本次没有运行正式配置。

存量结果已另按用户明确授权完成分组：原 Direct、RecMO、MIMO 三个结果目录整体迁入 `results/results_test/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/add_weather/`，fingerprint 与当前配置一致，72个原有文件逐文件 SHA-256 不变。`baseline/`、`baseline_opt/`、`add_weather_opt/` 目前只有空分组目录，没有正式结果；没有补跑模型。结果根目录的 `README.md` 保存本次整理汇总。

静态审计（项目根目录）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_model_configs.py 'config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/**/*.yaml'
```

可重复测试入口与覆盖范围见项目根 `tests/README.md`。`.hermes/plans/liantong-ets-completion.md` 与后续 review 收口记录是本地实施证据，不随仓库分发。
