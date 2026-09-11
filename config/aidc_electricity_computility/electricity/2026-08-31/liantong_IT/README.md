# 联通 IT 2026 年 8 月功率数据准备

本目录按四组保存 37 份物理模型 YAML；`liantong_power_process.py` 为独立离线处理入口，不修改模型配置。

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

输出目录：`dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/`。

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

## 九策略严格历史窗口

`add_weather/lgbm_*.yaml` 为原有九配置的原样语义迁移，仅输出组路径改变。九策略分别覆盖 Direct pointwise（含 horizon 特征的独立变体）、Direct、Recursive、DirRec、DIRMO、RecMO、DirRecMO、MIMO，使用 lag、rolling/expanding、datetime、holiday 和 weather，不做目标分解。

- `validation.train_history_steps: 4032`：每折先截取连续 14 天 5min 数据，再构造所有训练和预测特征；expanding 从该折起点重置。
- 预测长度 288 点，每日滚动；回测日期 8.15—8.31，共 17 折，包含正常计分的 8.19。
- 特征预热和标签均位于窗口内。Direct/DIRMO/MIMO 的有效监督原点数为 1728，其余变体为 1729；这是 lag 锚点差异，不用扩大历史强行统一。
- 配置只支持 `--backtest-only`，完整生命周期、final fit 和 bundle 导出显式拒绝。其他场景未启用 `train_history_steps` 时保持原行为。
- 验收包括九策略合成 LightGBM 回测及窗口外扰动不变性、真实数据首折/8.19/末折特征设计探针；不表示九份正式配置已经完成大规模拟合，也不提供部署资格或实测缺失日误差证明。

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
