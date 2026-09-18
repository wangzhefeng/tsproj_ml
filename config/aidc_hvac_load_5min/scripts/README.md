# AIDC 负荷数据提取脚本说明

本目录（`config/aidc_hvac_load_5min/scripts/`）保存 AIDC 暖通/IT 负荷预测的数据提取与处理脚本。

32份预测CSV的一对一天气资产由本目录 `prepare_weather.py` 适配共享处理后天气，输出独立 `weather_data/`，不修改目标窗口选择。公共源缺失由项目级 `scripts/build_scenario_weather.py` 统一处理，场景脚本不补值。逐文件路径、source片段及审计见 [场景README](../README.md)。

## 文件职责

| 文件 | 职责 |
|---|---|
| `build_hvac_tables.py` | 从点位源生成 `raw_data/{hvac_all_devices,hvac_remove_devices}/` 原始宽表；目标文件已存在时拒绝覆盖 |
| `build_it_load_tables.py` | 从列头柜源生成 `raw_data/IT_load/` 原始宽表；目标文件已存在时拒绝覆盖 |
| `analyze_hvac_window.py` | 只读 `raw_data/`，分析 2026-07-24 14:00～09-16 23:55 候选范围（内部仍有长缺口，并非完整连续段）；结果写统一 `analysis/` |
| `analyze_it_load.py` | 只读 `raw_data/IT_load/`，结果写统一 `analysis/`，CSV 使用 `IT_load_` 前缀避免与暖通报告重名 |
| `migrate_hvac_data.py` | 一次性迁移旧数据与 IT 分析文件，SHA256 校验；已有迁移记录时仅复核，不重复移动 |
| `impute_hvac_data.py` | 原始点位因果短缺口遮蔽选型、填补、严格总量重算；20 份 CSV 写 `imputed_data/`，审计写 `analysis/imputation/` |
| `select_hvac_windows.py` | 从填补数据筛选近期最长完整自然日段；32 场景写 `forecast_data/`，窗口/有效性/折几何审计写 `analysis/forecast_windows/` |
| `analyze_forecast_data.py` | 只读32份预测数据及点位汇总实测掩码，分析84个文件内数值列；每文件输出完整时序、日内热力图、最大相对跳变局部图及候选CSV到 `analysis/forecast_data_visual/` |

## 运行方式

从仓库根目录执行：

```bash
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/migrate_hvac_data.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/impute_hvac_data.py
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/select_hvac_windows.py
```

三个新入口均支持 `--root <数据根目录>`，默认按脚本位置定位仓库，不依赖 cwd。迁移幂等复核，填补/选窗拒绝覆盖任何已有产物；临时目录计算完成、全部输入哈希复核后才发布。源提取 `build_*` 仅用于新归档，不能用来覆写已迁移 raw。旧 `analyze_*` 是原始数据诊断，图中的双向插值估计不是本次因果填补结果。

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
  raw_data/                                  # 迁移前后字节一致；只读输入
    {hvac_all_devices,hvac_remove_devices}/{route_A,route_B}/
      A1_data.csv / A2_data.csv / A3_data.csv / data.csv
    IT_load/{A1_data,A2_data,A3_data,data}.csv
  imputed_data/                              # 与 raw_data 的20个CSV路径对应
    {hvac_all_devices,hvac_remove_devices}/{route_A,route_B}/...
    IT_load/...
  forecast_data/{hvac_all_devices,hvac_remove_devices}/{route_A,route_B}/
    A1_data.csv / A1_data_with_it.csv
    A2_data.csv / A2_data_with_it.csv
    A3_data.csv / A3_data_with_it.csv
    data.csv / data_with_it.csv
  analysis/                                  # 原始诊断、迁移记录、填补与选窗审计
```

- 原始归档求和规则保持历史事实：非空值求和，全部缺失才为 NaN（`min_count=1`）；不能据此认为总量完整。**填补派生表使用严格求和：任一所需点位仍缺失，总量为 NaN**；
- CSV 编码 `utf-8-sig`，索引列名 `time`。

## 列头柜负荷（IT_load）提取规则

- sheet：「列头柜负荷」，仅用 `data_type`（分楼）与 `spot_id`（点位）；不分 route、不按 `备注` 过滤；
- 时间网格：三楼统一 `2025-10-01 00:00:00` ~ `2026-09-16 23:55:00`，5min（101088 行，已核实三楼基础/增量目录范围一致）；
- 数据源拼接与 RAISE 规则同暖通脚本；
- **不存在点位的明确裁决**：A1 的39列、A3的43列视为楼栋不存在的点位。`../preparation.json` 列出逐楼精确 ID；原始归档保留历史全 NaN 列不改字节，填补派生表排除，不补零。出现清单之外的全空点位或清单内点位出现观测均 RAISE，不自动扩张排除范围；
- `raw_data/IT_load/` 保持 `A1_data.csv`（286 点位）/ `A2_data.csv`（168）/ `A3_data.csv`（304）/ `data.csv`（758 点位）；`imputed_data/IT_load/` 分别保留247/168/261/676点位。均为 `time + 点位列 + total_load`；
- 已知数据特征：源采样时刻不规则（非整 5min 对齐），reindex 后部分网格行所有点位同时缺测（采集侧整段缺口），`total_load` 对应为 NaN，不填补；
- 2026-09-18 经一次性补丁脚本（用后已删除）合并 `20260918/20260918_列头柜831/` 补采导出后（A1/A2 的 8/17~8/31 缺口已填；A3 该段源系统无数据仍缺），total_load 空置率：A1 6.2%、A2 4.2%、A3 8.6%（更新前为 A1 约 9.5%）。

## 注意事项

- 每楼大设备 36 点：A1/A3 楼 A 路 24 点、B 路 12 点；A2 楼 A 路 17 点、B 路 19 点；剔除二次泵后 A1/A3 为 20/10、A2 为 15/15。
- 源数据中 A1 楼最后一天（2026-07-31）存在规律性缺测、末两个刻度（23:50/23:55）无数据，输出中对应为 NaN，本脚本不填补。
- A2 楼增量数据缺口已由 20260918 补采填满：2026-08-17 15:20 ~ 08-31 23:55 覆盖 99.8%（仅余夜间 23:10 规律性单点缺失）；A1 同段覆盖 80.4%（补采文件内部有散布缺测）；A3 补采仅覆盖 08-17 15:20 ~ 08-18 15:20 一天，**08-18 15:25 ~ 09-10 16:50 仍整段缺失（约 24 天）**，09-10 傍晚起恢复，属存在性缺失，不应填充。补采合并已执行完毕（一次性脚本已删除，合并结果保留在各楼 `_20260801-20260916` 增量目录内）。
- 早期双向线性插值方案已被用户本次裁决替代：采用以下仅过去观测的遮蔽选型方案，不使用未来值填补。

## 因果填补与审计合同

- 每个点位只处理两端有记录、连续缺失不超过72个5min槽的完整缺口。73槽及以上整段不填；首尾不外推。填补数值和方法选择均只读取该缺口之前的**原始观测**，不递归借用先前填充值。
- 候选为 `locf`（前一观测）、`past_mean_1h`（前12槽完整观测均值）、`previous_day`（昨日对应时槽）。只比较当前缺口具有完整上下文的候选。
- 遮蔽长度桶为1/3/12/36/72槽，使用能覆盖当前缺口长度的最小桶。历史验证起点每6h一个，取缺口前30天内且已结束、真值与所有被比较候选均完整的段，至少3段。按各段MAE的中位数最小者选择；精确同分按上述候选顺序，样本不足保留NaN并审计。该分数是过去遮蔽选型证据，不是正式预测效果或独立测试误差。
- `analysis/imputation/gaps/<family>/<building>_data.csv` 保存每个缺口位置、状态、方法、各候选分数、样本数、验证起止、原始依赖起止。填充值本身保存在点位CSV；两者共同定位每个修复值。
- `analysis/imputation/masks/` 对应20表，保存原总量是否完整实测、是否经填补完整、`eligibility_known_at`。超长缺口是否闭合是离线资格判断：**数值不使用未来，但整段<=6h的资格直到右侧恢复时间才确定**。边界折须按该时间检查，不能仅凭“单侧方法”宣称全量离线数据能原样在线重放。
- 选型可能读取模型14天训练窗口之外的原始观测，不能宣称完整原始依赖也局限于14天。所有汇总在点位填补之后计算，三楼宽表从分楼点位重建，不独立填补 total_load。

## 预测数据窗口合同

- 近期下界在 `../preparation.json` 显式为2026-07-14，避免最长窗口选到冬季；CLI可显式用 `--recent-start` 改变。每栋楼取两版本×两路严格总量完整性交集，有IT再并入对应IT；按最长完整自然日连续段选择，同长度选较新的段。
- 有IT/无IT独立选窗，不拼接断档；这不是控制变量意义的IT增益对照，后续须对齐测试原点再比较。30天历史包含训练和测试；14+1、stride=1天时30天有16折。短窗口仍生成真实CSV，不足15天标记 `insufficient_14_plus_1`，不伪造折数。
- 分楼CSV只含 `time,hvac_total_load`，有IT增加 `it_total_load`。三楼 `data.csv` 再加 `A1_hvac_total_load,A2_hvac_total_load,A3_hvac_total_load`；`data_with_it.csv` 同时含 `it_total_load,A1_it_total_load,A2_it_total_load,A3_it_total_load`。
- `analysis/forecast_windows/windows.csv` 枚举32个文件、窗口、长度、SHA、结构折数和资格时间安全折数；`folds_14_1.csv` 枚举原点与测试实测行数；`masks/` 独立保存有效性，不污染预测CSV字段。**测试期估计值不是实测真值**，本次未创建预测YAML、未运行模型，也未自动将mask接入运行管线。
- 定向验证：`env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_data_preparation`；常规回归用 `fast`。

## 预测数据异常候选可视化

运行 `env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/analyze_forecast_data.py`，可用 `--root` 指定数据根目录。输出 `analysis/forecast_data_visual/{version}/{route}/{CSV文件名去扩展名}/`，每个输入对应：

- `timeseries.png`：各列独立纵轴的全量5min序列、填补标记和统计候选；不改值、不聚合降采样。
- `daily_heatmap.png`：每列独立色阶，横轴时刻、纵轴日期。
- `largest_jump_zoom.png`：按各列幅度门槛标准化后的最大单步跳变附近±3h，辅助观察局部形态。
- `series_summary.csv`、`candidate_points.csv`、`candidate_segments.csv`：逐列阈值/统计、候选点与连续候选段；根目录另存32文件完整性表、84列汇总、哈希manifest和图片索引README。

复用 `data_process.outlier_process.detect_anomalies` 的只读尖峰检测，不调用清洗入口。尖峰半窗口3槽，幅度至少 `max(实测中位数×10%,1kW)`，并满足 robust z≥6或相邻两端接近；跳变门槛为 `max(6×1.4826×实测相邻差分MAD,实测中位数×10%,1kW)`；低负荷候选≤实测中位数10%；恒值候选为相邻差≤1e-9kW且持续≥12槽。阈值是明示的探索性规则，不是设备物理限值，也不自动清洗。

逐列匹配 `analysis/imputation/masks`，三楼分量使用对应楼的掩码，不拿总量掩码冒充分量。跳变的 `transition_observed` 要求相邻两端均实测；当前点 `observed` 为真并不证明其跳变未涉及填补。IT与分楼分量在多文件中重复出现，候选数按文件列计，不能直接相加当独立物理事件数。居中尖峰使用未来邻域，仅供离线复核，不可直接作为预测特征。

输入和掩码SHA前后必须不变；输出目录存在则拒绝覆盖；先在临时目录生成全部32份，再整体发布。缺失、非有限值、负值和汇总分量关系与统计候选分开汇报；无额定容量信息时不宣称某个高值超过物理上限。
