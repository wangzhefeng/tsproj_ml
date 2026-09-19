# AIDC 暖通 5min 场景

数据提取、因果填补与预测窗口合同见 [scripts/README.md](scripts/README.md)。本场景包含四组共 **576 份物理模型 YAML**。当前预测数据已切换至红框过去观测清洗版本；**本轮清洗没有重跑模型，既有回测结果不代表新数据版本**。

## 模型配置矩阵

```text
<baseline | add_weather | add_endogenous_it | add_endogenous_route>/
  <hvac_all_devices | hvac_remove_devices>/
    <route_A | route_B>/
      <A1 | A2 | A3 | ALL>/
        lgbm_<方法>.yaml
```

每组144份，共4组×2设备版本×2路线×4楼栋×9方法。ALL对应三楼汇总，不是A/B联合预测；route_A仅预测`hvac_total_load_A`，route_B仅预测`hvac_total_load_B`。结果的`scenario_subpath`镜像配置组/设备版本/路线/楼栋，写正式`results/`，不复用其他组结果。

九方法沿用 `config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/baseline/` 的 LightGBM 标准模板：`direct`、`direct-pointwise`、`direct-pointwise-horizon`、`recursive`、`mimo`、`recmo`、`dirrec`、`dirmo`、`dirrecmo`。三类Direct是布局变体；七种策略不重复计数。块方法保留`output_chunk_length: 24`（5min下2小时），`target_adapter: independent`、`params: {}`；不带ETS、Ridge/XGBoost消融、优化组、目标分解或融合。

| 组 | 文件选择与入模信息 |
|---|---|
| `baseline` | 普通CSV，只投影本路目标；不读取其他负荷作为特征 |
| `add_weather` | 普通CSV＋对应weather source的六项天气，训练实测、历史测试预测预报 |
| `add_endogenous_it` | `_with_it.csv`，在本路目标外加入`it_total_load`历史 |
| `add_endogenous_route` | 普通CSV，加入另一路历史；仅ALL额外加入已有`hvac_total_load_AB`历史 |

所有组默认有10项时间戳派生特征，以及generated `chinese_holiday`的`is_holiday/next_holiday_days`。其余三组只增加各自点名变量，IT/跨路组不夹带天气；未声明的CSV列不入模。楼栋不生成AB合计列，不把三楼分量自动加入ALL特征。

### 训练窗口与回测几何

频率5min、每折预测下一完整自然日288点、stride=288、point。使用每份文件现有完整历史范围，训练窗口按原始数据严格截断，lag/rolling/expanding均不能借窗外数据。用户所称15/8天窗口是14/7天训练＋1天预测，而不是累计测试天数。

| 目标文件 | 总历史天数 | 每折训练天数 | 每折预测天数 | 每份配置回测折数 |
|---|---:|---:|---:|---:|
| `A1_data.csv` | 58 | 14 | 1 | 44 |
| `A1_data_with_it.csv` | 22 | 14 | 1 | 8 |
| `A2_data.csv` | 54 | 14 | 1 | 40 |
| `A2_data_with_it.csv` | 47 | 14 | 1 | 33 |
| `A3_data.csv` | 35 | 7 | 1 | 28 |
| `A3_data_with_it.csv` | 16 | 7 | 1 | 9 |
| `data.csv` | 24 | 7 | 1 | 17 |
| `data_with_it.csv` | 14 | 7 | 1 | 7 |

- 14天训练：目标lag为1至7天，rolling为1/2/4/7天，沿用联通标准模板。
- 7天训练：7天lag加1天标签无法形成训练样本；显式采用目标lag 1/2/3天、rolling 1/2天。保留mean/std/min/max及expanding mean/std；这是短窗配置选择，不是实测最优结论。
- IT/跨路协变量统一lag 1/2天，`observed_past + provider: persistence`。这些safe-lag在当前九配置的设计请求中只读原点之前的真值，provider仅为显式合同，不把未来IT/peer实测当已知。Direct/MIMO/DIRMO冻结历史锚点与其他策略的目标时间锚点沿用模板，不能将所有lag都解释为目标日同槽。
- `train_window_steps = train_history_steps - minimum_history_rows(config) - 288 + 1`；该值是监督原点数，不是原始训练点数。`history_steps = train_window_steps + 288 + (fold_count - 1) × 288`，为最后训练监督原点到首个测试原点之间的288步距离保留候选范围，以保证标签不重叠。训练标签可结束于测试原点，测试从下一点开始，**不额外空置或跳过一天**；按生产函数核验，不手填近似值。
- `forecast_origin`是文件最后一条已知点，用于评估截止；最后一折从此前一天23:55预测文件最后一天。它不表示已经有截止点之后的天气预报。

### 生成与执行边界

```bash
# 全部候选先通过typed parse、compiler构造与真实资产网格/声明列有限性预检，再写入
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/build_model_configs.py
# 只核验生成规则和物理文件一致，不写入、不训练
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/build_model_configs.py --check
# 获得运行授权后，单份历史回测示例；此次配置任务未执行此命令
env -u PYTHONPATH .venv/bin/python run.py --config-yaml config/aidc_hvac_load_5min/baseline/hvac_all_devices/route_A/A1/lgbm_direct-pointwise.yaml --backtest-only
```

生成器只创建缺少文件；相同内容不重写，已有不同内容或清单外YAML直接报错，不提供强制覆盖/删除。重建前仍需审阅模板变化。所有物理YAML自包含，不依赖运行时读取联通模板。

**当前严格原始历史窗口合同只支持`--backtest-only`，不是完整训练/未来预测/bundle配置。** “预测未来一天”在本次指每个历史原点的下一天。真正未来天气尚未提供；不为通过未来入口删除严格训练窗口字段或以实测回填预报。

IT组与非IT组使用独立历史范围，不能把全期评分差异直接归因为IT增益，后续比较应配对共同测试日期。现有离线补值按普通数值参与训练和评分，未将来源mask接入评分过滤；整段缺口资格与天气发布时间假设保留既有审计，不宣称严格在线回放或实盘可部署。

## A/B双路预测输入

当前`forecast_data/`绑定`outlier_remove_data/redbox_past_v3/`。`redbox_cleaning.json`将用户八张图的红框落为人工事件区间，`scripts/clean_hvac_redboxes.py`在区间内按点位筛选，并对受异常污染的既有缺失补值一并重估。检测基线只取事件前1小时原始观测；替换及候选方法评分只用缺口前原始观测，不用后侧数值、不递归使用新补值。人工红框本身是离线复核，缺口长度资格也在闭合后才确定，因此不宣称整套流程严格在线可得。

物理点位×时刻实际改变3950格，两设备版本内合计7402格；投影/求和传播至32份预测CSV的10448格。非选中点位、IT、原始raw与原imputed、旧isolated_v1及原两处修正均保留。预测时间窗、行数、A/B副本及总分关系保持；六个点位-事件组合因事件前1小时原始观测不足未做原观测离群判断，记录在screening.csv，不强判异常。

新准备根的`analysis/outliers/`保存逐格修正、判定/补值依据、8张前后对比图和验收报告。旧预测、选窗审计、可视化和天气完整归档至`analysis/archive/pre_redbox_past_v3/`；32份天气数值不变，仅刷新目标SHA绑定。`redbox_past_v2`是未发布的中间候选，正式生效版本以选窗manifest为准。新补值是估计值，当前模型评分不自动排除这些点；旧模型结果保留但未按新数据验证。

32个CSV保持原目录及时间窗，均同时含 `hvac_total_load_A`、`hvac_total_load_B`；路线目录区分预测目标而非输入覆盖范围。`route_A`的目标是A列，`route_B`的目标是B列，记录在选窗manifest的`target_column`。`data*.csv`另含六个楼栋×路线分量，以及 `hvac_total_load_AB`（三楼两路总暖通）；IT字段保持不分路。具体字段及严格求和合同见脚本README。

选窗重建只组装当前清单绑定的准备版本，不在选窗时清洗。单槽孤立异常的上游准备入口为`scripts/clean_hvac_outliers.py`，输出独立`outlier_remove_data/isolated_v1/`，原始raw/imputed保留；判定、来源掩码和离线可得性合同见[脚本说明](scripts/README.md#孤立异常清洗版本)。另一条路、楼栋分量与AB合计仅是历史信息，不能把测试期真实值作为未来已知特征。清洗、新增列或改名会改变target SHA，因此天气值即使不变，也必须重新执行本场景天气适配刷新元数据绑定。

## 每个预测文件配套一份天气

天气分两步构建（从仓库根运行；共享数据未变化时只执行第二步）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/build_scenario_weather.py --output dataset/shared/weather/processed/weather_hourly.csv
env -u PYTHONPATH .venv/bin/python config/aidc_hvac_load_5min/scripts/prepare_weather.py
```

以 `dataset/aidc_hvac_load_5min/` 为数据根目录：

```text
forecast_data/<设备版本>/<路线>/<目标名>.csv
weather_data/<设备版本>/<路线>/<目标名>/
  weather_history_5min_<开始日期>_<结束日期>.csv
  weather_history_5min_<开始日期>_<结束日期>.meta.json
  weather.source.yaml
weather_data/manifest.csv
```

- 两个设备版本 `hvac_all_devices/hvac_remove_devices` × 两条路线 `route_A/route_B` × 八个目标文件，共 **32 对**。不把天气混入 `forecast_data/`；每个目标独立配套，即使两份天气值相同也不合并。
- 直接读取每份目标 CSV 的 `time`，要求非空、唯一、升序、连续且对齐的 5min 网格；天气的 `ts` 与其逐行一致，不取所有目标的统一交集，不裁剪原目标文件。
- 共享天气小时值在原生小时内 hold；9 月 16 日 23:00 的小时值覆盖到 23:55。不用 hold 越过缺失小时。
- 此场景使用目标实际窗口；其他场景的 `HISTORY_END = 2026-08-31 23:59:59` 留在各自处理入口，不扩大其训练/回测窗口。公共脚本不再包含预测场景或固定窗口。
- `manifest.csv` 列出每份目标、天气文件、source 片段、精确起止时间、行数和天气 SHA256；metadata 绑定目标哈希、共享分片哈希、原修补 sidecar、共享处理后CSV及metadata哈希，以及窗口内离线插值审计。场景入口只读取共享资产，不重复补缺。

当前每类在两个设备版本和两条路线下各有四份。开始均为 00:00，结束均为 23:55：

| 目标 CSV | 开始日期 | 结束日期 | 每份天气行数 |
|---|---|---|---:|
| `A1_data.csv` | 2026-07-21 | 2026-09-16 | 16704 |
| `A1_data_with_it.csv` | 2026-08-04 | 2026-08-25 | 6336 |
| `A2_data.csv` | 2026-07-25 | 2026-09-16 | 15552 |
| `A2_data_with_it.csv` | 2026-08-01 | 2026-09-16 | 13536 |
| `A3_data.csv` | 2026-07-14 | 2026-08-17 | 10080 |
| `A3_data_with_it.csv` | 2026-08-02 | 2026-08-17 | 4608 |
| `data.csv` | 2026-07-25 | 2026-08-17 | 6912 |
| `data_with_it.csv` | 2026-08-04 | 2026-08-17 | 4032 |

## Weather source 片段

每个 `weather.source.yaml` 包含 `data.sources` 下的单个 `weather` 声明，复制该 source 项到后续 canonical 模型配置的 sources 列表中，**不要覆盖已有 target/IT source**。片段不是独立模型配置，不能直接传给 `run.py --config-yaml`。`history_path` 相对仓库根目录。

- `source_type: file`、`time_col: ts`、`availability: forecast_origin`。
- 六项模型面列为 `known_future`；对应 `pred_` 物理列声明为 `ignored`，通过 `inference_columns` 映射读取：

| 变量 | 历史训练列 | 历史测试预测列 |
|---|---|---|
| 温度 | `rt_tt2` | `pred_tt2` |
| 相对湿度 | `cal_rh` | `pred_rh` |
| 辐射 | `rt_ssr` | `pred_ssrd` |
| 风速 | `rt_ws10` | `pred_ws10` |
| 气压 | `rt_ps` | `pred_ps` |
| 降雨 | `rt_rain` | `pred_rain` |

没有 `future_path`：这里只提供历史评估资产，真正未来预测需要另行提供预报。`forecast_origin` 是已声明可得性假设，**不是供应商发布时间或真实 ex-ante 证据**；不生成 `available_at`，不修改共享分片。

## 经授权的共享源补值

共享源及既有缺口策略处理后，9 月 16 日仍存在长缺口。经用户裁决，以下修补已统一在公共脚本中完成，输出共享 `processed/weather_hourly.csv`，不回写 `extracted/`，不重跑旧repair脚本。场景只消费同一处理结果，不再维护HVAC专属补值函数：

- `rt_tt2`、`rt_dt`：16:00–22:00，15:00/23:00 为双端锚点。
- `rt_ws10`、`rt_rain`：16:00–20:00，15:00/21:00 为双端锚点。
- 只填原缺失单元格；温度/露点补后按原 Magnus–Tetens 公式派生缺失 `cal_rh`。不修改非缺失值或 `pred_`。
- 每份受影响 metadata 的 `offline_interpolation` 记录 `ts/column/old_value/new_value/method`，原料插值同时记录左右锚点及数值、`dependency_end`；共 24 个原料小时单元格及 7 个派生湿度单元格，出现在覆盖该时段的 12 份资产中。
- **双向离线插值依赖右侧观测，不等于当时可在线取得的实测。** 审计保留这种性质，不宣称严格在线回放。其余未授权缺口、非有限值或双端锚点缺失继续 RAISE；入模时不补值。
- 天气文件保留共享表的其他原料列，未声明入模的列不承诺完整；发布前严格校验六项实测与六项预报全部有限。

## 验证

```bash
env -u PYTHONPATH .venv/bin/python -m unittest tests.test_weather_history_coverage tests.test_weather_six_features
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_weather_hvac_windows
env -u PYTHONPATH .venv/bin/python tests/run_suite.py fast
```

天气测试覆盖独立窗口、异常网格拒绝、缺天气拒绝、插值授权范围及不回写输入；逐一核对32份真实资产的时间轴、行数、哈希和有限性，并通过生产 `SourceRegistry` 验证训练实测/历史预测预报的阶段切换。这是数据及信息集验证，不代表正式模型生命周期或效果验证。

模型配置门禁（不拟合正式模型）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_model_configs.py 'config/aidc_hvac_load_5min/**/*.yaml'
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_hvac_model_configs
```

本次配置验收：checker 576/576通过、零警告；配置定向6项、天气定向4项、fast 291项通过，均无跳过。配置定向覆盖全部576份真实训练origin与预测首/中/末步设计、全部回测折几何和代表协变量扰动；没有拟合正式模型，不代表预测效果或完整生命周期已验证。
