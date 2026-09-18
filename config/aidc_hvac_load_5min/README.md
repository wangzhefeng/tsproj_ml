# AIDC 暖通 5min 场景

数据提取、因果填补与预测窗口合同见 [scripts/README.md](scripts/README.md)。本场景目前准备数据资产与可复用 weather source 片段，**尚未创建模型 YAML，也未训练模型**。

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

新增测试覆盖独立窗口、异常网格拒绝、缺天气拒绝、插值授权范围及不回写输入；逐一核对32份真实资产的时间轴、行数、哈希和有限性，并通过生产 `SourceRegistry` 验证训练实测/历史预测预报的阶段切换。这是数据及信息集验证，不代表正式模型生命周期或效果验证。模型 YAML 和 `run.py` 冒烟按用户裁决留待后续模型配置任务。
