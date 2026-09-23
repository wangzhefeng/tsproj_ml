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
    forecast_data/build_model_configs.py # 六组A1_all配置，消费data_v3，独立发布
```

## 数据布局与版本含义

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

- `raw_data/`：20份共享原始宽表，保持字节不变，不按准备版本复制或覆盖。
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
