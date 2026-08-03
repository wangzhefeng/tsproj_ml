# data_process 说明

`data_process/` 提供时间序列离线数据处理工具，覆盖五条链路：**频率聚合与缺失填充**、**填充方法回测**、**原始数据异常检测与清洗**、**周期自动检测**、**峰谷检测与提取**。均为配置驱动、与模型运行解耦的通用算法工具，不依赖具体数据集。

## 模块结构

```text
data_process/
├── data_aggregate.py          # 单目标时间序列频率聚合 + 缺失填充 + 审计缓存
├── fill_method_backtest.py    # 填充方法回测：掩码真实观测，按缺口长度分桶对比 MAPE
├── outlier_process.py         # 原始序列异常标记、清洗与可视化
├── periodicity_analysis.py    # 周期自动检测：FFT 主导周期 + ACF 周期候选 + STL 季节性分解
└── peak_valley_detection.py   # 峰谷检测与提取：find_peaks + 幅度排序 Top-N
```

五个工具各自独立可运行，均接受 YAML 配置作为位置参数，支持 `--force` 忽略缓存/旧输出强制重建：

```bash
uv run python data_process/<tool>.py <config.yaml> [--force]
```

配置 YAML 的 schema 与模型配置完全独立（无 `base_config`/`overrides` 字段），避免被批量模型 driver 误当模型配置加载。

---

## 1. 频率聚合与缺失填充

### 算法流程

1. **规则化**：读取源 CSV 的时间列与目标列，时间转 `datetime`、目标转数值；按时间升序排重，`resample` 到源频率网格——缺失时间戳以 NaN 占位，得到规则序列。
2. **缺失统计**：统计连续缺失段（段数、最长段长度），供审计记录。
3. **缺失填充**：按配置的填充方法补齐源频率上的缺失点；填充后仍有缺失则报错（不允许带洞进入聚合）。
4. **频率聚合**：对补齐后的规则序列 `resample` 到目标频率，应用聚合方法（`mean`/`max`/`min`/`sum`/`median`）。
5. **原子落盘**：输出 CSV 与审计 JSON 先写临时文件再 `os.replace` 原子替换，避免中断留下半成品。

### 缺失填充方法

| 方法 | 算法 | 说明 |
|---|---|---|
| `linear` | 按时间权重线性插值（`interpolate(method="time")`），双向补端点 | 默认推荐；适合日内平坦、缓慢趋势的序列 |
| `seasonal_slot` | 局部 ±`fill_weeks` 周窗口中，取**同星期 × 同时刻**观测的均值填充 | 需要周周期上下文；窗口内无候选观测则保持 NaN |
| `none` | 不填充 | 仅做规则化与聚合，缺失点直接进入统计并可能触发报错 |

### 审计缓存

每次聚合落盘一个 `<output>.aggregate.json` 审计文件，记录：

- 源文件指纹（路径、大小、mtime）、输入输出配置（时间列、目标列、源/目标频率、聚合与填充方法、逻辑版本号）
- 运行统计（源/输出行数、插入时间戳数、填充点数、缺口段数、最长缺口、重复时间戳数、时间范围）

再次运行同一配置时，若源文件与配置未变（审计匹配），直接复用既有输出，避免重复计算。修改聚合/填充处理逻辑时递增逻辑版本号使旧缓存自动失效。

### 配置 schema

单任务（顶层平铺）或多任务（顶层 `tasks:` 列表）：

```yaml
# 单任务
source_path: dataset/<scenario>/<source>.csv
time_col: <time_col>
target_col: <target_col>
source_freq: 5min
target_freq: 1D
method: mean
fill_method: linear
fill_weeks: 4
output_path: dataset/<scenario>/<output>.csv

# 多任务
tasks:
  - source_path: ...
    ...
```

必填字段：`source_path`、`time_col`、`target_col`、`source_freq`、`target_freq`、`output_path`。可选：`method`（默认 `mean`）、`fill_method`（默认 `none`）、`fill_weeks`（默认 `4`，`seasonal_slot` 用）。`target_freq` 不得细于 `source_freq`；`output_path` 不得覆盖源文件；路径相对项目根解析。

---

## 2. 填充方法回测

用于在真实数据上对比不同填充方法（注册表 `_FILLERS`）的精度，新增填充方法注册后自动纳入回测。

### 算法流程

1. **构建真值基准**：读取一路规则化序列，缺失点用时间插值补齐，作为连续真值。
2. **构造掩码缺口**：按预设缺口形状（缺口长度 → 随机起点样本数）掩码真实观测为 NaN；全天缺口（等于每日槽数）用固定步长穷举起点，避免随机采样代表性不足。
3. **分方法填充**：对每个掩码缺口，依次用注册表中的每种填充方法恢复。
4. **误差评估**：对每种方法计算掩码段内的 MAPE（平均绝对百分比误差），按缺口长度分桶汇总输出（均值与样本数）。

### 使用要点

- 掩码起点需避开序列首尾（`--margin-weeks`，`seasonal_slot` 需要上下文窗口）。
- 支持 `--route`（A/B/both）与 `--seed` 控制随机性。
- 输出按缺口长度分桶的 MAPE 对比表，作为选择 `data_aggregate` 默认填充方法的依据。

---

## 3. 异常检测与清洗

### 检测规则（三类信号组合）

1. **绝对低值**：目标值低于配置阈值 `abs_low_threshold` 视为异常。适合量纲已知、存在物理下限的序列；量纲未知或允许大幅负值时设极大负阈值关闭该规则。
2. **局部 robust Z-score**：对目标序列做中心滚动窗口（可多窗口，如 `[25, 145]`）的 `median` 基线，残差 = 值 − 基线；尺度 = `1.4826 × 滚动 median(|残差|)`（MAD 估计）。高侧分数 = `clip(残差/尺度, 0)`，低侧分数 = `clip(−残差/尺度, 0)`；多窗口分数逐点取最大值融合。
3. **周期 robust Z-score**：按日内时间槽位（由序列频率自动推导，5min → 288 槽）分组，槽内 `median` 为基线，残差绝对值分组 `median` 经 robust 尺度变换为尺度，同样分高侧/低侧。

### run-length 过滤

短异常段（连续异常点数 ≤ `max_short_run_points`）全部保留；长段要求分数达到原阈值的 `long_run_score_multiplier` 倍才保留，避免正常波动聚集被误判为长异常。

### 标记输出

每类规则生成掩码，异常类型合并（`local_high`/`local_low`/`periodic_high`/`periodic_low`/`absolute_low`，可叠加以 `;` 连接）。输出三列：

- `是否异常`：`是`/`否`
- `异常类型`：命中的规则类型
- `异常分数`：各分数序列逐点有限最大值（四舍五入 6 位）

### 清洗与可视化

- 清洗：异常点目标列置 NaN → 按时间线性插值 → `ffill`/`bfill` 补端点，保留原时间列与目标列。
- 可视化：全序列折线图 + 红点标注异常点，标题含总点数、异常数、异常率。

### 配置 schema

单任务（顶层平铺）或多任务（顶层 `tasks:` 列表），路径相对项目根解析：

```yaml
tasks:
  - source_path: dataset/<scenario>/<source>.csv
    time_col: <time_col>
    target_col: <target_col>
    route: <label>           # 可选，用于图标题
    abs_low_threshold: 1000.0
    local_baseline_windows: [25, 145]
    local_robust_z_threshold: 3.0
    periodic_robust_z_threshold: 2.8
    max_short_run_points: 12
    long_run_score_multiplier: 2.5
    plot: true
```

必填字段：`source_path`、`time_col`、`target_col`。其余参数均有默认值。输出到源文件同级的 `outlier_analysis/` 子目录，文件名从源文件名派生：

```text
<stem>_outlier_detection.csv  异常标记（原始数据 + 是否异常/异常类型/异常分数）
<stem>_remove_outlier.csv     清洗后数据（异常点 time 插值 + ffill/bfill）
<stem>_anomalies.png          全序列折线图 + 红点标异常
```

---

## 4. 周期自动检测

### 算法流程

1. **线性去趋势**：对目标序列拟合一次多项式并扣除（强趋势会淹没周期信号，导致 ACF 单调下降无局部峰、FFT 主导频率退化为序列长度级）。
2. **FFT 主导周期**：去趋势序列做 FFT，取正频段幅度最大的频率分量，其倒数即主导周期（样本数），结合采样间隔换算为天。
3. **ACF 周期候选**：自相关函数（statsmodels `acf`）从 lag=1 起找**正相关**局部极大值（负值区的"峰"不是周期证据），按相关值降序取前 N 个滞后步数作为周期候选；候选为空表示未检测到显著周期。
4. **STL 季节性分解**：用配置的 `seasonal_period`（缺省取 ACF 主导周期）做 STL 分解，输出季节成分标准差、残差标准差及比值（越大季节性越强）。

### 输出

源文件同级的 `periodicity_analysis/` 子目录：

```text
<stem>_periodicity_report.csv  结构化报告（指标名/值/说明）
<stem>_acf_plot.png            自相关函数图（前 max_lags 步）
<stem>_fft_plot.png            FFT 幅度谱图（正频段）
```

### 配置 schema

单任务（顶层平铺）或多任务（顶层 `tasks:` 列表），路径相对项目根解析：

```yaml
tasks:
  - source_path: dataset/<scenario>/<source>.csv
    time_col: <time_col>
    target_col: <target_col>
    route: <label>
    max_lags: 2000         # ACF 最大滞后步数，默认 2000
    seasonal_period: null  # STL 季节周期（样本数）；null=用 ACF 检测周期
    top_n_periods: 3       # 报告输出前 N 个 ACF 周期候选
    plot: true
```

必填字段：`source_path`、`time_col`、`target_col`。

---

## 5. 峰谷检测与提取

### 算法流程

1. **局部极值检测**：`scipy.signal.find_peaks` 找局部极大值（峰），对 `-y` 找峰即局部极小值（谷）；支持 `height`/`distance`/`prominence`/`width` 过滤（`prominence` 显著度最常用，抑制噪声抖动）。
2. **幅度排序**：以 |值 − 序列中位数| 度量显著性，降序编号 `rank`；`top_n` 只保留最显著的前 N 个。
3. **输出**：峰谷明细 CSV（类型/位置索引/时间戳/值/显著度/幅度排序）+ 全序列折线图（峰谷三角标记）。

### 输出

源文件同级的 `peak_valley_analysis/` 子目录：

```text
<stem>_peaks_valleys.csv  峰谷明细（type/index/time/value/prominence/rank）
<stem>_peaks_valleys.png  全序列折线图 + 峰谷标注
```

### 配置 schema

单任务（顶层平铺）或多任务（顶层 `tasks:` 列表），路径相对项目根解析：

```yaml
tasks:
  - source_path: dataset/<scenario>/<source>.csv
    time_col: <time_col>
    target_col: <target_col>
    route: <label>
    height: null        # find_peaks height 过滤（绝对高度下限）
    distance: 1         # 相邻峰谷最小间距（样本数），默认 1
    prominence: null    # 峰谷显著度过滤（相对相邻极值）
    width: null         # 峰宽过滤（样本数）
    top_n: null         # 只保留幅度最大的前 N 个（null=全部）
    plot: true
```

必填字段：`source_path`、`time_col`、`target_col`。

---

## 验证

本模块无 pytest 套件，修改后验证方式为：

- `uv run python -m py_compile data_process/*.py` 确认语法。
- 用最小 YAML 实跑 `data_aggregate`/`outlier_process`/`periodicity_analysis`/`peak_valley_detection`，检查输出文件、审计 JSON（聚合）与各 `*_analysis/` 产物。
- 填充方法变更时运行 `fill_method_backtest` 对比 MAPE，确认无回归。
- 周期检测修改后，用已知周期序列（如合成正弦波）验证 FFT/ACF 能还原周期。
