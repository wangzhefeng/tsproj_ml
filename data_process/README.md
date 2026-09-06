# data_process

`data_process/` 是进模型前的离线数据准备工具库：频率聚合、填补方法回测、异常分析、事件检测、峰谷和周期分析。周期检测的纯数值核心在 `timeseries_analysis/periods.py`；`periodicity_analysis.py` 承载 DataFrame 排序/有效样本选择、时间单位换算、规格、可视化、报告和 CLI。FFT top-k（`fft_top_k`）与 Engle-Granger 协整诊断（`coint_col`）为可选报告项；协整检验留在本离线工具中。

进入 canonical runtime 的信息集默认缺失/异常即 RAISE；本目录工具只生成新的离线 CSV，不在训练窗口内静默改值。已删除未接生产的 legacy 训练窗口清洗实现。

改动数据链时必须先确认 source、派生产物、配置引用和审计 sidecar，再按依赖顺序重建。

## AIDC 负荷事件标签工具链

- **共享检测核心** `data_process/load_event_detection.py`（单测 `tests/test_load_event_detection.py`）：事件分类 shift_up/down（持久阶跃=集中上下架）、stress_up/down（1~21 天临时偏移=压测/临时操作）、burst_up/down（1.25h~24h 日内冲击）、spike_up/down（≤1h 功率突变）；三个探测器（自顶向下日级分段 + 短时偏移 + 15min 残差 MAD 突变）+ 边界伪影抑制 + 事件→逐点/逐日投影。
- **场景入口**：`config/aidc_load_15min_daily/load_event_analysis.py`（15min 逐点标签 + 特征）、`config/aidc_load_month/load_event_analysis.py`（日频逐日标签 + 特征）；产物在 `dataset/<场景>/event_label_features/`（labeled_features.csv / events.csv / overview.png / report.md）。两频率事件明细完全一致（同一检测核心、日水平基线均取 15min 聚合逐日中位数）。
- **列名前缀约定**：`feat_`=本频率 trailing 特征（无泄漏，可建模）、`xf_`=跨频率特征（15min↔日频互取）、`xr_`=跨 route 特征、`lbl_`=事件标签（检测含居中窗口=有未来信息，仅供离线分析/样本筛选，禁止直接作在线预测特征）。
- 注意 `dataset/aidc_load_month/` 目录名沿用场景名，文件实际是 **1day 粒度**。
