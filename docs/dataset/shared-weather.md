# 共享天气资产

> 本文自 `dataset/shared/weather/README.md` 迁入 docs 并纳入版本控制。

> **非活动链（2026-09-07）**：raw/normalized/manifests 资产层属于研究回放/下载工具链，不再作为活动配置的输入；活动配置全部走 file 两段制（场景级 `dataset/<场景>/weather_history_*` + `weather_future_*`，history=rt_ 实测 / future=pred_ 预报）+ `inference_columns` 合同。`extracted/actual/` 的权威源并集仍是场景数据的事实来源。

`dataset/shared/weather/` 目录集中保存天气原字节与研究资产。研究资产已经注册，但不具备严格历史 as-of 或实盘资格；注册通过不表示活动配置已迁移或业务场景已验收。

- `raw/objects/<sha256>/payload`：不可变原始字节，按内容去重；包括原响应、原 CSV、metadata 与来源证据。
- `raw/by-source/<source>/<location>/<snapshot>/<manifest-sha>.json`：来源、地点和版本索引；无元数据的原内容只归档，不编造地点/版本。
- `normalized/<sha256>/data.csv`：宽表规范视图（`time, <变量列...>, available_at`，UTC 带时区，缺测留空不填）。
- `normalized/import-views/`、`normalized/import-metadata/`：显式导入支撑段及元数据；不覆盖完整原响应。规范表保留真实接收可得时间，研究发布时间假设仅在配方中声明。
- `manifests/<sha256>.json`：`weather_asset_v1` 资产事实，引用均相对显式项目 `base_dir`。
- `extracted/actual/`：供应商 rt_ 实测零冲突并集，以**增量分片**形式保存：每个文件命名 `weather_in_<起始yyyymmdd>_<截止yyyymmdd>.csv`，起止为含数据的日期（inclusive），LF 行尾，完整小时网格、NaN 不填、含 available_at 证据列：historical_release_contract + 区间结束 30D 保守滞后。分片间允许时间重叠，但同 ts 同列不得存在两个不同的非空值；消费方（`scripts/build_scenario_weather.py`）按文件名排序加载全部 `weather_in_*.csv` 并做零冲突合并，冲突即 RAISE，不 keep-last、不静默取舍。新增分片只追加、不改写既有分片；每个分片可各自挂 `<同名>.six_features_repair.json` 离线修补证据（按分片自身 sha256 绑定）。
- `extracted/forecast/`：按场景的未来预报数据（仅 pred_* 全部非空的小时行，逐场景 available_at 取旧流水线拉取时刻常量；供应商批次 vintage 无证据不伪造；覆盖不足记入 sidecar，不填补）。权威仍以 raw 原字节为准，本层可由 `.hermes/plans/weather-extract-*.py` 重建。
- 场景级处理后天气数据（频率转换/重采样等消费形态）放入 `dataset/<场景>/` 对应数据目录；本目录只保存共享资产与提取视图。

模型 YAML 声明处理规则；本目录不保存第二套重采样、滚动窗口或 proxy 配方。不同预报快照不能按时间 keep-last 合并。原始路径别名保留在来源记录中，历史共享不扩大模型训练窗口。

本地准备入口为 `scripts/prepare_weather.py`，默认 dry-run，显式 `--write` 才发布；不联网、不覆盖既有资产。完整合同与限制见 [`../packages/weather_generator.md`](../packages/weather_generator.md)，CLI 见 [`../scripts.md`](../scripts.md)。

旧脚本/场景副本仅在替代验收、用户指定备份的恢复验证及逐文件删除批准后退出。当前备份位于 `dataset/tsproj_ml-weather-backup/`，不提供异地容灾。临时 fixture 不属于本目录的真实资产。下载失败的响应/capture保留，不通过填值或改发布时间补造合格资产。
