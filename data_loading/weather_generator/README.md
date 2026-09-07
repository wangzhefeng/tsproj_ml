# weather_generator

> **非活动链（2026-09-07）**：本子包保留为研究回放/下载工具链维护；全部活动配置已改走 file 两段制（history=rt_ 实测 / future=pred_ 预报）+ `inference_columns` 合同，不再经本包生成。新活动场景勿引用本包；`scripts/prepare_weather.py` 等准备入口仅在研究回放/取证时使用。

通用本地气象处理包，按 `contracts/assets/adapters/derivation/resampling/scenarios/pipeline/generator` 分工。
本目录正在实施；没有已验证来源就不能宣称完成场景迁移。

## 研究回放（独立资格）

配方可显式指定`research: {release_delay: '8h', rationale: '...'}`及`semantics_version: weather_research_v1`；普通`weather_v1`禁止研究假设。forecast回放的模拟发布时间为init_time加delay，prior_year_proxy的模拟可得时间为依赖time加delay（区间左标签还必须加完整原生区间）。这是实验假设，不是历史发布证据；原始manifest、normalized里的实际available_at不改写。回放允许hindcast但保留其data_kind。输出proof记录真实/模拟可得时间、假设、strict_asof_verified=false及production_eligible=false，bundle默认部署必须拒绝。

## 资产合同

manifest 版本为 `weather_asset_v1`，只含 `schema_version/snapshots`。每个 snapshot 显式声明：

- `source_id/product/model/snapshot_id/data_kind/location_id/latitude/longitude/coordinate_system/timezone`；坐标必须 WGS84，来源身份不能由列前缀推断。
- `init_time/issued_at/received_at/evidence_class/evidence_ref`；时间可以按产品为 null，但证据必须足以证明规范表的 available_at。forecast 不接受 hindcast 冒充；init 不是发布。
- `variables`：每项 `name/column/unit/semantics/native_freq/label`，语义是 point、interval_mean 或 interval_sum；interval 标签 left/right。
- `raw`：原始依赖的 `{path,sha256}` 列表；`normalized`：规范表的同类文件描述；所有路径按显式项目 base_dir 解析，不按 cwd 猜测。

规范表为宽表 CSV：`time, <每变量一列>, available_at`。time 与 available_at 必须带时区，全部按 UTC 校验；列集合必须精确等于声明变量加这两个证据列；缺测单元格留空（NaN）允许存储，inf 拒绝；请求触及缺失值仍在生成时 RAISE。变量物理单位与采样语义取自快照；内部计算帧由加载时 pivot 为 long，宽表语义不进算法层。无证据原始内容允许保全，但不生成伪合格 manifest。
Open-Meteo JSON 的 manifest 坐标记录响应网格点，必须与原响应精确一致；不能用请求场站坐标冒充响应网格点。请求位置与响应网格的对应关系应保存在原始请求/sidecar 证据中，正式取数前单独核实。当前 JSON 适配只接受 UTC `iso8601` 小时响应，不将 Unix 时间数值按纳秒解释。
`evidence_class` 为 `received_snapshot`（不能倒推 received_at 之前）、`documented_release`（有发布证据）、`historical_release_contract`（有来源版本合同，observed/reanalysis 仍不能在有效区间结束前可得）。引用证据也须作为 raw 哈希依赖，不能只写自由文本自证。

多快照保存不同 vintage，不按 time keep-last。读取必须核对 manifest 固定 hash 及所有 raw/normalized hash；异常、冲突与缺资产直接报错，不联网。

准备 CLI 将原字节集中在 `raw/objects/<sha256>/payload`，跨来源相同字节只存一次；`raw/by-source/<source>/<location>/<snapshot>/<manifest-sha>.json` 记录来源/地点/版本索引。尚无元数据的 archive 只保留来源与原路径记录，不编造地点或版本。注册同时保全输入 metadata 的原字节，manifest/normalized/raw 引用均相对显式 `base_dir`；注册输出目录必须位于该根目录内，整体搬迁后仍可校验恢复。

## 配方与依赖方向

`degree` 风向的 mean/rolling_mean 使用等权单位向量圆周均值，不隐式按风速加权；合向量接近零（模长不超过 1e-12）时方向不可判定，RAISE。min/max 无通用圆周顺序，拒绝。difference 使用最短有符号角差，输出 `delta_degree`（不再按绝对风向取模）；正好相反的方向差拒绝。所有原生预热与可得性传播规则保持不变。

历史 Open-Meteo JSON 注册可显式提供 `--availability-csv`，列严格为 `time,available_at`（均带时区）。逐时间一对一完整覆盖原响应，不补齐、不倒推发布时间；CSV 必须列入 metadata 的 raw 哈希依赖并随资产保全。它只是发布合同的逐行表达，不替代来源历史发布证据；没有证据仍拒绝入模。

模型 YAML 使用 `forecasting_core.specs.weather.WeatherGenerationSpec`，语法见 config README。
manifest 不放重采样/proxy/滚动参数。已污染派生文件不是黄金参照；预报、实测、再分析身份独立于输出列名。
本包仅依赖 forecasting_core 和基础库，不反向导入 registry。源文件读取、生成器绑定和特征缓存的接线由上层负责。

温度差使用 `delta_K` / `delta_degC`，两者只换尺度、不加绝对温度偏移；不能把差值声明成绝对 `K` / `degC`。日统计按配置时区的自然日覆盖，DST 日可以是 23/25 个真实小时。区间总量下采样使用求和；拆分为更细区间需显式 `hold`（区间内均匀率假设）及足够 `max_age`，不能将整小时总量复制给每个子区间。
