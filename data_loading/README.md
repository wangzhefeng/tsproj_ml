# data_loading

`data_loading/` 是 canonical 唯一数据读取与信息集构造层。

- `registry.py`：`SourceRegistry` 统一编排来源加载、角色分区和信息集交付，保留 path-version lineage。
- `sources/source_io.py`：文件读取、原始帧及验证帧缓存；缓存属于单个 registry，不保证外部文件热更新。
- `processing/validation.py`：列投影、时间、有限值与主键校验；`processing/visibility.py`：as-of vintage 选择及历史/标签时间边界。
- `processing/alignment.py`：identity 选择、序列顺序、角色投影及 known-future 覆盖检查。
- `information/information_set.py`：请求、物化信息集与 lineage 合同；`information/indexing.py`：角色限定的行位置索引实现，缓存状态仍由信息集持有。
- `information/providers.py`：observed-past 显式 future provider 及轨迹包装。
- `sources/discovery.py`：目标 history 的序列和时间覆盖事实，不暴露 target 值、不决定训练政策。
- `sources/provenance.py`：源文件内容哈希与生成器实现哈希；`sources/assets.py`：文件存在性、声明列和表头预检查。
- `calendar_generator/`：`calendar_features.py` 提供纯日历计算，`chinese_holiday.py` 适配请求，`__init__.py` 维护统一注册表。YAML 用法和可得性限制说明在 `chinese_holiday.py`；外部从 `data_loading` 公开入口导入，不保留旧文件兼容层。`chinese_holiday` 使用 chinese-calendar 后端；`next_holiday_days` 对超出已知年历的后续节日取删失哨兵 400。

只依赖 `forecasting_core.specs` 与基础库。缺失、重复、时间越界和可得性违规直接 RAISE；离线填补和清洗属于 `data_process/`。

## 职责收敛约定

目录按职责归组：`sources/` 管理读取、覆盖发现、资产与溯源；`processing/` 管理校验、可见性和对齐；`information/` 管理请求、结果、索引及 provider。根目录只保留 registry 和公共导出。子包不是互不依赖的层：provider 调用读取与处理规则，discovery 调用读取与对齐；禁止内部反向导入 registry。旧 information_set/providers 文件及 registry 的旧 provider 别名已删除，不支持对应旧 pickle 路径，不自动迁移或删除存量结果。

内部拆分按职责而非类数进行：`source_io` 拥有一次运行内的读取与验证帧缓存，`validation` 校验物理数据，`visibility` 处理 as-of/标签边界，`alignment` 处理序列与时间对齐，`indexing` 实现行索引。`registry` 保留统一请求编排入口，不采用 mixin 聚合。

`discovery` 只报告目标源序列与时间覆盖事实，训练顺序、缺失序列政策仍在 pipeline。`provenance` 提供源文件和生成器身份，特征设计缓存仍在 feature_engineering。`assets` 只做文件存在性与表头预检查，不扫描 YAML，不替代完整运行验证。上层通过公开接口消费，不读取 registry 私有状态。

`calendar_generator/` 分离纯日历计算、请求适配和注册表；旧 `holiday_generator.py` 与 `generators/` 路径已退出，不留 shim。信息集/provider 的根包公开导出与 YAML 标识保持，类实现路径使用 information 子包。节气采用依赖库公开 API 修复闰年边界，日内请求仅生成实际覆盖日；这些是明确的正确性修复，不承诺旧错误特征值保真。实现指纹变化会触发设计缓存重建，不自动重跑或删除旧结果。

## 调用入口与输出

上层根据 `DataSpec` 构造 `SourceRegistry`，用 `InformationSetRequest` 表达预测原点、目标时间网格和读取角色，再调用 `materialize()`。输出 `MaterializedInformationSet` 按 target history、observed-past、known-future、static 分区，并携带 `SourceLineage` 与行位置查询缓存；它不是已经拟合或缩放的训练矩阵。

`base_dir` 为只读属性，`generators` 返回防御性副本；`target_history_coverage()` 只报告完整 history 的覆盖事实，`latest_target_time()` 校验各 target 源的最后时间一致。模型输入仍必须经 `materialize()` 的 as-of 规则，覆盖发现不是放宽可见性的入口。

## 信息边界

- `data.sources` 是多文件接入入口。`columns` 是信息投影视图：未声明的物理列不进入模型，声明且非 ignored 的缺失列直接报错。
- `history_path/backtest_path/future_path` 区分历史事实、回测时已发布预报与正式未来输入；不能把回测期实测天气冒充 known-future。
- `target_access=supervised_labels` 仅在训练取标签时放开预测期 target，不放开其他动态 source 的 as-of 边界。
- observed-past 越过预测原点时必须显式选择 `persistence/auxiliary/provided_scenario`；真实历史 lag 无需用 provider 替代。
- `chinese_holiday` 在 `BUILTIN_GENERATORS` 注册；超出后端年历覆盖范围报错，不能生成伪节假日。

## 日历口径

`is_holiday` 包含普通休息周末；`holiday_name` 仅命名节日非空。`next_holiday_days`/`prev_holiday_days` 为距休息日距离，不是距命名法定节日距离。`solar_term` 为当前节令、非节气日继承上一节气；一张帧只构建一次节气表。

生成器对稀疏/月末网格返回指定日期的日状态，不做月内统计。`available_at=forecast_origin` 体现 calendar-known 假设，没有历史公告版本验证；它不是实际公告发布时间证明。
