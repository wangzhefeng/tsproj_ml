# model_pipeline

runner构造只做候选原点几何、一个真实原点的schema探测和资源规划，不构造完整训练矩阵。`prepare_training()`在fit/final或同组共享入口按需准备，一次准备由锁保护；直接访问训练数组也会触发准备。支持的历史路径保留`IndexedDesign`，ETS不编译监督特征。预测保留原batch/provider内核；bundle部署不构造训练矩阵。资源证据区分计划逻辑字节与实际保留数组字节，未准备时后者为零，不等于进程RSS。

联通严格原始历史通路：runner 的逐折 fit 接入原生 ETS（只消费完整原始历史，不用监督标签拟合）与 seasonal residual（逐监督原点减基线、预测原位恢复）。残差要求 Local/point/显式 training_window/无目标变换；窗口外污染隔离有测试。ETS 与 seasonal_baseline 均仅支持回测，final fit/bundle 显式拒绝；残差的生命周期入口在写结果前拒绝，独立 final 准备、拟合和 bundle 构造同样拒绝，限制不依赖已退役字段。块天气、同槽/近期状态由公共 compiler 实现。

`model_pipeline/` 负责单模型生命周期、监督设计与批量运行编排；根 `run.py` / `batch_run.py` 调用本包。融合仍由独立 `model_ensemble/` 负责，通过入口注入 runner 与执行服务复用单模型链。

- `runner.py`：`CanonicalBaseModelRunner` 与 `run_canonical_config()`；提供训练、预测、历史准备和只读证据能力，run 入口设置线程限制后委托生命周期。
- `lifecycle.py`：`run_lifecycle()` 管理完成状态与异常传播，`execute_lifecycle()` 组织回测、CQR、final fit、预测和持久化；包含 `CanonicalRuntimeResult` 与产物元数据助手。结果类型仍由 runner 公开导出。
- `supervised_design.py`：`SupervisedDesignBuilder`、information set、训练/预测设计、监督标签窗口、`probe_training_design()`、`minimum_history_rows()`；批编译不支持的设计保留 single 路径，不隐式替换 provider。
- `fold_fit.py`：fold/final 特征选择、变换与训练服务；回测和 final fit 共用配置训练窗口。
- `run_state.py`：running/completed/failed 状态写入和 completed 校验；状态不是跨目录事务，外部直接加载 pickle 不自动消费状态。
- `batch_runtime.py`：`run_canonical_batch()`、`verify_batch_results()` 与报告，负责跨配置 preflight、调度、恢复和验收。
- `batch_artifacts.py`：产物路径、摘要、时间网格、维度与 bundle 身份验收；只看文件存在不能判 completed。

## 训练原点选择

显式 training_window 经 temporal 合同统一历史下界和预测网格；temporal_backtest_windows 基于真实时间覆盖生成折，折 runner/final 共用标签截止和采样规则。采样前置，fit/final 不重复采样；窗口、预测区间、采样进入原始设计身份。旧字段及旧滚动切分器已退役。

`training_origins.select_training_origins`在安全训练窗内执行间隔/固定时刻/锚点周期/最近数量筛选。artifact 区分采样前 candidate 与采样后 selected 原点数；运行计时不是语义。自然月保留日窗合同。

## 编排边界

天气阶段通过 `forecast_designs(..., data_phase="historical"|"future")` 传至不可变请求。默认 historical，供滑窗测试、OOF 及当前生命周期末次历史留出预测使用；训练始终 historical/实测列，测试预测为 historical/预报列。真正未来调用方必须显式传 future，递归 provider 捕获同阶段信息集。训练设计缓存仅哈希映射天气源的 history，不依赖未来文件；其他 source 原合同不变。

rolling 时 history_start = origin - (W-1) * freq；expanding 使用显式 start_time。for_backtest_window/for_forecast_origin 重建独立 as-of 上下文，拟合、预测和递归 provider 共用下界。主 runner 只规划；批量 final 设计可共享，但当前严格窗口的各模型各折仍独立编译，不承诺沿用旧无界路径的单次编译次数。

`SupervisedDesignBuilder` 经 `SourceRegistry.target_history_coverage()` 获取目标源序列/时间覆盖，再在本包决定 `series_order`、unknown/incomplete policy、训练窗口和监督张量。数据读取、验证及公共 identity 选择由数据层提供；runner、batch runtime、lifecycle 通过 registry 的公开 `base_dir`/`generators` 取得上下文，不穿透私有状态。递归预测目标 provider 与 oracle 标签策略仍属于本包，不迁入通用数据层。

fixed-step/calendar-month 循环分别位于 `model_testing/fixed_step.py` 与 `model_testing/calendar_month.py`；通过显式回测协议消费 runner 能力，逐折评分都委托 `model_testing/scoring.py`。目标变换识别与并行拟合所需的标签历史由 runner 的 `backtest_target_histories()` 提供，测试包不导入具体变换实现。预测/部署和 bundle 构造持久化位于 `model_forecasting/`，final bundle 构造由 `build_strategy_model_bundle()` 统一处理。

`CanonicalBaseModelRunner.execution_evidence(artifact, target_transform)` 提供公开只读证据能力，供回测与融合成员通过协议调用；不为收集证据再次拟合或预测。CQR 收集在 final fit 前完成，部署只应用已保存校准状态。

天气源的 `weather_evidence` 随 source lineage 持久化到生命周期产物与 bundle；去重身份包含证据，不能把不同原点/快照折叠为同一来源路径。非天气源保持既有字段格式。

资源规划、checkpoint、性能档和内存缓存属于`model_performance/`；目标/特征变换属于`feature_engineering/transforms/`。单模型、批量、融合业务入口不读写`_compiled_features/`；原始设计身份仍绑定源内容、生成器、依赖及编译链，供内存共享/checkpoint使用，不等于配置语义fingerprint。磁盘编译缓存API、参数及旧目录已退役；身份校验迁入`feature_engineering/design_identity.py`。运行证据以`design_preparation`记录编译/内存共享状态，不再输出磁盘命中字段。模型checkpoint、融合OOF及正式结果不受删除影响。

批量全局preflight只保留规划摘要，不通过磁盘传递设计；每组执行时完整编译一次并在组内共享，组结束释放设计及动态缓存。动态horizon复用受字节/条目上限约束，严格窗口的折设计仍独立。后续组数值构造失败时，之前已完成组保持completed，失败项明确记录；不再承诺所有组完整编译成功后才开始训练。融合通过`share_training_design()`延迟共享相同身份的原始数组，不共享成员拟合态。
