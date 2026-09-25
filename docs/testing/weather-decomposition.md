# 天气、分解与定向检查

## 天气生成器（实施中）

`test_weather_*` 默认纳入 integration，无额外 skip/发现排除。覆盖强类型配方、真实临时文件导入/哈希、原生小时派生、完整区间聚合、proxy 因果性、Registry、缓存、静态资产审计与 YAML→Compiler 单/批证据。全部天气资产是明确合成 fixture，不代替 P5 真实响应、P6 训练/回测/final fit/预测/bundle 证据；退役场景的旧场景测试已随 2026-09-25 收敛删除（Git 溯源），未改成污染输出的新黄金值。

## 分解职责收敛覆盖

- `test_decomposition_parameters`：STL/MSTL 的非默认 degree/cycles/damped-lookback 与直接 statsmodels + 显式外推公式比较，覆盖原参数未生效缺陷。
- `test_decomposition_validation`：分解包统一别名、类型、范围及周期校验；checker 与目标变换使用同一入口。
- `test_decomposition_analysis`：通用周期包导入不加载 decomposition；残差报告使用显式窗口/汇总列，不借列存值。
- `test_decomposition_persistence`：新 preset 状态往返、旧布局/路径拒绝、Composer 不广播或截断。
- `test_decomposition_layout`：根目录公共入口、职责子包依赖方向、轻量初始化与新进程导入；物理迁移不修改 semantic fingerprint。
- `test_decomposition_runtime`：合成输入下实际 Ridge/QR、STL/MSTL、双目标的训练/回测/预测落盘及 bundle 部署预测一致性；不替代正式场景全量重跑。
- 原冻结分解 fixture 不改写，默认与 linear 数值参照继续保留；新增测试默认进入 integration，无 skip 或漏发现。

## 禁止模型执行时的定向检查

不要把 `fast` 当作“绝无模型调用”的保证。可靠性收口仅选择以下已审阅的测试：

- `test_intraday_schedule_contract.py`、`test_oof_gap_contract.py`：纯调度与标签边界。
- `test_transform_window_geometry.py`、`test_batch_time_grid.py`：纯标签窗口与 CSV 时间网格。
- `test_model_catalog_contract.py`、`test_package_layering.py`、`test_lifecycle_static_contract.py`：描述表、依赖门禁及生命周期静态接线/状态合同。
- `test_execution_evidence_contract.py`：合成元数据的缓存读写、旧缓存缺证据、公开能力错误传播、JSON 参数快照和静态调用接线；临时测试数组不是模型结果或运行验收证据。

本地复验入口为 `.hermes/plans/verify-architecture-no-models.py`，安装调用拦截器后执行上述白名单，遇到 fit/predict 类调用立即中止。各文件仍纳入原生 discovery，不从完整测试集合排除。
