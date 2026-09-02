# PROGRESS.md — 特征编译批量向量化 + 产物缓存

## 回执（2026-09-02）
- 目标：编译 ≤5 分钟/配置（基线 47min），语义等价（向量化 vs 逐点逐值对照），编译产物缓存命中 <1s。
- 顺序：任务1 等价性对照测试(先红) → 任务2 compile_batch → 任务3 design 接线 → 任务4 缓存 → 任务5 性能验收。
- 最大风险：advanced 变换（rolling/diff）向量化与逐行路径的边界条件不一致（首行 NaN、窗口不足）。
- 基线核对：dev 69ae04d，662 tests OK（4.2min）；单 origin 编译 0.09~0.24s（均值 0.11s，与任务书 0.14s 偏差 <20%，通过）。

## 完成情况（2026-09-02）
- ✅ 任务 1：等价性对照测试已写（tests/test_compiler_batch_equivalence.py，3 测试），反向验证红（compile_batch 前身不存在时 3 errors）；现已收紧为 VisibilityProof dataclass 全字段逐条相等。
- ✅ 任务 2：compile_batch v2 真向量化已完成。lag 使用 int64 ns + row-position lookup 批量定位；known_future 使用 `Index.get_indexer`；rolling 对共享完整历史一次计算、difference 按 origin 位置索引；frame 列一次组装，保留逐行 proof 合同。递归/provider 依赖和未批量化历史变换显式告警后回退 `compile()`，`compile()` 本身未改。
- ✅ 性能验收：目标配置 10 origins 独占冷启动 0.127862s，即 0.012786s/origin；v1 0.131522s/origin → v2 0.012786s/origin，10.286x。按 19776 origins 线性投影 252.856s（4.214min），低于 5min 目标。
- ✅ 语义验收：`tests.test_compiler_batch_equivalence -v` 3/3 OK；`tests.test_feature_visibility_compiler tests.test_canonical_runtime_smoke` 24/24 OK；另核验 direct_horizon/direct_pointwise/recmo 各 2 origins 的 frame + 完整 proof 精确相等。
- ⏸ 原总任务书任务 3/4（design 接线、编译产物缓存）不在本次允许文件范围，未实施；仓库当前生产 design 尚无 compile_batch 调用点，完整 run 提速不可冒充已验证。
