# BLOCKED.md — 待裁决/交接清单

## 2. direct 策略完整运行接线与实测
`compile_batch` v2 已达到 0.012638s/origin（19776 origins 线性投影 4.165min），但当前 `model_forecasting/design.py` 尚无调用点。本次允许 diff 明确排除 design/runtime，因此未接线，也未重新执行完整配置 run；现阶段只能声明批编译方法达标，不能声明生产端到端已提速。后续接线时仍需按原任务书核验完整 run、MAPE 与产物缓存。

## 3. 并发会话改动共存
工作树存在另一会话对 decomposition 组配置的并发修改（lgbm_recmo/recursive-decomp-* 删除等），非本任务改动面，未合并处理。

## 4. 并发配置重构导致的测试失同步（2026-09-02 晚发现）
另一会话对 15min 三场景配置矩阵做了二次重构（新增 weather_only/holiday_only 变体、dirmo/dirrec/dirrecmo 策略、移除线性模型与部分 decomp 配置，模型总数 707→687）。以下 3 个测试需同步（非本任务改动面）：
- tests/test_config_entrypoints.py:410 引用的 enet_direct.yaml 已不存在
- tests/test_config_entrypoints.py:886 checker 数量阈值 700 > 实际 687
- tests/test_validation_geometry_manifest.py:38 manifest（707 条目）与现役 687 不符
