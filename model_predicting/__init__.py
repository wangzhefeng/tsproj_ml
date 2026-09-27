"""预测执行面、部署证据与 bundle 持久化。

子包划分（2026-09-28，按消费方）：``contracts/`` 注入协议（FeatureProvider）、
``loops/`` 预测与部署执行（predictor/deployment）、``artifacts/`` bundle 持久化、
结果写盘与证据（persistence/results/evidence_collect/evidence_assembly）。

张量类型定义在合同层 ``forecasting_core.tensors``；本包不再转发导出
（2026-09-26 门面收口：曾从包根 re-export 三个张量类型，全仓唯一消费方
是测试自身的导出断言，业务代码均直接走合同层路径，转发门面删除）。
"""
