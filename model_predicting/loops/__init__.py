"""预测与部署执行回路。

``predictor`` 为训练期 point/marginal quantile 执行与 crossing 修复、
median path 组装唯一实现；``deployment`` 为 bundle 自包含部署预测。
无包级转发门面，消费方走完整点路径导入。
"""
