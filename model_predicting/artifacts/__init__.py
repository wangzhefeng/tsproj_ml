"""Bundle 持久化、结果写盘与执行证据。

``persistence`` schema-2 bundle 构造与落盘；``results`` canonical long
结果写盘/绘图/读取；``evidence_collect`` 运行时只读采集；
``evidence_assembly`` 生命周期证据纯组装（采集 vs 组装对偶）。
无包级转发门面，消费方走完整点路径导入。
"""
