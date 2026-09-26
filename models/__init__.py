"""模型静态注册与构造层（项目分层 L1，只允许依赖 utils）。

包结构：
- ``catalog.py``：模型静态描述唯一事实来源（descriptor / 别名 / quantile 合同）；
- ``factory.py``：按 catalog 合同构造 wrapper 实例；
- ``wrappers/``：各引擎训练/预测本体（模板化 base + 9 个实现）；
- ``preflight/``：引擎参数预检收口（「未知参数 RAISE」统一执行层）；
- ``adapters/``：canonical ndarray 合同适配与原生序列模型接线；
- ``pickle_io.py``：模型与缩放器 pickle 保存/加载（部署侧消费）。

门面约定：本包根不 re-export 任何符号（消费方一律全路径导入，如
``from models.catalog import MODEL_CATALOG``）；子包门面各自约定，
见 ``models/preflight/__init__.py`` 与 ``models/adapters/__init__.py``。
"""
