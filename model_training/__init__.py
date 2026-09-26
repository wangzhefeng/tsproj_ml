"""监督回归训练行为层（项目分层 L2，依赖 forecasting_core 与 models）。

包结构：
- ``trainer.py``：``CanonicalTrainer``，按策略调用组织训练，产出
  ``CanonicalStrategyArtifact``（不拥有结果 IO 与部署 bundle）；
- ``quantile.py``：``CanonicalMarginalQuantileTrainer``，按 quantile
  grid 逐 level 训练编排（level 间线程并行，与串行数值完全一致）；
- ``weights/``：训练样本权重子包（``temporal.py`` 指数衰减算法本体 +
  ``resolve.py`` 配置接线与能力前置校验），由编排层拟合入口消费；
- ``strategies/``：七种标准多步策略 executor（base 承载 target plan
  与共享预测循环，子类零代码声明 strategy_name）；
- ``estimators/``：能力注册表、多分位共享池与三种 multi-target
  adapter（independent / regressor-chain / native）。

门面约定：与 ``models`` 包一致，本包根不 re-export 任何符号（消费方
一律全路径导入，如 ``from model_training.trainer import CanonicalTrainer``）。
"""
