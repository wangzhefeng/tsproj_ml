# models

`models/` 承载底层 estimator factory、模型静态描述、原生参数预检与 pickle IO。

- `factory.py`：`ModelFactory` 与 catalog→wrapper 构造接线。配置别名：`lgb`/`lightgbm`、`xgb`/`xgboost`、`cat`/`catboost`、`rf`/`randomforest`、`histgb`/`histgradientboosting`、`ridge`、`enet`/`elasticnet`、`lasso`、`qr`/`quantileregressor`、`st`/`seasonaltemplate`。
- `wrappers/`：按 family 保留原生模型封装；`base.py` 为共享基类/签名校验，`lightgbm.py`、`xgboost.py`、`catboost.py` 为 boosting 封装，`sklearn_tree.py` 为 RF/HistGB，`linear.py` 为 Ridge/ElasticNet/Lasso/QuantileRegressor，`seasonal_template.py` 为 SeasonalTemplate。
- `pickle_io.py`：`ModelDeployPkl` 的底层 pickle 保存/加载，导入不再修改 `sys.path`。

训练在 `model_training/`，推理/产物在 `model_forecasting/`，生命周期编排在 `model_pipeline/`，稳定 bundle 合同在 `forecasting_core/artifacts.py`。本包不得反向 import 上述高层包。

## 描述表与参数校验

- `catalog.py`：`MODEL_CATALOG` / `ModelDescriptor` 是别名、wrapper、quantile、类别输入及线程参数等静态描述的唯一来源，不构造模型。
- `quantile_parameters()` 根据 catalog 注入原生分位数参数，拒绝非法分位点、未知模型和无 scalar quantile 能力的模型。
- `xgb_validation.py`：`validate_xgb_parameters()` 在隔离子进程做 XGBoost 原生参数预检，不在并行拟合父进程捕获全局 warnings。
- `ModelFactory.resolve_model_params()` 与 `create_model()` 统一参数解析和构造。线性/RF/HistGB 按原生签名校验；LightGBM 使用原生 alias 表；CatBoost 先归一化显式参数再合并默认值，不覆盖用户显式 seed。

底层 pickle IO 的实际类名为 `ModelDeployPkl`，`load_model()` 返回加载对象；schema-2 bundle 的构造和生命周期验收由上层完成。pickle 只能读取可信产物，不应加载外部不可信文件。

## Pickle 路径兼容边界

wrapper 类的持久化路径为 `models.wrappers.<family>`。按 D3 裁决，旧 `models.ModelFactory` 路径的存量 bundle 作废，不提供 shim 或自动迁移；需要重新训练生成新 bundle。此兼容边界不改变 YAML、schema-2 bundle 字段或预测数值，也不删除存量产物。
