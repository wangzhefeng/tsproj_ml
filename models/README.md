# models

`models/` 承载底层 estimator factory、模型静态描述、原生参数预检与 pickle IO。

- `factory.py`：`ModelFactory` 与 catalog→wrapper 构造接线。配置别名：`lgb`/`lightgbm`、`xgb`/`xgboost`、`cat`/`catboost`、`rf`/`randomforest`、`histgb`/`histgradientboosting`、`ridge`、`enet`/`elasticnet`、`lasso`、`qr`/`quantileregressor`、`st`/`seasonaltemplate`、`ets`。
- `wrappers/`：按 family 保留原生模型封装；`base.py` 为共享基类/签名校验，`lightgbm.py`、`xgboost.py`、`catboost.py` 为 boosting 封装，`sklearn_tree.py` 为 RF/HistGB，`linear.py` 为 Ridge/ElasticNet/Lasso/QuantileRegressor，`seasonal_template.py` 为 SeasonalTemplate，`ets.py` 为直接消费原始时序历史的 ETS。
- `pickle_io.py`：`ModelDeployPkl` 的底层 pickle 保存/加载，导入不再修改 `sys.path`。

训练在 `model_training/`，推理/产物在 `model_forecasting/`，生命周期编排在 `model_pipeline/`，稳定 bundle 合同在 `forecasting_core/artifacts.py`。本包不得反向 import 上述高层包。

## 描述表与参数校验

- `catalog.py`：`MODEL_CATALOG` / `ModelDescriptor` 是别名、wrapper、quantile、类别输入及线程参数等静态描述的唯一来源，不构造模型。
- `quantile_parameters()` 根据 catalog 注入原生分位数参数，拒绝非法分位点、未知模型和无 scalar quantile 能力的模型。
- `xgb_validation.py`：`validate_xgb_parameters()` 在隔离子进程做 XGBoost 原生参数预检，不在并行拟合父进程捕获全局 warnings。
- `ModelFactory.resolve_model_params()` 与 `create_model()` 统一参数解析和构造。线性/RF/HistGB 按原生签名校验；LightGBM 使用原生 alias 表；CatBoost 先归一化显式参数再合并默认值，不覆盖用户显式 seed。

底层 pickle IO 的实际类名为 `ModelDeployPkl`，`load_model()` 返回加载对象；schema-2 bundle 的构造和生命周期验收由上层完成。pickle 只能读取可信产物，不应加载外部不可信文件。

## 原生 ETS 合同

ETS 属于原生序列模型，不是监督回归 wrapper：拟合只接受带规则时间轴的单目标原始历史与显式 as-of 原点，不接受二维监督标签展平。候选使用 statsmodels 原生指数平滑，限定加性误差的无趋势、加性趋势和阻尼加性趋势，可选择加性季节项；按明确的信息准则选择，记录每个候选的参数、收敛状态、失败原因与分数，全失败直接 RAISE。5min 每日季节周期保持 288，不降采样。未知参数、非有限值、缺失/重复/乱序时间及未来历史直接拒绝。

默认候选 `ANA/AAA/AAdA`（可显式选择 `ANN`），默认 BIC，也支持 AICc；最多四个互异候选，默认每候选 300 次迭代、上限 1000。采用历史内 heuristic 初始化，避免把 288 个初始季节状态全部作为优化参数；每次拟合重新初始化，不复用其他折状态。参考 M5 ES_bu 的自动指数平滑思想，不是零售层级方法复现。

`catalog.native_history` 与资源计划显式区分无监督特征的序列模型；runner 保留共用调度几何，但不使用监督 Y 拟合 ETS。原生对象可以在可信 pickle 中往返恢复并从拟合原点续预测；不把对象往返视为 schema-2 部署 bundle 验收。联通严格原始历史窗口仍只支持 backtest-only，final fit/bundle 拒绝合同保持不变。测试覆盖及执行入口见 `tests/README.md`；`.hermes/plans/` 下的实施证据仅保留在本地，不随仓库分发。

## Pickle 路径兼容边界

wrapper 类的持久化路径为 `models.wrappers.<family>`。按 D3 裁决，旧 `models.ModelFactory` 路径的存量 bundle 作废，不提供 shim 或自动迁移；需要重新训练生成新 bundle。此兼容边界不改变 YAML、schema-2 bundle 字段或预测数值，也不删除存量产物。
