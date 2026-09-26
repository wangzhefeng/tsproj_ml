# models

`models/` 承载底层 estimator factory、模型静态描述、原生参数预检与 pickle IO。

- `factory.py`：`ModelFactory` 与 catalog→wrapper 构造接线。配置别名：`lgb`/`lightgbm`、`xgb`/`xgboost`、`cat`/`catboost`、`rf`/`randomforest`、`histgb`/`histgradientboosting`、`ridge`、`enet`/`elasticnet`、`lasso`、`qr`/`quantileregressor`、`st`/`seasonaltemplate`、`ets`、`naive`、`theta`。
- `wrappers/`：按 family 保留原生模型封装；`base.py` 为共享基类/构造模板（`_resolve_params` → `_build_estimator` 钩子、`_require_fitted`、NaN 防御共享 `nan_defense_fit_state`），`lightgbm.py`、`xgboost.py`、`catboost.py` 为 boosting 封装，`sklearn_tree.py` 为 RF/HistGB，`linear.py` 为 Ridge/ElasticNet/Lasso/QuantileRegressor，`seasonal_template.py` 为 SeasonalTemplate（支持 `day_type_split` 与 `holiday_split` 按 `is_holiday` 列分组，后者优先且列缺失时回退），`ets.py`/`naive.py`/`theta.py` 为直接消费原始时序历史的原生序列模型。参数校验统一来自 `preflight/`，wrappers 只保留训练/预测本体。
- `pickle_io.py`：`ModelDeployPkl` 的底层 pickle 保存/加载，导入不修改 `sys.path`，也不再因未用 logger 导入而初始化项目日志；对象保存/加载行为不变。

训练在 `model_training/`，推理/产物在 `model_forecasting/`，生命周期编排在 `model_pipeline/`，稳定 bundle 合同在 `forecasting_core/artifacts.py`。本包不得反向 import 上述高层包。

## 描述表与参数校验

- `catalog.py`：`MODEL_CATALOG` / `ModelDescriptor` 是别名、wrapper、quantile、类别输入及线程参数等静态描述的唯一来源，不构造模型。
- `quantile_parameters()` 根据 catalog 注入原生分位数参数，拒绝非法分位点、未知模型和无 scalar quantile 能力的模型。
- `preflight/`：引擎参数预检收口（「未知参数 RAISE」合同的统一执行层）。`signature.py` 为通用构造/fit 签名过滤；`lightgbm.py` 为别名全集 introspection；`catboost.py` 为同义词归一化薄封装；`xgb_preflight.py` 在隔离子进程做 XGBoost 原生参数预检（不在并行拟合父进程捕获全局 warnings，worker 经 `Path(__file__)` 直跑）；`xgboost_estimator.py` 为父进程侧特征名提取与 wrapper fit 入口。新增引擎的预检在 `preflight/` 加模块，不进 wrappers。
- `adapters/`：canonical 合同适配层（2026-09-26 自 model_training/estimators 下沉）。`canonical.py` 把 ModelFactory 的 DataFrame 接口适配为 ``(N, K)`` ndarray fit/predict 合同（`ModelFactoryEstimator`，原 `_ModelFactoryEstimator`）、提供 `make_model_factory` quantile 注入与 `probe_native_multioutput` 行为探测，零 forecasting_core 依赖；`native_registry.py` 为原生序列模型注册表（ets/naive/theta 三件套合同接线，catalog 声明 `native_history` 漏接线 RAISE）。能力注册表（返回合同层 `EstimatorCapabilities` 实例）与依赖 checkpoint 的 `SharedMultiQuantilePool` 仍在 `model_training/estimators/capabilities.py`。
- `ModelFactory.resolve_model_params()` 与 `create_model()` 统一参数解析和构造。线性/RF/HistGB 按原生签名校验；LightGBM 使用原生 alias 表；CatBoost 先归一化显式参数再合并默认值，不覆盖用户显式 seed。

## 第三方私有 API 依赖清单（升级检查表）

以下私有接口被 `preflight/` 依赖（升级对应依赖版本时必须逐项复核；消费点已全部收口在该包）：

| 依赖 | 私有 API | 消费点 | 失效后果 |
|---|---|---|---|
| lightgbm | `lgb.basic._ConfigAliases._get_all_param_aliases()` | `preflight/lightgbm.py::validate_lgbm_params` | 参数白名单全集丢失，回退为 RuntimeError（显式报错，不静默） |
| catboost | `cab.core._process_synonyms()` | `preflight/catboost.py::process_synonym_params` | 同义参数（如 `iterations`/`n_estimators`）归一化失效，默认值可能压过用户显式别名 |
| xgboost | `xgboost.data.pandas_feature_info()` | `preflight/xgboost_estimator.py` 特征名提取 | XGB 预检的特征名维度校验失效 |

命名约束：`preflight/` 内作为子进程 worker 直跑的文件（当前仅 `xgb_preflight.py`）不得与任何第三方包同名——直跑时脚本目录位于 `sys.path[0]`，同名会遮蔽真实包。

另有一条**特征层隐式合同**：`SeasonalTemplateModel` 通过正则 `^.+_lag_\d+$` 自动识别特征矩阵的滞后列（命名来自 feature_engineering 的 lag 特征编译），无匹配列时 RAISE；上游改滞后列命名规范时此处会显式失败而非静默。

底层 pickle IO 的实际类名为 `ModelDeployPkl`，`load_model()` 返回加载对象；schema-2 bundle 的构造和生命周期验收由上层完成。pickle 只能读取可信产物，不应加载外部不可信文件。

## 原生序列模型合同（ets / naive / theta）

三者均属原生序列模型，不是监督回归 wrapper：拟合只接受带规则时间轴的单目标原始历史与显式 as-of 原点，不接受二维监督标签展平；统一实现 `fit_history(history, as_of, freq) / forecast(steps) / execution_evidence()` 三件套，规则时间网格、末端恰为 as_of、有限值、未知参数一律 RAISE。

- **ETS**（`ets.py`）：statsmodels 指数平滑，候选使用加性误差的无趋势/加性趋势/阻尼加性趋势（可选加性季节），按 BIC/AICc 多候选比选，记录每个候选的参数、收敛状态、失败原因与分数，全失败直接 RAISE。默认候选 `ANA/AAA/AAdA`（可显式选 `ANN`）；5min 每日季节周期默认 288，不降采样；heuristic 初始化避免把 288 个初始季节状态全部作为优化参数。
- **Naive**（`naive.py`）：M3/M4 标准闭式基线，`mode` 支持 `naive`（末值平推）/ `drift`（随机游走加漂移）/ `seasonal_naive`（季节同位置，`seasonal_periods` 生效）；纯 numpy 闭式计算、无估计参数、不存在不收敛问题；`seasonal_naive` 要求历史长度 ≥ 季节周期。
- **Theta**（`theta.py`）：statsmodels `ThetaModel`，经典 Theta 方法（theta=2 且带漂移形态等价于带漂移 SES）；单一 theta 线组合（`theta` 参数须 ≥ 1），加性季节去季节化（`deseasonalize`/`use_test`/`method`/`difference` 透传），估计失败直接 RAISE 不降级，warnings 与参数记入执行证据。

**SARIMA 不在本层**：日内高频季节周期（如 288）下 SARIMA 的季节 AR 多项式阶数不可行，工程上不提供该成员；低频月度场景如需再立项。LightGBM 多分位共享池同样未做：当前全部配置为 `mode: point`（quantile 消费者为零），待 quantile 场景立项。

`catalog.native_history` 与资源计划显式区分无监督特征的序列模型；runner 保留共用调度几何，但不使用监督 Y 拟合原生模型——`model_pipeline/runner.py::NATIVE_HISTORY_MODELS` 注册表按 model_type 分发（新增成员 = catalog 一条 descriptor + 注册表一条映射，漏接线 RAISE）。原生对象可以在可信 pickle 中往返恢复并从拟合原点续预测；不把对象往返视为 schema-2 部署 bundle 验收。严格原始历史窗口（历史来源：联通场景，已随 2026-09-25 收敛退役）仍只支持 backtest-only，final fit/bundle 拒绝合同保持不变。测试覆盖及执行入口见 `docs/testing/`；`.hermes/plans/` 下的实施证据仅保留在本地，不随仓库分发。

## Pickle 路径兼容边界

wrapper 类的持久化路径为 `models.wrappers.<family>`。按 D3 裁决，旧 `models.ModelFactory` 路径的存量 bundle 作废，不提供 shim 或自动迁移；需要重新训练生成新 bundle。此兼容边界不改变 YAML、schema-2 bundle 字段或预测数值，也不删除存量产物。
