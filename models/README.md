# models

`models/` 只保留底层 estimator factory 与 pickle IO。

- `ModelFactory.py`：LightGBM、XGBoost、CatBoost、RandomForest、HistGB、Ridge、ElasticNet、Lasso、QuantileRegressor、SeasonalTemplate。
- `ModelSaveLoad.py`：底层 pickle 保存/加载。

训练在 `model_training/`，推理/产物编排在 `model_forecasting/`，稳定 bundle 合同在 `forecasting_core/artifacts.py`。本包不得反向 import 上述高层包。
