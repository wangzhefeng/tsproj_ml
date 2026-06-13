# features 模块说明

`features/` 负责把 `data_provider/` 输出的历史/未来时间序列表转换为模型可用的
特征矩阵和目标矩阵，并提供训练阶段可选的数据增强、特征选择、特征缩放和目标缩放。

## 文件职责

| 文件 | 职责 |
|---|---|
| `FeatureEngineering.py` | 外生特征、内生滞后特征、多步目标列和高级统计特征构造 |
| `FeatureScalering.py` | 特征缩放、目标缩放、目标逆变换和训练/预测 schema 对齐 |
| `FeatureSelection.py` | 基于 `SelectKBest` 的训练集特征选择，并在预测阶段复用选择结果 |
| `DataAugment.py` | 训练集 bootstrap + numeric jitter 数据增强 |

## FeatureEngineering

`FeatureEngineer.create_features()` 是主流程入口，由 `main.Model.run()` 在历史训练数据
和未来预测数据构造前后复用。输入通常包括：

- `df_series`：已统一 `time` 和目标列 `y` 的历史或未来序列表。
- `df_date_history` / `df_date_future`：日期类型外生数据。
- `df_weather_history` / `df_weather_future`：天气外生数据。
- `endogenous_features_with_target`：目标列和可用内生变量。
- `target_feature`：主目标列，通常为 `y`。
- `horizon`：预测步长，用于生成多步目标列。

返回值：

```text
df_series_featured
predictor_features
target_output_features
categorical_features
```

`predictor_features` 是模型输入列，`target_output_features` 是多步训练目标列，
`categorical_features` 会传给缩放器和模型训练层。

## 特征类型

外生特征：

- 日期时间特征：由 `enable_datetime_features` 和 `datetime_features` 控制。
- 日期类型特征：由 `enable_date_features`、`datetype_features` 控制。
- 天气特征：由 `enable_weather_features`、`weather_features` 控制。
- Direct 方法可选 horizon-aware 外生展开：`use_horizon_exogenous_for_direct`。
- Global 模式可透传 `series_id_feature` 作为静态类别特征。

内生基本特征：

- 单变量方法只使用目标 `y` 构造滞后特征。
- 多变量方法可使用配置中的其他内生变量。
- `enable_lags_features=True` 时按 `lags` 生成 `<col>_lag_<n>`。
- Direct 类方法会构造 `y_shift_1 ... y_shift_horizon` 这类多步目标列。

高级内生特征：

- 总开关：`enable_advanced_features`。
- 滚动统计：`enable_rolling_features`、`rolling_columns`、`rolling_windows`、`rolling_stats`。
- 扩展窗口统计：`enable_expanding_features`。
- 差分：`enable_diff_features`、`diff_columns`、`diff_periods`。
- 百分比变化：`enable_pct_change_features`。
- 距峰/谷时间：`enable_time_since_features`。
- 周期编码：`enable_cyclical_features`。
- 交互项：`enable_interaction_features`。
- 多项式项：`enable_polynomial_features`。

## FeatureScalering

`FeatureScaler` 处理预测特征 `X`：

- `scale_features` 控制是否缩放数值特征。
- `feature_scaler_type` 支持 `standard`、`minmax`。
- `encode_categorical_features=True` 时会对类别特征编码。
- 未编码的类别特征会转为 pandas `category` 类型，便于 LightGBM 等模型使用。
- 训练阶段会记录 schema，预测阶段按训练 schema 对齐列。

`TargetScaler` 处理训练目标 `Y`：

- `scale_target` 控制是否缩放目标。
- `target_scaler_type` 支持 `none`、`standard`、`minmax`、`log1p`、`robust`、`yeo-johnson`。
- `inverse_target` 控制评估和预测保存时是否恢复到目标原始量纲。
- 分位数和多输出预测会通过 `get_prediction_target_columns()` 对齐目标列。

兼容字段：

- `scale_features` 优先；旧字段 `scale` 仍作为 fallback。
- `scale_target` 优先；旧字段 `scale` 仍作为 fallback。
- `inverse_target` 优先；旧字段 `inverse` 仍作为 fallback。

## FeatureSelection

`FeatureSelector` 只应在训练阶段 fit，然后在测试/预测阶段复用 `selected_features_`。

配置字段：

- `enable_feature_selection`
- `feature_selection_method`: `f_regression` 或 `mutual_info`
- `feature_selection_max_features`
- `feature_selection_min_features`

选择边界：

- 当前选择器会把非数值列转为可计算形式后评分。
- 类别特征和 `force_keep_features` 会尽量保留。
- 如果原始特征数不超过 `min_features`，不会裁剪。

## DataAugment

`TimeSeriesAugmenter` 只用于训练集增强，不用于测试真实值或未来预测数据。

配置字段：

- `enable_data_augmentation`
- `augmentation_ratio`
- `augmentation_feature_noise_std`
- `augmentation_target_noise_std`
- `augmentation_random_state`

增强方式：

- 从训练样本 bootstrap 抽样。
- 仅对数值预测特征添加小幅噪声，类别特征保持不变。
- 对数值目标列添加更小幅度噪声。
- 样本数少于 10、比例为 0 或开关关闭时直接返回原始 `(X, y)`。

## 与其他模块的边界

- `features/` 不读取 CSV；输入来自 `data_provider.DataLoader`。
- `features/` 不训练模型；训练由 `models.ModelTraining.Trainer` 完成。
- `features/` 不决定测试窗口；窗口切分由 `models.ModelTesting.Tester` 完成。
- 目标列在进入特征工程前通常已由 `DataLoader` 统一为 `y`。
- 预测阶段必须复用训练阶段的缩放器、特征选择结果和特征 schema。

## 验证

改动特征工程、缩放、选择或增强逻辑后至少运行：

```bash
uv run python -m py_compile features/FeatureEngineering.py features/FeatureScalering.py features/FeatureSelection.py features/DataAugment.py
uv run python -m unittest discover -s tests -p "test_*.py"
```
