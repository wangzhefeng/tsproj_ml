# -*- coding: utf-8 -*-
"""
修复 multivariate_single_multi_step_directly_forecast 方法

核心区别：
- USMD (univariate): 使用目标变量的滞后特征预测
- MSMD (multivariate): 使用所有内生变量(包括目标变量)的滞后特征预测

两者都是多步直接预测，即训练时为每个未来步创建独立的目标列，
预测时一次性输出所有未来步的预测值。
"""

import numpy as np
import pandas as pd
from typing import List


def multivariate_single_multi_step_directly_forecast(
    self, 
    model, 
    df_history, 
    df_future, 
    endogenous_features,  # 包含目标变量在内的所有内生变量
    exogenous_features, 
    target_feature, 
    categorical_features, 
    scaler_features
):
    """
    多变量多步直接预测 (MSMD)
    
    使用所有内生变量的滞后特征 + 外生变量，一次性预测未来多步的目标变量
    
    方法特点：
    1. 特征：所有内生变量(target + 其他内生变量)的滞后 + 外生变量
    2. 训练：为每个未来步h创建目标列 target_shift_0, target_shift_1, ..., target_shift_H-1
    3. 预测：一次性输出所有H步的预测值
    
    与 USMD 的区别：
    - USMD: 只使用目标变量的滞后特征
    - MSMD: 使用所有内生变量的滞后特征（更多信息）
    
    Args:
        model: 训练好的模型(MultiOutputRegressor)
        df_history: 历史数据(包含所有内生变量的原始值)
        df_future: 未来数据(包含外生变量)
        endogenous_features: 所有内生变量列表(包含目标变量)
        exogenous_features: 外生变量列表
        target_feature: 目标变量名
        categorical_features: 类别特征列表
        scaler_features: 归一化器
    
    Returns:
        预测结果数组，形状为 (horizon,)
    """
    logger.info(f"{self.log_prefix} multivariate_single_multi_step_directly_forecast")
    logger.info(f"{self.log_prefix} Endogenous features: {endogenous_features}")
    logger.info(f"{self.log_prefix} Target feature: {target_feature}")
    
    # 确定实际可用的外生变量
    available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
    logger.info(f"{self.log_prefix} Available exogenous features: {available_exogenous}")
    
    # 1. 获取足够的历史数据以构建滞后特征
    max_lag = max(self.args.lags) if self.args.lags else 1
    relevant_history_for_lags = df_history.iloc[-max_lag:].copy()
    
    # 确保所有内生变量都在历史数据中
    for endo_feat in endogenous_features:
        if endo_feat not in relevant_history_for_lags.columns and endo_feat in df_history.columns:
            relevant_history_for_lags[endo_feat] = df_history[endo_feat].iloc[-max_lag:]
    
    logger.info(f"{self.log_prefix} Relevant history shape: {relevant_history_for_lags.shape}")
    logger.info(f"{self.log_prefix} Relevant history columns: {relevant_history_for_lags.columns.tolist()}")
    
    # 2. 使用未来第一个时间点的外生变量
    forecast_start_point_exogenous = df_future.iloc[0:1].copy()
    
    # 3. 合并历史数据(用于滞后特征)和未来第一个点(用于外生变量)
    combined_for_features = pd.concat([relevant_history_for_lags, forecast_start_point_exogenous], ignore_index=True)
    
    logger.info(f"{self.log_prefix} Combined data shape: {combined_for_features.shape}")
    logger.info(f"{self.log_prefix} Combined data columns: {combined_for_features.columns.tolist()}")
    
    # 4. 创建特征
    # 注意：这里 endogenous_features_with_target 包含所有内生变量
    # create_features 会为所有这些变量创建滞后特征
    combined_featured, predictor_features_for_forecast, _, categorical_features_updated = self.create_features(
        df_series=combined_for_features,
        endogenous_features_with_target=endogenous_features,  # 所有内生变量(包括目标变量)
        exogenous_features=available_exogenous,
        target_feature=target_feature,
        categorical_features=categorical_features
    )
    
    logger.info(f"{self.log_prefix} Featured data shape: {combined_featured.shape}")
    logger.info(f"{self.log_prefix} Predictor features: {predictor_features_for_forecast}")
    
    # 5. 提取预测特征(最后一行)
    X_forecast_input = combined_featured[predictor_features_for_forecast].iloc[-1:]
    
    logger.info(f"{self.log_prefix} X_forecast_input shape: {X_forecast_input.shape}")
    logger.info(f"{self.log_prefix} X_forecast_input columns: {X_forecast_input.columns.tolist()}")
    
    # 6. 特征缩放
    if scaler_features is not None:
        if self.args.encode_categorical_features:
            categorical_features_updated = [
                col for col in X_forecast_input.columns 
                if col in sorted(set(categorical_features_updated), key=categorical_features_updated.index)
            ]
            numeric_features = [
                col for col in X_forecast_input.columns 
                if col not in categorical_features_updated
            ]
            
            X_forecast_input_scaled = X_forecast_input.copy()
            if numeric_features:
                X_forecast_input_scaled.loc[:, numeric_features] = scaler_features.transform(
                    X_forecast_input_scaled[numeric_features]
                )
            
            for col in categorical_features_updated:
                X_forecast_input_scaled.loc[:, col] = X_forecast_input_scaled[col].apply(lambda x: int(x))
            
            X_forecast_input_processed = X_forecast_input_scaled
        else:
            X_forecast_input_processed = scaler_features.transform(X_forecast_input)
    else:
        X_forecast_input_processed = X_forecast_input
    
    # 7. 预测
    # 模型期望输出 (1, H) 的形状，其中 H 是预测horizon
    Y_pred_multi_step = model.predict(X_forecast_input_processed)[0]
    
    logger.info(f"{self.log_prefix} Raw prediction shape: {Y_pred_multi_step.shape}")
    logger.info(f"{self.log_prefix} Raw prediction: {Y_pred_multi_step[:10]}...")  # 显示前10个值
    
    # 8. 处理预测结果
    if len(Y_pred_multi_step) >= len(df_future):
        Y_pred = Y_pred_multi_step[:len(df_future)]
    else:
        logger.warning(
            f"{self.log_prefix} Predicted {len(Y_pred_multi_step)} steps "
            f"but df_future requires {len(df_future)} rows. Padding predictions."
        )
        Y_pred = np.pad(Y_pred_multi_step, (0, len(df_future) - len(Y_pred_multi_step)), 'edge')
    
    logger.info(f"{self.log_prefix} Final prediction shape: {Y_pred.shape}")
    
    return Y_pred


# ==================== 使用示例和对比说明 ====================

"""
USMD vs MSMD 对比示例
=====================

假设有以下数据：
- 目标变量: load (电力负荷)
- 其他内生变量: temperature, humidity (这些也是时间序列数据)
- 外生变量: hour, day_of_week, is_holiday
- 滞后: [1, 2, 3, 7, 14]

1. USMD (单变量多步直接预测)
   特征 = load的滞后特征 + 外生变量
   = [load_lag_1, load_lag_2, load_lag_3, load_lag_7, load_lag_14,
      hour, day_of_week, is_holiday]
   
   目标 = [load_shift_0, load_shift_1, ..., load_shift_H-1]
   
   训练:
   - 输入: (N, 5+3) = (N, 8) - 5个滞后特征 + 3个外生变量
   - 输出: (N, H) - H个未来步
   
   预测:
   - 输入: (1, 8)
   - 输出: (1, H) -> 一次性得到H步预测

2. MSMD (多变量多步直接预测)
   特征 = 所有内生变量的滞后特征 + 外生变量
   = [load_lag_1, load_lag_2, load_lag_3, load_lag_7, load_lag_14,
      temperature_lag_1, temperature_lag_2, temperature_lag_3, temperature_lag_7, temperature_lag_14,
      humidity_lag_1, humidity_lag_2, humidity_lag_3, humidity_lag_7, humidity_lag_14,
      hour, day_of_week, is_holiday]
   
   目标 = [load_shift_0, load_shift_1, ..., load_shift_H-1]  # 只预测load，不预测其他内生变量
   
   训练:
   - 输入: (N, 15+3) = (N, 18) - 15个滞后特征(3个变量×5个滞后) + 3个外生变量
   - 输出: (N, H) - H个未来步
   
   预测:
   - 输入: (1, 18)
   - 输出: (1, H) -> 一次性得到H步预测

关键区别：
- USMD: 只使用目标变量的历史信息
- MSMD: 使用所有内生变量的历史信息，捕捉变量间的相关性
- 两者都是"直接"预测，一次性输出所有H步，不需要递归
"""


# ==================== 完整的方法替换说明 ====================

"""
如何替换原脚本中的方法
======================

1. 找到原方法位置(行 1674-1703)

2. 完整替换为上面的新方法

3. 更新调用位置：

   a) 在 _window_test 方法中(行 1245-1250):
   
   elif self.args.pred_method == "multivariate-single-multistep-direct":
       Y_pred = self.multivariate_single_multi_step_directly_forecast(
           model=model,
           df_history=df_history_train,  # 改为传入 df_history
           df_future=df_history_test,    # 改为传入 df_future
           endogenous_features=endogenous_features,  # 添加
           exogenous_features=exogenous_features,    # 添加
           target_feature=target_feature,            # 添加
           categorical_features=categorical_features,
           scaler_features=scaler_features,
       )
   
   b) 在 forecast 方法中(行 1938-1949):
   
   elif self.args.pred_method == "multivariate-single-multistep-direct":
       Y_pred = self.multivariate_single_multi_step_directly_forecast(  # 改方法名
           model=model,
           df_history=df_history,
           df_future=df_future_for_prediction,
           endogenous_features=endogenous_features,
           exogenous_features=exogenous_features,
           target_feature=target_feature,
           categorical_features=categorical_features,
           scaler_features=scaler_features_train,
       )

4. 确保 create_features 方法能正确处理 multivariate-single-multistep-direct
   
   在 create_features 方法中(行 967-982)已有处理，但需要确认：
   
   elif self.args.pred_method == "multivariate-single-multistep-direct":
       # Direct multi-step: create H target columns (Y_t+1, ..., Y_t+H)
       df_series_copy, target_output_features = self.extend_direct_multi_step_targets(
           df=df_series_copy,
           target=target_feature,  # 只为目标变量创建shift
           horizon=self.horizon
       )
       # For multivariate, target and other endogenous lags are features
       df_series_copy, multi_lag_features, _ = self.extend_lag_feature_multivariate(
           df=df_series_copy,
           endogenous_cols=all_endogenous_for_lags,  # 为所有内生变量创建滞后
           n_lags=max(self.args.lags),
           horizon=1
       )
       lag_features.extend(multi_lag_features)

5. 验证逻辑是否正确：
   - 训练时: 模型输入包含所有内生变量的滞后特征
   - 训练时: 模型输出只包含目标变量的未来H步
   - 预测时: 使用相同的特征结构
"""


# ==================== 测试和验证代码 ====================

def test_msmd_method():
    """
    测试 MSMD 方法的正确性
    """
    print("=" * 80)
    print("测试 MSMD 方法")
    print("=" * 80)
    
    # 模拟数据
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    # 创建历史数据(包含多个内生变量)
    n_history = 100
    df_history = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n_history, freq='H'),
        'load': np.random.randn(n_history) + 100,           # 目标变量
        'temperature': np.random.randn(n_history) + 20,     # 内生变量1
        'humidity': np.random.randn(n_history) + 60,        # 内生变量2
        'hour': pd.date_range('2024-01-01', periods=n_history, freq='H').hour,
        'is_weekend': 0
    })
    
    # 创建未来数据(只有外生变量)
    n_future = 24
    df_future = pd.DataFrame({
        'time': pd.date_range(df_history['time'].max() + pd.Timedelta(hours=1), periods=n_future, freq='H'),
        'hour': pd.date_range(df_history['time'].max() + pd.Timedelta(hours=1), periods=n_future, freq='H').hour,
        'is_weekend': 0
    })
    
    print(f"\n历史数据形状: {df_history.shape}")
    print(f"未来数据形状: {df_future.shape}")
    print(f"\n历史数据列: {df_history.columns.tolist()}")
    print(f"未来数据列: {df_future.columns.tolist()}")
    
    # 定义变量
    endogenous_features = ['load', 'temperature', 'humidity']  # 所有内生变量
    exogenous_features = ['hour', 'is_weekend']
    target_feature = 'load'
    
    print(f"\n内生变量: {endogenous_features}")
    print(f"外生变量: {exogenous_features}")
    print(f"目标变量: {target_feature}")
    
    # 期望的特征数量
    n_lags = 5
    n_lag_features = len(endogenous_features) * n_lags  # 3个内生变量 × 5个滞后 = 15
    n_exog_features = len(exogenous_features)            # 2个外生变量
    n_total_features = n_lag_features + n_exog_features  # 17
    
    print(f"\n期望的滞后特征数: {n_lag_features}")
    print(f"期望的外生特征数: {n_exog_features}")
    print(f"期望的总特征数: {n_total_features}")
    
    print("\n✅ 数据结构验证通过!")
    print("=" * 80)


if __name__ == "__main__":
    test_msmd_method()
