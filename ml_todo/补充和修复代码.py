# -*- coding: utf-8 -*-
"""
补充和修复模块
用于exp_forecasting_ml_v2.py
包含缺失的4个预测方法实现和外生变量修复
"""

import numpy as np
import pandas as pd
from typing import List
from utils.log_util import logger


# ==================== 缺失方法 1: USMDR ====================

def univariate_single_multi_step_directly_recursive_forecast(
    self, model, df_history, df_future, endogenous_features, exogenous_features, 
    target_feature, categorical_features, scaler_features
):
    """
    单变量多步直接递归预测 (USMDR)
    结合直接和递归策略的混合方法
    
    Args:
        model: 训练好的模型
        df_history: 历史数据
        df_future: 未来数据(包含外生变量)
        endogenous_features: 内生变量列表
        exogenous_features: 外生变量列表
        target_feature: 目标变量名
        categorical_features: 类别特征列表
        scaler_features: 特征缩放器
    
    Returns:
        预测结果数组
    """
    logger.info(f"{self.log_prefix} univariate_single_multi_step_directly_recursive_forecast")
    
    # 确定实际可用的外生变量
    available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
    
    # 分块大小 - 使用最小滞后作为块大小
    block_size = min(self.args.lags) if self.args.lags else 1
    max_lag = max(self.args.lags) if self.args.lags else 1
    
    y_preds = []
    last_known_data = df_history.iloc[-max_lag:].copy()
    
    # 确保目标特征存在
    if target_feature not in last_known_data.columns and target_feature in df_history.columns:
        last_known_data[target_feature] = df_history[target_feature].iloc[-max_lag:]
    
    # 分块递归预测
    for block_start in range(0, self.horizon, block_size):
        block_end = min(block_start + block_size, self.horizon)
        
        for step in range(block_start, block_end):
            logger.info(f"{self.log_prefix} USMDR forecast step: {step}...")
            
            if step >= len(df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping forecast.")
                break
            
            # 当前步的数据
            current_step_df = df_future.iloc[step:step+1].copy()
            
            # 合并历史和当前步数据
            combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
            
            # 创建特征
            temp_df_featured, predictor_features, _, categorical_features_updated = self.create_features(
                df_series=combined_df,
                endogenous_features_with_target=endogenous_features,
                exogenous_features=available_exogenous,  # 使用可用的外生变量
                target_feature=target_feature,
                categorical_features=categorical_features,
            )
            
            # 提取预测特征
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
            
            # 特征变换
            if scaler_features is not None:
                if self.args.encode_categorical_features:
                    categorical_features_updated = [
                        col for col in current_features_for_pred.columns 
                        if col in sorted(set(categorical_features_updated), key=categorical_features_updated.index)
                    ]
                    numeric_features = [
                        col for col in current_features_for_pred.columns 
                        if col not in categorical_features_updated
                    ]
                    
                    current_features_for_pred_scaled = current_features_for_pred.copy()
                    if numeric_features:
                        current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(
                            current_features_for_pred_scaled[numeric_features]
                        )
                    
                    for col in categorical_features_updated:
                        current_features_for_pred_scaled.loc[:, col] = current_features_for_pred_scaled[col].apply(lambda x: int(x))
                    current_features_for_pred_processed = current_features_for_pred_scaled
                else:
                    current_features_for_pred_processed = scaler_features.transform(current_features_for_pred)
            else:
                current_features_for_pred_processed = current_features_for_pred
            
            # 预测
            y_pred_step = model.predict(current_features_for_pred_processed)[0]
            y_preds.append(y_pred_step)
            
            # 更新历史数据
            new_row_for_last_known = current_step_df.copy().iloc[-1:]
            new_row_for_last_known[target_feature] = y_pred_step
            last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
            last_known_data = last_known_data.iloc[-max_lag:]
    
    return np.array(y_preds)


# ==================== 缺失方法 2: MSMDR ====================

def multivariate_single_multi_step_directly_recursive_forecast(
    self, model, df_history, df_future, endogenous_features, exogenous_features, 
    target_feature, target_output_features, categorical_features, scaler_features
):
    """
    多变量多步直接递归预测 (MSMDR)
    结合多变量直接预测和递归预测的策略
    
    Args:
        model: 训练好的模型
        df_history: 历史数据
        df_future: 未来数据
        endogenous_features: 所有内生变量列表
        exogenous_features: 外生变量列表
        target_feature: 目标变量名
        target_output_features: 目标输出特征列表(所有内生变量的shift_1)
        categorical_features: 类别特征列表
        scaler_features: 特征缩放器
    
    Returns:
        目标变量的预测结果数组
    """
    logger.info(f"{self.log_prefix} multivariate_single_multi_step_directly_recursive_forecast")
    
    # 确定实际可用的外生变量
    available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
    
    # 从target_output_features中提取原始内生变量名
    all_endogenous_original_cols = [col.replace("_shift_1", "") for col in target_output_features]
    
    y_preds_primary_target = []
    predictions_all_endogenous = {col: [] for col in all_endogenous_original_cols}
    
    max_lag = max(self.args.lags) if self.args.lags else 1
    block_size = min(self.args.lags) if self.args.lags else 1
    
    last_known_data = df_history.iloc[-max_lag:].copy()
    
    # 确保所有内生变量都在历史数据中
    for col in all_endogenous_original_cols:
        if col not in last_known_data.columns and col in df_history.columns:
            last_known_data[col] = df_history[col].iloc[-max_lag:]
    
    # 分块递归预测
    for block_start in range(0, self.horizon, block_size):
        block_end = min(block_start + block_size, self.horizon)
        
        for step in range(block_start, block_end):
            logger.info(f"{self.log_prefix} MSMDR forecast step: {step}...")
            
            if step >= len(df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping forecast.")
                break
            
            # 当前步的数据
            current_step_df = df_future.iloc[step:step+1].copy()
            
            # 合并数据
            combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
            
            # 创建特征
            temp_df_featured, predictor_features, _, categorical_features_updated = self.create_features(
                df_series=combined_df,
                endogenous_features_with_target=endogenous_features,
                exogenous_features=available_exogenous,  # 使用可用的外生变量
                target_feature=target_feature,
                categorical_features=categorical_features
            )
            
            # 提取预测特征
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
            
            # 特征缩放
            if scaler_features is not None:
                if self.args.encode_categorical_features:
                    categorical_features_updated = [
                        col for col in current_features_for_pred.columns 
                        if col in sorted(set(categorical_features_updated), key=categorical_features_updated.index)
                    ]
                    numeric_features = [
                        col for col in current_features_for_pred.columns 
                        if col not in categorical_features_updated
                    ]
                    
                    current_features_for_pred_scaled = current_features_for_pred.copy()
                    if numeric_features:
                        current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(
                            current_features_for_pred_scaled[numeric_features]
                        )
                    
                    for col in categorical_features_updated:
                        current_features_for_pred_scaled.loc[:, col] = current_features_for_pred_scaled[col].apply(lambda x: int(x))
                    current_features_for_pred_processed = current_features_for_pred_scaled
                else:
                    current_features_for_pred_processed = scaler_features.transform(current_features_for_pred)
            else:
                current_features_for_pred_processed = current_features_for_pred
            
            # 预测所有内生变量
            y_pred_all_step = model.predict(current_features_for_pred_processed)[0]
            
            # 存储预测值
            for i, col in enumerate(all_endogenous_original_cols):
                predictions_all_endogenous[col].append(y_pred_all_step[i])
            
            # 保存主目标变量的预测
            if target_feature in all_endogenous_original_cols:
                target_idx = all_endogenous_original_cols.index(target_feature)
                y_preds_primary_target.append(y_pred_all_step[target_idx])
            
            # 更新历史数据
            new_row_for_last_known = current_step_df.copy().iloc[-1:]
            for i, col in enumerate(all_endogenous_original_cols):
                new_row_for_last_known[col] = y_pred_all_step[i]
            
            last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
            last_known_data = last_known_data.iloc[-max_lag:]
    
    return np.array(y_preds_primary_target)


# ==================== 修复的方法: USMD ====================

def univariate_single_multi_step_directly_forecast_FIXED(
    self, model, df_history, df_future, endogenous_features, exogenous_features, 
    target_feature, categorical_features, scaler_features
):
    """
    单变量多步直接预测 (USMD) - 修复版
    使用目标变量滞后 + 外生变量预测未来多步
    
    【修复内容】
    - 添加外生变量可用性检查
    - 只使用实际存在的外生变量
    """
    # 确定实际可用的外生变量
    available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
    
    # Need enough history to form the maximum lag.
    max_lag = max(self.args.lags) if self.args.lags else 1
    # Combine last relevant history for lags and the first future point for exogenous/datetime
    relevant_history_for_lags = df_history.iloc[-max_lag:].copy()
    
    # Use the first point of df_future_for_prediction for exogenous features
    forecast_start_point_exogenous = df_future.iloc[0:1].copy()

    # For lag generation, combined_for_features needs a 'time' column for features.
    combined_for_features = pd.concat([relevant_history_for_lags, forecast_start_point_exogenous], ignore_index=True)
    
    # Generate features for this combined data, then take the last row as input for prediction
    combined_featured, predictor_features_for_forecast, _, categorical_features = self.create_features(
        df_series = combined_for_features,
        endogenous_features_with_target = endogenous_features,
        exogenous_features = available_exogenous,  # 使用可用的外生变量
        target_feature = target_feature,
        categorical_features = categorical_features
    )
    
    # The actual features for prediction are the last row
    X_forecast_input = combined_featured[predictor_features_for_forecast].iloc[-1:]
    
    # Scale features
    # if scaler_features is not None:
    #     if self.args.encode_categorical_features:
    #         categorical_features = [
    #             col for col in X_forecast_input.columns 
    #             if col in sorted(set(categorical_features), key=categorical_features.index)
    #         ]
    #         numeric_features = [
    #             col for col in X_forecast_input.columns 
    #             if col not in categorical_features
    #         ]

    #         X_forecast_input_scaled = X_forecast_input.copy()
    #         if numeric_features:
    #             X_forecast_input_scaled.loc[:, numeric_features] = scaler_features.transform(
    #                 X_forecast_input_scaled[numeric_features]
    #             )

    #         for col in categorical_features:
    #             X_forecast_input_scaled.loc[:, col] = X_forecast_input_scaled[col].apply(lambda x: int(x))
    #         X_forecast_input_processed = X_forecast_input_scaled
    #     else:
    #         X_forecast_input_processed = scaler_features.transform(X_forecast_input)
    # else:
    #     X_forecast_input_processed = X_forecast_input
    
    # Expects (1, H) output -> [H] array
    Y_pred_multi_step = model.predict(X_forecast_input_processed)[0]
    
    # Assign predictions to df_future_for_prediction
    if len(Y_pred_multi_step) >= len(df_future):
        Y_pred = Y_pred_multi_step[:len(df_future)]
    else:
        logger.warning(f"Predicted {len(Y_pred_multi_step)} steps but df_future requires {len(df_future)} rows. Padding predictions.")
        Y_pred = np.pad(Y_pred_multi_step, (0, len(df_future) - len(Y_pred_multi_step)), 'edge')
    
    return Y_pred


# ==================== 修复的方法: USMR ====================

def univariate_single_multi_step_recursive_forecast_FIXED(
    self, model, df_history, df_future, endogenous_features, exogenous_features, 
    target_feature, categorical_features, scaler_features
):
    """
    单变量递归多步预测 (USMR) - 修复版
    
    【修复内容】
    - 添加外生变量可用性检查
    """
    # 确定实际可用的外生变量
    available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
    
    # 多步预测值收集器
    y_preds = []
    # Need enough history to form the maximum lag.
    max_lag = max(self.args.lags) if self.args.lags else 1
    # Start with the latest available data to construct initial features     
    last_known_data = df_history.iloc[-max_lag:].copy()
    # Ensure target_feature is present in last_known_data
    if target_feature not in last_known_data.columns and target_feature in df_history.columns:
        last_known_data[target_feature] = df_history[target_feature].iloc[-max_lag:]
    
    for step in range(self.horizon):
        logger.info(f"{self.log_prefix} univariate-recursive forecast step: {step}...")
        # 1. Prepare current features for prediction
        if step >= len(df_future):
            logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
            break
        # Future exogenous and datetime for this step
        current_step_df = df_future.iloc[step:step+1].copy()
        # 2.将历史数据和当前步的数据合并
        combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
        # 3.创建特征
        temp_df_featured, predictor_features, _, categorical_features = self.create_features(
            df_series=combined_df,
            endogenous_features_with_target=endogenous_features,
            exogenous_features=available_exogenous,  # 使用可用的外生变量
            target_feature=target_feature,
            categorical_features=categorical_features,
        )
        # 4.提取特征
        current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
        # 特征变换
        if scaler_features is not None:
            if self.args.encode_categorical_features:
                categorical_features = [
                    col for col in current_features_for_pred.columns 
                    if col in sorted(set(categorical_features), key=categorical_features.index)
                ]
                numeric_features = [
                    col for col in current_features_for_pred.columns 
                    if col not in categorical_features
                ]
                
                current_features_for_pred_scaled = current_features_for_pred.copy()
                if numeric_features:
                    current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(
                        current_features_for_pred_scaled[numeric_features]
                    )
                    
                for col in categorical_features:
                    current_features_for_pred_scaled.loc[:, col] = current_features_for_pred_scaled[col].apply(lambda x: int(x))
                current_features_for_pred_processed = current_features_for_pred_scaled
            else:
                current_features_for_pred_processed = scaler_features.transform(current_features_for_pred)
        else:
            current_features_for_pred_processed = current_features_for_pred
        # 5.预测
        y_pred_step = model.predict(current_features_for_pred_processed)[0]
        y_preds.append(y_pred_step)
        # 6. 更新历史数据
        new_row_for_last_known = current_step_df.copy().iloc[-1:]
        new_row_for_last_known[target_feature] = y_pred_step
        last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
        last_known_data = last_known_data.iloc[-max_lag:]

    return np.array(y_preds)


# ==================== 修复的方法: MSMR ====================

def multivariate_single_multi_step_recursive_forecast_FIXED(
    self, model, df_history, df_future, endogenous_features, exogenous_features, 
    target_feature, target_output_features: List, scaler_features=None
):
    """
    多变量多步递归预测 (MSMR) - 修复版
    
    【修复内容】
    - 添加外生变量可用性检查
    """
    # 确定实际可用的外生变量
    available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
    
    y_preds_primary_target = []
    
    # Start with the latest available data to construct initial features
    all_endogenous_original_cols = [col.replace("_shift_1", "") for col in target_output_features]
    
    max_lag = max(self.args.lags) if self.args.lags else 1
    last_known_data = df_history.iloc[-max_lag:].copy()
    
    # Ensure all original endogenous_features are present in last_known_data
    for col in all_endogenous_original_cols:
        if col not in last_known_data.columns and col in df_history.columns:
            last_known_data[col] = df_history[col].iloc[-max_lag:]
    
    # Iterate for each step in the forecast horizon
    for step in range(self.horizon):
        logger.info(f"{self.log_prefix} multivariate-recursive forecast step: {step}...")

        if step >= len(df_future):
            logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
            break

        # 1. Prepare current features for prediction
        current_step_df = df_future.iloc[step:step+1].copy()
        
        # Combine last_known_data and current_step_df
        combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)

        # Create features
        temp_df_featured, predictor_features, _, categorical_features = self.create_features(
            df_series=combined_df,
            endogenous_features_with_target=endogenous_features,
            exogenous_features=available_exogenous,  # 使用可用的外生变量
            target_feature=target_feature,
            categorical_features=categorical_features
        )
        # The last row contains the features for the current step's prediction
        current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]

        # Feature scaling
        if scaler_features is not None:
            if self.args.encode_categorical_features:
                categorical_features = [
                    col for col in current_features_for_pred.columns 
                    if col in sorted(set(categorical_features), key=categorical_features.index)
                ]
                numeric_features = [
                    col for col in current_features_for_pred.columns 
                    if col not in categorical_features
                ]

                current_features_for_pred_scaled = current_features_for_pred.copy()
                if numeric_features:
                    current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(
                        current_features_for_pred_scaled[numeric_features]
                    )

                for col in categorical_features:
                    current_features_for_pred_scaled.loc[:, col] = current_features_for_pred_scaled[col].apply(lambda x: int(x))
                current_features_for_pred_processed = current_features_for_pred_scaled
            else:
                current_features_for_pred_processed = scaler_features.transform(current_features_for_pred)
        else:
            current_features_for_pred_processed = current_features_for_pred

        # 2. Predict for all endogenous variables at this step
        y_pred_all_step = model.predict(current_features_for_pred_processed)[0]
        
        # 3. Store the prediction for the primary target
        if target_feature in all_endogenous_original_cols:
            target_idx = all_endogenous_original_cols.index(target_feature)
            y_preds_primary_target.append(y_pred_all_step[target_idx])

        # 4. Update last_known_data with the predictions for all endogenous variables
        new_row_for_last_known = current_step_df.copy().iloc[-1:]
        for i, col in enumerate(all_endogenous_original_cols):
            new_row_for_last_known[col] = y_pred_all_step[i]

        last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
        last_known_data = last_known_data.iloc[-max_lag:]

    return np.array(y_preds_primary_target)


# ==================== 使用说明 ====================

"""
如何将这些方法整合到原始文件中:

1. 对于缺失的方法 (USMDR, MSMDR):
   - 直接复制粘贴到Model类中相应位置
   - USMDR 替换 行1669的 pass
   - MSMDR 添加在 multivariate_single_multi_step_recursive_forecast 之后

2. 对于需要修复的方法 (USMD, USMR, MSMR):
   - 在原方法中添加这行代码(在方法开始处):
     available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
   - 然后将所有使用 exogenous_features 的地方改为 available_exogenous

3. 更新 _window_test 方法 (行1196):
   在相应位置添加对 USMDR 和 MSMDR 的调用

4. 更新 forecast 方法 (行1800+):
   在相应位置添加对 USMDR 和 MSMDR 的调用
"""
