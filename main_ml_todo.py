# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main_ml_todo.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-13
# * Version     : 1.0.021310
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class Model:
    
    '''
    # ------------------------------
    # 单变量（目标变量滞后特征）预测单变量（目标变量）
    # ------------------------------
    def univariate_single_multi_step_direct_output_forecast(self, model, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, feature_scaler):
        """
        单变量多步直接输出预测
        [datetype, datetime, weather] -> [Y]
        """
        if not self.args.is_testing and self.args.is_forecasting:
            # 特征工程
            df_future_featured, predictor_features, target_output_features, categorical_features = self.create_features(
                df_series=df_future, 
                endogenous_features_with_target=endogenous_features,
                exogenous_features=exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features
            )
            df_future_featured = df_future_featured.dropna()
            logger.info(f"{self.log_prefix} df_future_featured: \n{df_future_featured.head()}")
            logger.info(f"{self.log_prefix} predictor_features: {predictor_features}")
            logger.info(f"{self.log_prefix} target_output_features: {target_output_features}")
            logger.info(f"{self.log_prefix} categorical_features: {categorical_features}")
            # 特征选择
            X_test_future = df_future_featured[predictor_features]
            X_test_future_df = X_test_future.copy()
            logger.info(f"{self.log_prefix} X_test_future_df: \n{X_test_future_df}")
        else:
            X_test_future_df = df_future
        
        # 特征预处理（预测模式）
        X_test_processed = feature_scaler.transform(X_test_future_df, categorical_features)
        feature_scaler.validate_features(X_test_processed, stage="prediction")
        # 模型推理
        if len(X_test_future_df) > 0:
            Y_pred = model.predict(X_test_processed)
        else:
            Y_pred = []
            logger.info(f"{self.log_prefix} X_future length is 0!")
        
        return Y_pred

    def univariate_single_multi_step_direct_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, feature_scaler):
        """
        单变量多步直接预测(USMD): 使用目标变量滞后 + 外生变量预测未来多步
        [Y, datetime, weather, datetype]
        """
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        # Need enough history to form the maximum lag
        max_lag = max(self.args.lags) if self.args.lags else 1
        # Combine last relevant history for lags and the first future point for exogenous/datetime
        relevant_history_for_lags = df_history.iloc[-max_lag:].copy()
        # Use the first point of df_future_for_prediction for exogenous features
        forecast_start_point_exogenous = df_future.iloc[0:1].copy()
        # For lag generation, combined_for_features needs a 'time' column for features
        combined_for_features = pd.concat([relevant_history_for_lags, forecast_start_point_exogenous], ignore_index=True)
        # Generate features for this combined data, then take the last row as input for prediction
        combined_featured, predictor_features_for_forecast, _, categorical_features = self.create_features(
            df_series = combined_for_features,
            endogenous_features_with_target = endogenous_features,
            exogenous_features = available_exogenous,
            target_feature = target_feature,
            categorical_features = categorical_features,
        )
        # The actual features for prediction are the last row (corresponding to the moment before forecast starts)
        X_forecast_input = combined_featured[predictor_features_for_forecast].iloc[-1:]
        
        # 特征预处理（预测模式）
        X_forecast_input_processed = feature_scaler.transform(X_forecast_input, categorical_features)
        feature_scaler.validate_features(X_forecast_input_processed, stage="prediction")
        
        # Expects (1, H) output -> [H] array
        Y_pred_multi_step = model.predict(X_forecast_input_processed)[0]
        
        # Assign predictions to df_future_for_prediction
        if len(Y_pred_multi_step) >= len(df_future):
            Y_pred = Y_pred_multi_step[:len(df_future)]
        else:
            logger.warning(f"Predicted {len(Y_pred_multi_step)} steps but df_future requires {len(df_future)} rows. Padding predictions.")
            Y_pred = np.pad(Y_pred_multi_step, (0, len(df_future) - len(Y_pred_multi_step)), 'edge')
        
        return Y_pred

    def univariate_single_multi_step_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, feature_scaler):
        """
        单变量递归多步预测(USMR)
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
            # 2.将历史数据和当前步的数据合并，以便创建特征
            combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
            # 3.为合并后的数据创建特征
            temp_df_featured, predictor_features, _, categorical_features = self.create_features(
                df_series=combined_df,
                endogenous_features_with_target=endogenous_features,
                exogenous_features=available_exogenous,
                target_feature=target_feature,
                categorical_features=categorical_features,
            )
            # 4.提取出当前预测步所需要的特征（最后一行）
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
            # 5.特征预处理（预测模式）
            current_features_for_pred_processed = feature_scaler.transform(current_features_for_pred, categorical_features)
            feature_scaler.validate_features(current_features_for_pred_processed, stage="prediction")
            # 5.进行预测
            y_pred_step = model.predict(current_features_for_pred_processed)[0]
            y_preds.append(y_pred_step)
            # 6.将预测值更新回 df_history，以便为下一步预测提供滞后特征
            new_row_for_last_known = current_step_df.copy().iloc[-1:]
            new_row_for_last_known[target_feature] = y_pred_step # Add the prediction as the new 'actual' for lag calculation
            # 将新行添加到历史数据中，进行下一次循环
            last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
            last_known_data = last_known_data.iloc[-max_lag:]

        return np.array(y_preds)

    def univariate_single_multi_step_direct_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, feature_scaler):
        """
        单变量多步直接递归预测 (USMDR)
    
        核心思想：
        1. 将预测horizon分成多个块（block_size = min(lags)）
        2. 在每个块内进行递归预测
        3. 块与块之间也是递归的（使用前一块的预测值）
        
        与 USMR 的区别：
        - USMR: 完全递归，每步都用上一步的预测
        - USMDR: 分块递归，块内递归，减少误差累积
        
        与 USMD 的区别：
        - USMD: 完全直接，为每步训练独立模型
        - USMDR: 只训练一个模型，但采用分块策略
        
        特征构成：
        - 只使用目标变量的滞后特征
        - 加上外生变量
        
        Args:
            model: 训练好的单输出模型
            df_history: 历史数据
            df_future: 未来数据(包含外生变量)
            endogenous_features: 内生变量列表(但只使用目标变量)
            exogenous_features: 外生变量列表
            target_feature: 目标变量名
            categorical_features: 类别特征列表
            scaler_features: 归一化器
        
        Returns:
            预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} univariate_single_multi_step_directly_recursive_forecast (USMDR)")
        logger.info(f"{self.log_prefix} Target feature: {target_feature}")
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        logger.info(f"{self.log_prefix} Available exogenous features: {available_exogenous}")
        
        # 初始化
        max_lag = max(self.args.lags) if self.args.lags else 1
        block_size = min(self.args.lags) if self.args.lags else 1  # 分块大小
        
        logger.info(f"{self.log_prefix} Max lag: {max_lag}, Block size: {block_size}")
        logger.info(f"{self.log_prefix} Forecast horizon: {self.horizon}")
        
        # 预测值收集器
        y_preds = []
        
        # 获取历史数据
        last_known_data = df_history.iloc[-max_lag:].copy()
        
        # 确保目标特征在历史数据中
        if target_feature not in last_known_data.columns and target_feature in df_history.columns:
            last_known_data[target_feature] = df_history[target_feature].iloc[-max_lag:]
        
        logger.info(f"{self.log_prefix} Last known data shape: {last_known_data.shape}")
        
        # 分块递归预测
        num_blocks = int(np.ceil(self.horizon / block_size))
        logger.info(f"{self.log_prefix} Number of blocks: {num_blocks}")
        
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, self.horizon)
            
            logger.info(f"{self.log_prefix} Processing block {block_idx + 1}/{num_blocks}: steps {block_start} to {block_end-1}")
            
            # 在当前块内进行递归预测
            for step in range(block_start, block_end):
                if step >= len(df_future):
                    logger.warning(f"{self.log_prefix} Exhausted df_future at step {step}. Stopping.")
                    break
                
                # 1. 获取当前步的外生变量
                current_step_df = df_future.iloc[step:step+1].copy()
                
                # 2. 合并历史数据和当前步数据
                combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
                
                # 3. 创建特征（只为目标变量创建滞后特征）
                temp_df_featured, predictor_features, _, categorical_features_updated = self.create_features(
                    df_series=combined_df,
                    endogenous_features_with_target=[target_feature],  # 只使用目标变量！
                    exogenous_features=available_exogenous,
                    target_feature=target_feature,
                    categorical_features=categorical_features,
                )
                
                # 4. 提取预测特征（最后一行）
                current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
                
                # 5. 特征缩放
                current_features_for_pred_processed = feature_scaler.transform(current_features_for_pred, categorical_features)
                feature_scaler.validate_features(current_features_for_pred_processed, stage="prediction")
                
                # 6. 预测
                y_pred_step = model.predict(current_features_for_pred_processed)[0]
                y_preds.append(y_pred_step)
                
                logger.info(f"{self.log_prefix}   Step {step}: predicted {y_pred_step:.4f}")
                
                # 7. 更新历史数据（用于下一步预测）
                new_row_for_last_known = current_step_df.copy().iloc[-1:]
                new_row_for_last_known[target_feature] = y_pred_step
                
                last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
                last_known_data = last_known_data.iloc[-max_lag:]  # 只保留需要的历史长度
        
        logger.info(f"{self.log_prefix} USMDR forecast completed, predicted {len(y_preds)} steps")
        
        return np.array(y_preds)
    # ------------------------------
    # 多变量（除目标变量外的内生变量）预测单变量（目标变量）
    # ------------------------------
    def multivariate_single_multi_step_direct_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, feature_scaler):
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
        logger.info(f"{self.log_prefix} multivariate_single_multi_step_direct_forecast")
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
        X_forecast_input_processed = feature_scaler.transform(X_forecast_input, categorical_features)
        feature_scaler.validate_features(X_forecast_input_processed, stage="prediction")
        
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

    def multivariate_single_multi_step_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, target_output_features, categorical_features, feature_scaler):
        """
        多变量多步递归预测 (predicts all endogenous variables recursively)
        """
        # target_output_features contains names like "target_feature_shift_1", "endogenous1_shift_1".
        # model predicts target_t+1, endogenous1_t+1, ...
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        y_preds_primary_target = []
        
        # Start with the latest available data to construct initial features
        # `last_known_data` will hold values (actuals or predictions) for all endogenous variables needed for lags.
        all_endogenous_original_cols = [col.replace("_shift_1", "") for col in target_output_features] # original names of predicted endogenous
        
        max_lag = max(self.args.lags) if self.args.lags else 1
        last_known_data = df_history.iloc[-max_lag:].copy()
        
        # Ensure all original endogenous_features are present in last_known_data for lag creation
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
            current_step_df = df_future.iloc[step:step+1].copy() # Future exogenous and datetime features for this step
            
            # Combine last_known_data (with actuals or previous predictions for lags) and current_step_df
            combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)

            # Create features based on the combined data (lags will be based on last_known_data)
            temp_df_featured, predictor_features, _, categorical_features = self.create_features(
                df_series=combined_df,
                endogenous_features_with_target=endogenous_features, # All actual endogenous cols including target
                exogenous_features=available_exogenous,
                target_feature=target_feature,
                categorical_features=categorical_features,
            )
            # The last row of temp_df_featured contains the features for the current step's prediction
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]

            # 特征预处理（预测模式）
            current_features_for_pred_processed = feature_scaler.transform(current_features_for_pred, categorical_features)
            feature_scaler.validate_features(current_features_for_pred_processed, stage="prediction")
            
            # 2. Make prediction for the next step for all target_output_features
            next_pred_values_array = model.predict(current_features_for_pred_processed)[0] # [0] because predict returns [[val1, val2, ...]]
            # Map predictions back to their shifted column names
            next_pred_dict = dict(zip(target_output_features, next_pred_values_array))

            # Store the prediction for the primary target (assuming it's the first in target_output_features)
            if target_feature:
                primary_target_shifted_name = f"{target_feature}_shift_1"
                if primary_target_shifted_name in next_pred_dict:
                    y_preds_primary_target.append(next_pred_dict[primary_target_shifted_name])
                else: # Fallback if primary target not in output cols
                    y_preds_primary_target.append(next_pred_values_array[0]) # Default to first predicted output
            else:
                y_preds_primary_target.append(next_pred_values_array[0]) # No target_feature, take first predicted

            # 3. Update `last_known_data` for the next recursive step
            new_row_for_last_known = current_step_df.copy().iloc[-1:]
            
            # Update endogenous variables (target and other endogenous) with predicted values
            for shifted_col_name, pred_val in next_pred_dict.items():
                original_col_name = shifted_col_name.replace("_shift_1", "")
                new_row_for_last_known[original_col_name] = pred_val
            
            # Carry over any other features from previous last_known_data that are needed but not predicted
            # This is crucial for exogenous features that remain constant or are known in future
            # And also other endogenous that are not predicted but are part of the lag features
            for col in last_known_data.columns:
                if col not in new_row_for_last_known.columns:
                    # If it's an exogenous feature in current_step_df, prefer that
                    if col in current_step_df.columns:
                        new_row_for_last_known[col] = current_step_df[col].iloc[-1]
                    else: # Otherwise, take from the last known data point
                        new_row_for_last_known[col] = last_known_data[col].iloc[-1]

            last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
            last_known_data = last_known_data.iloc[-max_lag:]

        return np.array(y_preds_primary_target)

    def multivariate_single_multi_step_direct_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, feature_scaler):
        """
        多变量多步直接递归预测 (MSMDR)
        
        核心思想：
        1. 使用所有内生变量的滞后特征（不只是目标变量）
        2. 分块递归预测目标变量
        3. 对于其他内生变量，使用持久性预测或简单方法估计
        
        与 USMDR 的核心区别：
        - USMDR: 只用目标变量的滞后 → 特征少
        - MSMDR: 用所有内生变量的滞后 → 特征多，信息丰富
        
        与 MSMR 的区别：
        - MSMR: 递归预测所有内生变量
        - MSMDR: 只递归预测目标变量，其他内生变量用简化方法
        
        特征构成示例：
        假设 endogenous_features = ['load', 'temperature', 'humidity']
            target_feature = 'load'
            lags = [1, 2, 7]
        
        特征 = [load_lag_1, load_lag_2, load_lag_7,           # 目标变量的滞后
            temperature_lag_1, temperature_lag_2, temperature_lag_7,  # 其他内生变量的滞后
            humidity_lag_1, humidity_lag_2, humidity_lag_7,
            hour, day_of_week, ...]  # 外生变量
        
        Args:
            model: 训练好的单输出模型
            df_history: 历史数据（包含所有内生变量）
            df_future: 未来数据（包含外生变量）
            endogenous_features: 所有内生变量列表（包括目标变量）
            exogenous_features: 外生变量列表
            target_feature: 目标变量名
            categorical_features: 类别特征列表
            scaler_features: 归一化器
        
        Returns:
            目标变量的预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} multivariate_single_multi_step_directly_recursive_forecast (MSMDR)")
        logger.info(f"{self.log_prefix} Endogenous features: {endogenous_features}")
        logger.info(f"{self.log_prefix} Target feature: {target_feature}")
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        logger.info(f"{self.log_prefix} Available exogenous features: {available_exogenous}")
        
        # 初始化
        max_lag = max(self.args.lags) if self.args.lags else 1
        block_size = min(self.args.lags) if self.args.lags else 1
        
        logger.info(f"{self.log_prefix} Max lag: {max_lag}, Block size: {block_size}")
        logger.info(f"{self.log_prefix} Forecast horizon: {self.horizon}")
        
        # 预测值收集器
        y_preds_target = []
        
        # 获取历史数据
        last_known_data = df_history.iloc[-max_lag:].copy()
        
        # 确保所有内生变量都在历史数据中
        for endo_feat in endogenous_features:
            if endo_feat not in last_known_data.columns and endo_feat in df_history.columns:
                last_known_data[endo_feat] = df_history[endo_feat].iloc[-max_lag:]
        
        logger.info(f"{self.log_prefix} Last known data shape: {last_known_data.shape}")
        logger.info(f"{self.log_prefix} Last known data columns: {last_known_data.columns.tolist()}")
        
        # 为其他内生变量准备持久性预测
        # （假设其他内生变量在未来保持最后观测值不变，或使用简单趋势）
        other_endogenous = [feat for feat in endogenous_features if feat != target_feature]
        last_values_other_endogenous = {}
        
        for feat in other_endogenous:
            if feat in last_known_data.columns:
                last_values_other_endogenous[feat] = last_known_data[feat].iloc[-1]
            else:
                last_values_other_endogenous[feat] = 0
                logger.warning(f"{self.log_prefix} Feature {feat} not found, using 0")
        
        logger.info(f"{self.log_prefix} Last values for other endogenous: {last_values_other_endogenous}")
        
        # 分块递归预测
        num_blocks = int(np.ceil(self.horizon / block_size))
        logger.info(f"{self.log_prefix} Number of blocks: {num_blocks}")
        
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, self.horizon)
            
            logger.info(f"{self.log_prefix} Processing block {block_idx + 1}/{num_blocks}: steps {block_start} to {block_end-1}")
            
            # 在当前块内进行递归预测
            for step in range(block_start, block_end):
                if step >= len(df_future):
                    logger.warning(f"{self.log_prefix} Exhausted df_future at step {step}. Stopping.")
                    break
                
                # 1. 获取当前步的外生变量
                current_step_df = df_future.iloc[step:step+1].copy()
                
                # 2. 合并历史数据和当前步数据
                combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
                
                # 3. 创建特征（为所有内生变量创建滞后特征）
                temp_df_featured, predictor_features, _, categorical_features_updated = self.create_features(
                    df_series=combined_df,
                    endogenous_features_with_target=endogenous_features,  # 所有内生变量！
                    exogenous_features=available_exogenous,
                    target_feature=target_feature,
                    categorical_features=categorical_features,
                )
                
                # 4. 提取预测特征（最后一行）
                current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
                
                # 5. 特征缩放
                current_features_for_pred_processed = feature_scaler.transform(current_features_for_pred, categorical_features)
                feature_scaler.validate_features(current_features_for_pred_processed, stage="prediction")
                
                # 6. 预测目标变量
                y_pred_target = model.predict(current_features_for_pred_processed)[0]
                y_preds_target.append(y_pred_target)
                
                logger.info(f"{self.log_prefix}   Step {step}: predicted target = {y_pred_target:.4f}")
                
                # 7. 更新历史数据
                new_row_for_last_known = current_step_df.copy().iloc[-1:]
                
                # 更新目标变量的值（使用预测值）
                new_row_for_last_known[target_feature] = y_pred_target
                
                # 更新其他内生变量的值（使用持久性预测）
                # 策略1: 保持最后观测值不变（最简单）
                for feat in other_endogenous:
                    new_row_for_last_known[feat] = last_values_other_endogenous[feat]
                
                # 策略2: 也可以使用简单的移动平均或趋势（更复杂但可能更准确）
                # for feat in other_endogenous:
                #     # 计算最近几个值的平均作为预测
                #     if feat in last_known_data.columns:
                #         recent_mean = last_known_data[feat].tail(3).mean()
                #         new_row_for_last_known[feat] = recent_mean
                
                last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
                last_known_data = last_known_data.iloc[-max_lag:]
        
        logger.info(f"{self.log_prefix} MSMDR forecast completed, predicted {len(y_preds_target)} steps")
        
        return np.array(y_preds_target)
    '''




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
