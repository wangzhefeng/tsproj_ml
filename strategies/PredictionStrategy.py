# -*- coding: utf-8 -*-

# ***************************************************
# * File        : prediction_strategies.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class PredictionHelper:
    """
    预测辅助类-所有预测方法的公共逻辑
    """
    
    def __init__(self, 
                 args: Dict,
                 model: Any, 
                 df_history: pd.DataFrame, 
                 df_future: pd.DataFrame, 
                 endogenous_features: List[str], 
                 exogenous_features: List[str], 
                 target_feature: str, 
                 target_output_features: List[str], 
                 categorical_features: List[str], 
                 feature_scaler,
                 log_prefix: str):
        self.args = args
        self.model = model
        self.df_history = df_history
        self.df_future = df_future
        self.endogenous_features = endogenous_features
        self.exogenous_features = exogenous_features
        self.target_feature = target_feature
        self.target_output_features = target_output_features
        self.categorical_features = categorical_features
        self.feature_scaler = feature_scaler
        self.log_prefix = log_prefix
        logger.info(f"{self.log_prefix} Endogenous features: {self.endogenous_features}")
        logger.info(f"{self.log_prefix} Target feature: {self.target_feature}")
        logger.info(f"{self.log_prefix} Forecast horizon: {self.args.horizon}")
    
    def prepare_features(self):
        """
        统一的特征预处理
        
        所有7种预测方法共用此方法，避免重复代码
        """
        # 确定实际可用的外生变量
        self.available_exogenous = [feat for feat in self.exogenous_features if feat in self.df_future.columns]
        logger.info(f"{self.log_prefix} available_exogenous: {self.available_exogenous}")
        
        # history data length： Need enough history to form the maximum lag
        self.max_lag = max(self.args.lags) if self.args.lags else 1
        logger.info(f"{self.log_prefix} max_lag: {self.max_lag}")

        # 分块大小
        self.block_size = min(self.args.lags) if self.args.lags else 1
        logger.info(f"{self.log_prefix} block_size: {self.block_size}")
        self.num_blocks = int(np.ceil(self.args.horizon / self.block_size))
        logger.info(f"{self.log_prefix} Number of blocks: {self.num_blocks}")
        
        # 1.获取足够的历史数据以构建滞后特征
        self.df_history_for_lags = self.df_history.iloc[-self.max_lag:].copy()
        logger.info(f"{self.log_prefix} df_history_for_lags shape: {self.df_history_for_lags.shape}")
        logger.info(f"{self.log_prefix} df_history_for_lags shape: {self.df_history_for_lags.shape}")
        logger.info(f"{self.log_prefix} df_history_for_lags columns: {self.df_history_for_lags.columns.tolist()}")
    
    @staticmethod
    def build_lag_features(df, features, lags):
        """
        统一的滞后特征构建
        
        适用于单变量和多变量方法
        """
        for feat in features:
            for lag in lags:
                df[f'{feat}_lag_{lag}'] = df[feat].shift(lag)
        
        return df
    
    @staticmethod
    def recursive_predict_step(model, current_features, history, target, step):
        """
        递归预测的单步逻辑
        
        USMR和MSMR共用此方法
        """
        # 预测
        prediction = model.predict(current_features)[0]
        
        # 更新历史
        history.loc[len(history)] = prediction
        
        return prediction
    # ------------------------------
    # 单变量（目标变量滞后特征）预测单变量（目标变量）
    # ------------------------------
    def univariate_single_multi_step_direct_output_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步直接输出预测(USMDO)
        """
        logger.info(f"{self.log_prefix} univariate_single_multi_step_direct_output_forecast(USMDO)")
        
        # 多步预测值收集器
        Y_preds = []
        
        # 特征工程
        if not self.args.is_testing and self.args.is_forecasting:
            # TODO 特征工程
            df_future_featured, predictor_features, target_output_features, categorical_features_updated = self.create_features(
                df_series = self.df_future, 
                endogenous_features_with_target = self.endogenous_features,
                exogenous_features = self.exogenous_features,
                target_feature = self.target_feature,
                categorical_features = self.categorical_features,
            )
            df_future_featured = df_future_featured.dropna()
            logger.info(f"{self.log_prefix} df_future_featured: \n{df_future_featured.head()}")
            logger.info(f"{self.log_prefix} predictor_features: {predictor_features}")
            logger.info(f"{self.log_prefix} target_output_features: {target_output_features}")
            logger.info(f"{self.log_prefix} categorical_features_updated: {categorical_features_updated}")
            # 特征选择
            X_test_future = df_future_featured[predictor_features]
        else:
            X_test_future = self.df_future
        logger.info(f"{self.log_prefix} X_test_future: \n{X_test_future}")

        # 特征预处理（预测模式）
        X_test_processed = self.feature_scaler.transform(X_test_future, categorical_features_updated)
        self.feature_scaler.validate_features(X_test_processed, stage="prediction")
        
        # 模型推理
        if len(X_test_processed) > 0:
            Y_preds = self.model.predict(X_test_processed)
        
        logger.info(f"{self.log_prefix} USMDO forecast completed, predicted {len(Y_preds)} steps")

        return Y_preds

    def univariate_single_multi_step_direct_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步直接预测(USMD)
        """
        logger.info(f"{self.log_prefix} univariate_single_multi_step_direct_forecast(USMD)")

        # 多步预测值收集器
        Y_preds = []

        # 1.构建预测特征数据
        df_future_exogenous = self.df_future.iloc[0:1].copy()

        # 2.合并历史数据(用于滞后特征)和未来第一个点(用于外生变量)
        df_forecast = pd.concat([self.df_history_for_lags, df_future_exogenous], ignore_index=True)

        # 3.特征工程
        (df_forecast_featured, 
         predictor_features, 
         target_output_features, 
         categorical_features_updated) = self.create_features(
            df_series = df_forecast,
            endogenous_features_with_target = self.endogenous_features,
            exogenous_features = self.available_exogenous,
            target_feature = self.target_feature,
            categorical_features = self.categorical_features,
        )

        # 4.提取出当前预测步所需要的特征（最后一行）
        X_forecast_input = df_forecast_featured[predictor_features].iloc[-1:]

        # 5.特征预处理(预测模式)
        X_test_processed = self.feature_scaler.transform(X_forecast_input, categorical_features_updated)
        self.feature_scaler.validate_features(X_test_processed, stage="prediction")

        # 6.模型预测
        Y_pred_multi_step = self.model.predict(X_test_processed)[0]
        
        #TODO 7.Assign predictions to df_future_for_prediction
        if len(Y_pred_multi_step) >= len(self.df_future):
            Y_preds = Y_pred_multi_step[:len(self.df_future)]
        else:
            Y_preds = np.pad(Y_pred_multi_step, (0, len(self.df_future) - len(Y_pred_multi_step)), 'edge')
        
        logger.info(f"{self.log_prefix} USMD forecast completed, predicted {len(Y_preds)} steps")
        
        return np.array(Y_preds)

    def univariate_single_multi_step_recursive_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步递归预测(USMR)
        """
        logger.info(f"{self.log_prefix} univariate_single_multi_step_recursive_forecast(USMR)")
        
        # 多步预测值收集器
        Y_preds = []
        
        for step in range(self.args.horizon):
            logger.info(f"{self.log_prefix} recursive forecast step: {step}...")
            # 0.Prepare current features for prediction
            if step >= len(self.df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break
            
            # 1.构建预测特征数据
            df_future_exogenous = self.df_future.iloc[step:step+1].copy()

            # 2. 合并历史数据和当前步数据
            df_forecast = pd.concat([self.df_history_for_lags, df_future_exogenous], ignore_index=True)
            
            # 3.特征工程
            df_forecast_featured, predictor_features, target_output_features, categorical_features_updated = self.create_features(
                df_series = df_forecast,
                endogenous_features_with_target = self.endogenous_features,
                exogenous_features = self.available_exogenous,
                target_feature = self.target_feature,
                categorical_features = self.categorical_features,
            )
            
            # 4.提取出当前预测步所需要的特征（最后一行）
            X_forecast_input = df_forecast_featured[predictor_features].iloc[-1:]
            
            # 5.特征预处理（预测模式）
            X_forecast_processed = self.feature_scaler.transform(X_forecast_input, categorical_features_updated)
            self.feature_scaler.validate_features(X_forecast_processed, stage="prediction")
            
            # 6.模型预测
            y_pred_step = self.model.predict(X_forecast_processed)[0]
            Y_preds.append(y_pred_step)

            # 7.将预测值更新回 df_future_exogenous，以便为下一步预测提供滞后特征
            df_future_exogenous_new_row = df_future_exogenous.copy().iloc[-1:]
            df_future_exogenous_new_row[self.target_feature] = y_pred_step

            # 8.将新行添加到历史数据中，进行下一次循环
            self.df_history_for_lags = pd.concat([self.df_history_for_lags, df_future_exogenous_new_row], ignore_index=True)
            self.df_history_for_lags = self.df_history_for_lags.iloc[-self.max_lag:]
        logger.info(f"{self.log_prefix} USMR forecast completed, predicted {len(Y_preds)} steps")

        return np.array(Y_preds)

    def univariate_single_multi_step_direct_recursive_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步直接+递归预测(USMDR)
        
        - 核心思想：
            1. 将预测 horizon 分成多个块（block_size = min(lags)）
            2. 在每个块内进行递归预测
            3. 块与块之间也是递归的（使用前一块的预测值）
        - 与其他方法的区别
            - 与 USMD 的区别：
                - USMD: 完全直接，为每步训练独立模型
                - USMDR: 只训练一个模型，但采用分块策略
            - 与 USMR 的区别：
                - USMR: 完全递归，每步都用上一步的预测
                - USMDR: 分块递归，块内递归，减少误差累积
        - 特征构成：
            - 内生变量：目标变量的滞后特征
            - 外生变量：日期时间特征+节假日类型特征+气象特征
        
        Returns:
            预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} univariate_single_multi_step_direct_recursive_forecast(USMDR)")
        
        # 多步预测值收集器
        Y_preds = []
        
        # 分块递归预测
        for block_idx in range(self.num_blocks):
            block_start = block_idx * self.block_size
            block_end = min(block_start + self.block_size, self.args.horizon)
            logger.info(f"{self.log_prefix} Processing block {block_idx+1}/{self.num_blocks}: steps {block_start} to {block_end-1}")
            
            # 在当前块内进行递归预测
            for step in range(block_start, block_end):
                if step >= len(self.df_future):
                    logger.warning(f"{self.log_prefix} Exhausted df_future at step {step}. Stopping.")
                    break
                
                # 1. 获取当前步的外生变量
                df_future_exogenous = self.df_future.iloc[step:step+1].copy()
                
                # 2. 合并历史数据和当前步数据
                df_forecast = pd.concat([self.df_history_for_lags, df_future_exogenous], ignore_index=True)
                
                # 3. 创建特征（只为目标变量创建滞后特征）
                df_forecast_featured, predictor_features, target_output_features, categorical_features_updated = self.create_features(
                    df_series = df_forecast,
                    endogenous_features_with_target = [self.target_feature],
                    exogenous_features = self.available_exogenous,
                    target_feature = self.target_feature,
                    categorical_features = self.categorical_features,
                )
                
                # 4. 提取预测特征（最后一行）
                X_forecast_input = df_forecast_featured[predictor_features].iloc[-1:]
                
                # 5. 特征缩放
                X_forecast_processed = self.feature_scaler.transform(X_forecast_input, categorical_features_updated)
                self.feature_scaler.validate_features(X_forecast_processed, stage="prediction")
                
                # 6. 预测
                y_pred_step = self.model.predict(X_forecast_processed)[0]
                Y_preds.append(y_pred_step)
                
                # 7. 更新历史数据（用于下一步预测）
                df_future_exogenous_new_row = df_future_exogenous.copy().iloc[-1:]
                df_future_exogenous_new_row[self.target_feature] = y_pred_step
                
                # 8.将新行添加到历史数据中，进行下一次循环
                self.df_history_for_lags = pd.concat([self.df_history_for_lags, df_future_exogenous_new_row], ignore_index=True)
                self.df_history_for_lags = self.df_history_for_lags.iloc[-self.max_lag:]  # 只保留需要的历史长度
        logger.info(f"{self.log_prefix} USMDR forecast completed, predicted {len(Y_preds)} steps")
        
        return np.array(Y_preds)
    # ------------------------------
    # 多变量（除目标变量外的内生变量）预测单变量（目标变量）
    # ------------------------------
    def multivariate_single_multi_step_direct_forecast(self):
        """
        多变量(内生变量)预测单变量(目标变量)多步直接预测(MSMD)
        - 方法特点：
            1. 特征：所有内生变量(target + 其他内生变量)的滞后 + 外生变量
            2. 训练：为每个未来步 H 创建目标列 target_shift_0, target_shift_1, ..., target_shift_H-1
            3. 预测：一次性输出所有 H 步的预测值
        - 与 USMD 的区别：
            - USMD: 只使用目标变量的滞后特征
            - MSMD: 使用所有内生变量的滞后特征（更多信息）
         
        Returns:
            预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} multivariate_single_multi_step_direct_forecast(MSMD)")
        
        # 多步预测值收集器
        Y_preds = []
        
        # 0.确保所有内生变量都在历史数据中
        for endo_feat in self.endogenous_features:
            if endo_feat not in self.df_history_for_lags.columns and endo_feat in self.df_history.columns:
                self.df_history_for_lags[endo_feat] = self.df_history[endo_feat].iloc[-self.max_lag:] 
        
        # 1.构建预测特征数据
        df_future_exogenous = self.df_future.iloc[0:1].copy()
        
        # 2.合并历史数据(用于滞后特征)和未来第一个点(用于外生变量)
        df_forecast = pd.concat([self.df_history_for_lags, df_future_exogenous], ignore_index=True)
        
        # 3.创建特征
        (df_forecast_featured, 
         predictor_features, 
         target_output_features, 
         categorical_features_updated) = self.create_features(
            df_series = df_forecast,
            endogenous_features_with_target = self.endogenous_features,
            exogenous_features = self.available_exogenous,
            target_feature = self.target_feature,
            categorical_features = self.categorical_features
        )
        
        # 4.提取出当前预测步所需要的特征（最后一行）
        X_forecast_input = df_forecast_featured[predictor_features].iloc[-1:]
        
        # 5.特征预处理(预测模式)
        X_test_processed = self.feature_scaler.transform(X_forecast_input, categorical_features_updated)
        self.feature_scaler.validate_features(X_test_processed, stage="prediction")
        
        # 6.模型预测
        Y_pred_multi_step = self.model.predict(X_test_processed)[0]
        
        # 7.处理预测结果
        if len(Y_pred_multi_step) >= len(self.df_future):
            Y_preds = Y_pred_multi_step[:len(self.df_future)]
        else:
            Y_preds = np.pad(Y_pred_multi_step, (0, len(self.df_future) - len(Y_pred_multi_step)), 'edge')
        
        logger.info(f"{self.log_prefix} MSMD forecast completed, predicted {len(Y_preds)} steps")

        return np.array(Y_preds)

    def multivariate_single_multi_step_recursive_forecast(self):
        """
        多变量(内生变量)预测单变量(目标变量)多步递归预测(MSMR)
        - 方法特点：
            1. 特征：所有内生变量(target + 其他内生变量)的滞后 + 外生变量
            2. 训练：TODO
            3. 预测：TODO
        - 与 USMD 的区别：
            - USMR: 只使用目标变量的滞后特征
            - MSMR: 使用所有内生变量的滞后特征（更多信息）
        
        Returns:
            目标变量的预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} multivariate_single_multi_step_recursive_forecast(MSMR)")

        # 多步预测值收集器
        Y_preds = []
        
        # Ensure all original endogenous_features are present in last_known_data for lag creation
        all_endogenous_original_cols = [col.replace("_shift_1", "") for col in self.target_output_features]
        for col in all_endogenous_original_cols:
            if col not in self.df_history_for_lags.columns and col in self.df_history.columns:
                self.df_history_for_lags[col] = self.df_history[col].iloc[-self.max_lag:]
        
        # Iterate for each step in the forecast horizon
        for step in range(self.args.horizon):
            logger.info(f"{self.log_prefix} multivariate-recursive forecast step: {step}...")
            # 0.Prepare current features for prediction
            if step >= len(self.df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break
            
            # 1. Prepare current features for prediction
            df_future_exogenous = self.df_future.iloc[step:step+1].copy()
            df_forecast = pd.concat([self.df_history_for_lags, df_future_exogenous], ignore_index=True)

            # 2.特征工程
            df_forecast_featured, predictor_features, target_output_features, categorical_features = self.create_features(
                df_series = df_forecast,
                endogenous_features_with_target = self.endogenous_features,
                exogenous_features = self.available_exogenous,
                target_feature = self.target_feature,
                categorical_features = self.categorical_features,
            )

            # 3.提取出当前预测步所需要的特征（最后一行）
            X_forecast_input = df_forecast_featured[predictor_features].iloc[-1:]

            # 4.特征预处理（预测模式）
            X_forecast_processed = self.feature_scaler.transform(X_forecast_input, categorical_features)
            self.feature_scaler.validate_features(X_forecast_processed, stage="prediction")

            # 5.模型预测
            y_pred_step = self.model.predict(X_forecast_processed)[0]
            # Map predictions back to their shifted column names
            next_pred_dict = dict(zip(self.target_output_features, y_pred_step))

            # 6.Store the prediction for the primary target (assuming it's the first in target_output_features)
            if self.target_feature:
                primary_target_shifted_name = f"{self.target_feature}_shift_1"
                if primary_target_shifted_name in next_pred_dict:
                    Y_preds.append(next_pred_dict[primary_target_shifted_name])
                else:
                    Y_preds.append(y_pred_step[0])
            else:
                Y_preds.append(y_pred_step[0])

            # 6.将预测值更新回 df_future_exogenous，以便为下一步预测提供滞后特征
            df_future_exogenous_new_row = df_future_exogenous.copy().iloc[-1:]
            
            # 7.Update endogenous variables (target and other endogenous) with predicted values
            for shifted_col_name, pred_val in next_pred_dict.items():
                original_col_name = shifted_col_name.replace("_shift_1", "")
                df_future_exogenous_new_row[original_col_name] = pred_val

            # 8.todo
            for col in self.df_history_for_lags.columns:
                if col not in df_future_exogenous_new_row.columns:
                    # If it's an exogenous feature in current_step_df, prefer that
                    if col in df_future_exogenous.columns:
                        df_future_exogenous_new_row[col] = df_future_exogenous[col].iloc[-1]
                    else: # Otherwise, take from the last known data point
                        df_future_exogenous_new_row[col] = self.df_history_for_lags[col].iloc[-1]

            # 9.将新行添加到历史数据中，进行下一次循环
            self.df_history_for_lags = pd.concat([self.df_history_for_lags, df_future_exogenous_new_row], ignore_index=True)
            self.df_history_for_lags = self.df_history_for_lags.iloc[-self.max_lag:]

        logger.info(f"{self.log_prefix} MSMR forecast completed, predicted {len(Y_preds)} steps")

        return np.array(Y_preds)

    def multivariate_single_multi_step_direct_recursive_forecast(self):
        """
        多变量(内生变量)预测单变量(目标变量)多步直接+递归预测(MSMDR)
        
        - 核心思想：
            1. 使用所有内生变量的滞后特征（不只是目标变量）
            2. 分块递归预测目标变量
            3. 对于其他内生变量，使用持久性预测或简单方法估计
        - 与其他方法的区别：
            - 与 USMDR 的核心区别：
                - USMDR: 只用目标变量的滞后 → 特征少
                - MSMDR: 用所有内生变量的滞后 → 特征多，信息丰富
            - 与 MSMR 的区别：
                - MSMR: 递归预测所有内生变量
                - MSMDR: 只递归预测目标变量，其他内生变量用简化方法
        - 特征构成示例：
            假设 endogenous_features = ['load', 'temperature', 'humidity']
                target_feature = 'load'
                lags = [1, 2, 7]
            
            特征 = [load_lag_1, load_lag_2, load_lag_7,           # 目标变量的滞后
                temperature_lag_1, temperature_lag_2, temperature_lag_7,  # 其他内生变量的滞后
                humidity_lag_1, humidity_lag_2, humidity_lag_7,
                hour, day_of_week, ...]  # 外生变量
        
        Returns:
            目标变量的预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} multivariate_single_multi_step_direct_recursive_forecast(MSMDR)")

        # 多步预测值收集器
        Y_preds = []
        
        # 确保所有内生变量都在历史数据中
        for endo_feat in self.endogenous_features:
            if endo_feat not in self.df_history_for_lags.columns and endo_feat in self.df_history.columns:
                self.df_history_for_lags[endo_feat] = self.df_history[endo_feat].iloc[-self.max_lag:]
        
        # 为其他内生变量准备持久性预测（假设其他内生变量在未来保持最后观测值不变，或使用简单趋势）
        other_endogenous = [feat for feat in self.endogenous_features if feat != self.target_feature]
        last_values_other_endogenous = {}
        for feat in other_endogenous:
            if feat in self.df_history_for_lags.columns:
                last_values_other_endogenous[feat] = self.df_history_for_lags[feat].iloc[-1]
            else:
                last_values_other_endogenous[feat] = 0
                logger.warning(f"{self.log_prefix} Feature {feat} not found, using 0")
        
        # 分块递归预测
        for block_idx in range(self.num_blocks):
            # block index
            block_start = block_idx * self.block_size
            block_end = min(block_start + self.block_size, self.args.horizon)
            logger.info(f"{self.log_prefix} Processing block {block_idx+1}/{self.num_blocks}: steps {block_start}~{block_end-1}")
            
            # 在当前块内进行递归预测
            for step in range(block_start, block_end):
                logger.info(f"{self.log_prefix} recursive forecast step: {step}...")
                if step >= len(self.df_future):
                    logger.warning(f"{self.log_prefix} Exhausted df_future at step {step}. Stopping.")
                    break
                
                # 1.获取当前步的外生变量
                df_future_exogenous = self.df_future.iloc[step:step+1].copy()
                
                # 2. 合并历史数据和当前步数据
                df_forecast = pd.concat([self.df_history_for_lags, df_future_exogenous], ignore_index=True)
                
                # 3. 创建特征（为所有内生变量创建滞后特征）
                (df_forecast_featured, 
                 predictor_features, 
                 target_output_features, 
                 categorical_features_updated) = self.create_features(
                    df_series = df_forecast,
                    endogenous_features_with_target = self.endogenous_features,  # 所有内生变量！
                    exogenous_features = self.available_exogenous,
                    target_feature=self.target_feature,
                    categorical_features=self.categorical_features,
                )
                
                # 4. 提取预测特征（最后一行）
                X_forecast_input = df_forecast_featured[predictor_features].iloc[-1:]
                
                # 5. 特征缩放
                X_forecast_processed = self.feature_scaler.transform(X_forecast_input, self.categorical_features)
                self.feature_scaler.validate_features(X_forecast_processed, stage="prediction")
                
                # 6. 预测目标变量
                y_pred_target = self.model.predict(X_forecast_processed)[0]
                Y_preds.append(y_pred_target)
                
                # 7. 更新历史数据
                df_forecast_exogenous_new_row = X_forecast_input.copy().iloc[-1:]
                df_forecast_exogenous_new_row[self.target_feature] = y_pred_target
                
                # 8.更新其他内生变量的值（使用持久性预测）
                # 8.1 策略 1: 保持最后观测值不变（最简单）
                for feat in other_endogenous:
                    df_forecast_exogenous_new_row[feat] = last_values_other_endogenous[feat]
                # 8.2 策略 2: 也可以使用简单的移动平均或趋势（更复杂但可能更准确）
                # for feat in other_endogenous:
                #     # 计算最近几个值的平均作为预测
                #     if feat in self.df_history_for_lags.columns:
                #         recent_mean = self.df_history_for_lags[feat].tail(3).mean()
                #         df_forecast_exogenous_new_row[feat] = recent_mean
                
                # 9.将新行添加到历史数据中，进行下一次循环
                self.df_history_for_lags = pd.concat([self.df_history_for_lags, df_forecast_exogenous_new_row], ignore_index=True)
                self.df_history_for_lags = self.df_history_for_lags.iloc[-self.max_lag:]
        logger.info(f"{self.log_prefix} MSMDR forecast completed, predicted {len(Y_preds)} steps")
        
        return np.array(Y_preds)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
