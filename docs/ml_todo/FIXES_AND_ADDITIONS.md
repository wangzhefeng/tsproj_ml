# 时间序列预测框架修复说明

## 问题1: KeyError - 'date_type' not in index

### 根本原因
在预测时，代码尝试使用配置中定义的所有外生变量(如`date_type`)，但这些变量可能不存在于`df_future`中。

### 修复方案
在所有预测方法中，添加外生变量可用性检查。需要修改以下方法：

1. **univariate_single_multi_step_directly_forecast** (行 1536-1594)
2. **univariate_single_multi_step_recursive_forecast** (行 1596-1666)  
3. **multivariate_single_multi_step_directly_forecast** (行 1674-1703)
4. **multivariate_single_multi_step_recursive_forecast** (行 1705-1800+)

### 具体修复代码

在每个预测方法的开始部分，添加：

```python
# 确定实际可用的外生变量
available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
```

然后在使用外生变量时，使用`available_exogenous`代替`exogenous_features`。

---

## 问题2: 缺失的预测方法实现

需要补充以下4个方法的完整实现：

### 1. univariate_single_multi_step_directly_recursive_forecast (USMDR)

```python
def univariate_single_multi_step_directly_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, scaler_features):
    """
    单变量多步直接递归预测
    结合直接和递归策略的混合方法
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
                    categorical_features_updated = [col for col in current_features_for_pred.columns if col in sorted(set(categorical_features_updated), key=categorical_features_updated.index)]
                    numeric_features = [col for col in current_features_for_pred.columns if col not in categorical_features_updated]
                    
                    current_features_for_pred_scaled = current_features_for_pred.copy()
                    if numeric_features:
                        current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(current_features_for_pred_scaled[numeric_features])
                    
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
```

### 2. multivariate_single_multi_step_directly_forecast (MSMD)

已存在但需要修复外生变量问题：

```python
def multivariate_single_multi_step_directly_forecast(self, model, X_test_input, categorical_features, scaler_features=None):
    """
    模型预测 - 多变量多步直接预测
    """
    if not X_test_input.empty:
        # Scale features
        if scaler_features is not None:
            if self.args.encode_categorical_features:
                categorical_features = [col for col in X_test_input.columns if col in sorted(set(categorical_features), key=categorical_features.index)]
                numeric_features = [col for col in X_test_input.columns if col not in categorical_features]
                
                X_test_input_scaled = X_test_input.copy()
                if numeric_features:
                    X_test_input_scaled.loc[:, numeric_features] = scaler_features.transform(X_test_input_scaled[numeric_features])
                
                for col in categorical_features:
                    X_test_input_scaled.loc[:, col] = X_test_input_scaled[col].apply(lambda x: int(x))
                X_test_input_processed = X_test_input_scaled
            else:
                X_test_input_processed = scaler_features.transform(X_test_input)
        else:
            X_test_input_processed = X_test_input
        
        Y_pred_multi_step = model.predict(X_test_input_processed)
        return Y_pred_multi_step[:, 0]
    else:
        return np.array([])
```

### 3. multivariate_single_multi_step_directly_recursive_forecast (MSMDR)

```python
def multivariate_single_multi_step_directly_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, target_output_features, categorical_features, scaler_features):
    """
    多变量多步直接递归预测
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
                    categorical_features_updated = [col for col in current_features_for_pred.columns if col in sorted(set(categorical_features_updated), key=categorical_features_updated.index)]
                    numeric_features = [col for col in current_features_for_pred.columns if col not in categorical_features_updated]
                    
                    current_features_for_pred_scaled = current_features_for_pred.copy()
                    if numeric_features:
                        current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(current_features_for_pred_scaled[numeric_features])
                    
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
```

---

## 实施步骤

1. **备份原文件**
   ```bash
   cp exp_forecasting_ml_v2.py exp_forecasting_ml_v2.backup.py
   ```

2. **修改现有方法** 
   在以下方法开始处添加外生变量检查：
   - `univariate_single_multi_step_directly_forecast` (行 1536)
   - `univariate_single_multi_step_recursive_forecast` (行 1596)
   - `multivariate_single_multi_step_recursive_forecast` (行 1705)

3. **替换占位方法**
   - 将 `univariate_single_multi_step_directly_recursive_forecast` (行 1669) 替换为完整实现

4. **添加新方法**
   - 在 `multivariate_single_multi_step_recursive_forecast` 后面添加 `multivariate_single_multi_step_directly_recursive_forecast`

5. **更新forecast方法调用**
   确保forecast方法中正确调用新增的方法 (行 1900+)

---

## 测试建议

1. 先测试单变量方法 (USMDO, USMD, USMR, USMDR)
2. 再测试多变量方法 (MSMD, MSMR, MSMDR)  
3. 使用不同的外生变量配置进行测试
4. 验证滑窗测试功能正常工作

---

## 关键注意事项

1. **第一种方法 (USMDO)** 的逻辑:
   - 只使用外生变量(datetype, datetime, weather)
   - 不使用目标变量的滞后特征
   - 一次性直接输出所有未来步

2. **第二种方法 (USMD)** 的逻辑:
   - 使用目标变量的滞后特征 + 外生变量
   - 为每个预测步训练独立模型(或使用MultiOutputRegressor)
   - 直接预测未来多步

3. **保留原有滑窗测试逻辑**:
   - 保持`_evaluate_split_index`方法
   - 保持`_evaluate_split`方法
   - 保持`_evaluate_score`方法
   - 保持`_evaluate_result`方法
   - 保持`test`方法的完整逻辑

4. **外生变量处理**:
   - 始终检查外生变量在目标DataFrame中的可用性
   - 只使用实际存在的外生变量
   - 避免KeyError

## 版本信息
- 原始版本: 1.0.121116
- 修复版本: 2.0.240210
- 修复日期: 2026-02-10
