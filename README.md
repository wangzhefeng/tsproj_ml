# 时间序列预测方法完整对比说明

## 📊 7种预测方法总览

| 方法代码 | 方法名称 | 特征来源 | 目标输出 | 预测策略 | 适用场景 |
|---------|---------|---------|---------|---------|---------|
| USMDO | 单变量多步直接输出 | 仅外生变量 | 目标变量未来H步 | 一次输出 | 外生变量主导的预测 |
| USMD | 单变量多步直接 | 目标变量滞后+外生 | 目标变量未来H步 | 一次输出 | 短期预测，避免误差累积 |
| USMR | 单变量多步递归 | 目标变量滞后+外生 | 目标变量未来1步 | 递归预测 | 长期预测 |
| USMDR | 单变量多步直接递归 | 目标变量滞后+外生 | 目标变量未来1步 | 分块递归 | 中长期预测 |
| MSMD | 多变量多步直接 | 所有内生变量滞后+外生 | 目标变量未来H步 | 一次输出 | 多变量相关性强 |
| MSMR | 多变量多步递归 | 所有内生变量滞后+外生 | 所有内生变量未来1步 | 递归预测 | 需要所有内生变量的预测 |
| MSMDR | 多变量多步直接递归 | 所有内生变量滞后+外生 | 所有内生变量未来1步 | 分块递归 | 中长期多变量预测 |

---

## 🔍 详细说明

### 1️⃣ USMDO - 单变量多步直接输出预测

**英文名**: Univariate Single-target Multi-step Direct Output forecast

**核心思想**: 只使用外生变量(时间特征、天气等)，不使用目标变量的历史信息

#### 特征构成
```python
特征 X = [hour, day_of_week, month, is_holiday, temperature, humidity, ...]
       # 只包含外生变量，不包含任何滞后特征
```

#### 目标构成
```python
目标 Y = [load_t+1, load_t+2, ..., load_t+H]
       # H个未来步的目标变量
```

#### 训练过程
```python
输入: (N, num_exogenous_features)
输出: (N, H)
模型: MultiOutputRegressor(LGBMRegressor())
```

#### 预测过程
```python
# 一次性预测所有H步
X_test = [当前时刻的外生变量]  # 形状: (1, num_exogenous_features)
Y_pred = model.predict(X_test)  # 形状: (1, H)
结果 = Y_pred[0]                # 形状: (H,)
```

#### 优点
- ✅ 不需要历史目标值，适合冷启动
- ✅ 不会累积误差
- ✅ 计算简单快速

#### 缺点
- ❌ 忽略了目标变量的自相关性
- ❌ 性能可能较差
- ❌ 只适合外生变量主导的场景

#### 适用场景
- 目标变量主要受外部因素驱动
- 没有历史数据的新场景
- 外生变量信息非常丰富

---

### 2️⃣ USMD - 单变量多步直接预测

**英文名**: Univariate Single-target Multi-step Direct forecast

**核心思想**: 使用目标变量的滞后特征 + 外生变量，为每个未来步训练独立模型

#### 特征构成
```python
特征 X = [load_lag_1, load_lag_2, load_lag_7, load_lag_14, 
         hour, day_of_week, is_holiday, temperature, ...]
       # 目标变量的滞后 + 外生变量
```

#### 目标构成
```python
目标 Y = [load_t+1, load_t+2, ..., load_t+H]
       # H个未来步的目标变量
```

#### 训练过程
```python
输入: (N, num_lags + num_exogenous)
输出: (N, H)
模型: MultiOutputRegressor(LGBMRegressor())
     # 内部为每个未来步训练一个模型
```

#### 预测过程
```python
# 构建滞后特征
X_test = [最近的历史值作为滞后特征, 当前时刻的外生变量]
Y_pred = model.predict(X_test)  # 一次性输出所有H步
```

#### 优点
- ✅ 利用目标变量的自相关性
- ✅ 不会累积误差(一次性输出)
- ✅ 各步独立，更稳定

#### 缺点
- ❌ 训练成本高(H个模型)
- ❌ 忽略了未来步之间的依赖关系
- ❌ 需要足够的历史数据

#### 适用场景
- 短期预测(H较小)
- 追求预测稳定性
- 有充足训练数据

---

### 3️⃣ USMR - 单变量多步递归预测

**英文名**: Univariate Single-target Multi-step Recursive forecast

**核心思想**: 只训练一步预测模型，递归地使用预测值作为下一步的输入

#### 特征构成
```python
特征 X = [load_lag_1, load_lag_2, load_lag_7, load_lag_14,
         hour, day_of_week, is_holiday, temperature, ...]
```

#### 目标构成
```python
目标 Y = load_t+1  # 只预测下一步
```

#### 训练过程
```python
输入: (N, num_lags + num_exogenous)
输出: (N, 1)
模型: LGBMRegressor()  # 单输出模型
```

#### 预测过程(递归)
```python
predictions = []
for step in range(H):
    # 使用最新的滞后特征(包含之前的预测值)
    X = [最近的历史值/预测值作为滞后, 当前步的外生变量]
    y_pred = model.predict(X)
    predictions.append(y_pred)
    
    # 更新历史：将预测值加入，用于下一步的滞后特征
    history.append(y_pred)
```

#### 优点
- ✅ 只需训练一个模型
- ✅ 模型简单，训练快
- ✅ 适合长期预测

#### 缺点
- ❌ 误差会累积(越往后越不准)
- ❌ 预测速度慢(需要H次预测)
- ❌ 对初始误差敏感

#### 适用场景
- 长期预测
- 训练数据有限
- 计算资源受限

---

### 4️⃣ USMDR - 单变量多步直接递归预测

**英文名**: Univariate Single-target Multi-step Direct-Recursive forecast

**核心思想**: 结合直接和递归策略，将H步分成K个块，块内直接预测，块间递归

#### 特征构成
```python
特征 X = [load_lag_1, load_lag_2, load_lag_7, ...]
```

#### 目标构成
```python
目标 Y = load_t+1
```

#### 预测过程(分块递归)
```python
block_size = min(lags)  # 例如: 3
predictions = []

for block_start in range(0, H, block_size):
    block_end = min(block_start + block_size, H)
    
    for step in range(block_start, block_end):
        X = [构建滞后特征(包含块内已预测值)]
        y_pred = model.predict(X)
        predictions.append(y_pred)
        
        # 只在块内更新
        if step < block_end - 1:
            history.append(y_pred)
```

#### 优点
- ✅ 平衡了直接和递归的优点
- ✅ 减少误差累积
- ✅ 相对灵活

#### 缺点
- ❌ 复杂度介于中间
- ❌ 需要调整块大小

#### 适用场景
- 中长期预测
- 需要权衡准确性和效率

---

### 5️⃣ MSMD - 多变量多步直接预测 ⭐(重点修复)

**英文名**: Multivariate Single-target Multi-step Direct forecast

**核心思想**: 使用所有内生变量(不仅是目标变量)的滞后特征，预测目标变量的未来H步

#### 特征构成(关键区别！)
```python
假设内生变量 = [load, temperature, humidity]
滞后 = [1, 2, 7, 14]

特征 X = [
    # 目标变量的滞后
    load_lag_1, load_lag_2, load_lag_7, load_lag_14,
    
    # 其他内生变量的滞后(USMD没有这些!)
    temperature_lag_1, temperature_lag_2, temperature_lag_7, temperature_lag_14,
    humidity_lag_1, humidity_lag_2, humidity_lag_7, humidity_lag_14,
    
    # 外生变量
    hour, day_of_week, is_holiday, ...
]

# 特征数 = (3内生变量 × 4滞后) + num_exogenous = 12 + num_exogenous
```

#### 目标构成
```python
目标 Y = [load_t+1, load_t+2, ..., load_t+H]
       # 只预测目标变量，不预测其他内生变量
```

#### 训练过程
```python
输入: (N, num_endogenous*num_lags + num_exogenous)
输出: (N, H)  # 只输出目标变量的H步
模型: MultiOutputRegressor(LGBMRegressor())
```

#### 预测过程
```python
# 1. 获取所有内生变量的历史值
history_load = [100, 102, 101, 103]
history_temp = [20, 21, 19, 20]
history_humidity = [60, 62, 61, 63]

# 2. 构建特征
X_test = [
    history_load[-1], history_load[-2], history_load[-7], history_load[-14],  # load的滞后
    history_temp[-1], history_temp[-2], history_temp[-7], history_temp[-14],  # temp的滞后
    history_humidity[-1], ...                                                  # humidity的滞后
    hour, day_of_week, ...                                                     # 外生变量
]

# 3. 一次性预测
Y_pred = model.predict(X_test)  # 输出所有H步
```

#### 与 USMD 的核心区别
```python
# USMD 的特征
X_usmd = [load_lag_1, load_lag_2, ..., exogenous]
       # 只有目标变量的滞后

# MSMD 的特征
X_msmd = [load_lag_1, load_lag_2, ...,           # 目标变量的滞后
          temperature_lag_1, temperature_lag_2, ...,  # 其他内生变量的滞后
          humidity_lag_1, humidity_lag_2, ...,
          exogenous]
       # 所有内生变量的滞后

# 特征数量
USMD: 约10-20个特征
MSMD: 约30-60个特征(取决于内生变量数量)
```

#### 优点
- ✅ 捕捉多变量之间的相关性
- ✅ 更丰富的信息
- ✅ 不会累积误差
- ✅ 性能通常优于USMD

#### 缺点
- ❌ 需要所有内生变量的历史数据
- ❌ 特征维度更高
- ❌ 训练成本更高

#### 适用场景
- 多个变量之间有强相关性
- 有所有内生变量的历史数据
- 追求更高的预测精度

#### 实际例子
```
电力负荷预测:
- 目标: load(电力负荷)
- 其他内生变量: temperature(温度), humidity(湿度), wind_speed(风速)
  (这些都是时间序列，随时间变化)
- 外生变量: hour, day_of_week, is_holiday

USMD:
  只用 load 的历史 → 预测 load
  
MSMD:
  用 load、temperature、humidity、wind_speed 的历史 → 预测 load
  (因为温度、湿度等会影响电力负荷，用它们的历史信息可以提高预测精度)
```

---

### 6️⃣ MSMR - 多变量多步递归预测

**英文名**: Multivariate Multi-target Multi-step Recursive forecast

**核心思想**: 递归预测所有内生变量，保持变量间的依赖关系

#### 特征构成
```python
特征 X = [load_lag_1, load_lag_2, ...,
         temperature_lag_1, temperature_lag_2, ...,
         humidity_lag_1, humidity_lag_2, ...,
         hour, day_of_week, ...]
```

#### 目标构成(关键区别！)
```python
目标 Y = [load_t+1, temperature_t+1, humidity_t+1]
       # 预测所有内生变量的下一步(不只是目标变量)
```

#### 训练过程
```python
输入: (N, num_endogenous*num_lags + num_exogenous)
输出: (N, num_endogenous)  # 输出所有内生变量
模型: MultiOutputRegressor(LGBMRegressor())
```

#### 预测过程(递归)
```python
predictions_load = []
predictions_temp = []
predictions_humidity = []

for step in range(H):
    X = [构建所有内生变量的滞后特征]
    
    # 预测所有内生变量
    [load_pred, temp_pred, humidity_pred] = model.predict(X)
    
    predictions_load.append(load_pred)
    predictions_temp.append(temp_pred)
    predictions_humidity.append(humidity_pred)
    
    # 更新所有内生变量的历史(递归)
    history_load.append(load_pred)
    history_temp.append(temp_pred)
    history_humidity.append(humidity_pred)

# 返回目标变量的预测
return predictions_load
```

#### 与 MSMD 的区别
```python
# MSMD
输入: 所有内生变量的滞后
输出: 只有目标变量的H步
策略: 直接预测，一次性输出

# MSMR
输入: 所有内生变量的滞后
输出: 所有内生变量的1步(但递归H次)
策略: 递归预测，保持变量间依赖
```

#### 优点
- ✅ 保持所有变量的时间一致性
- ✅ 变量间依赖关系建模
- ✅ 可以同时得到所有变量的预测

#### 缺点
- ❌ 所有变量的误差都会累积
- ❌ 计算成本高
- ❌ 需要所有内生变量的未来预测

#### 适用场景
- 需要同时预测多个变量
- 变量间依赖关系复杂
- 系统建模和仿真

---

### 7️⃣ MSMDR - 多变量多步直接递归预测

**英文名**: Multivariate Multi-target Multi-step Direct-Recursive forecast

**核心思想**: MSMD + MSMR 的结合，分块递归预测所有内生变量

#### 特征和目标
与 MSMR 相同，但采用分块递归策略

#### 优点
- ✅ 减少误差累积
- ✅ 保持变量依赖关系

#### 缺点
- ❌ 实现复杂
- ❌ 需要调参

---

## 📈 方法选择决策树

```
开始
│
├─ 是否有目标变量的历史数据？
│  ├─ 否 → USMDO (只能用外生变量)
│  └─ 是 → 继续
│
├─ 是否有其他内生变量？
│  ├─ 否 (只有目标变量) → 单变量方法
│  │  ├─ 预测horizon短(H≤12)？
│  │  │  ├─ 是 → USMD (直接预测，不累积误差)
│  │  │  └─ 否 → 继续
│  │  └─ 能接受误差累积？
│  │     ├─ 是 → USMR (递归，训练简单)
│  │     └─ 否 → USMDR (分块递归，折中方案)
│  │
│  └─ 是 (有多个内生变量) → 多变量方法
│     ├─ 只需要预测目标变量？
│     │  ├─ 是 → MSMD (最推荐!)
│     │  └─ 否 → 需要预测所有内生变量
│     │     ├─ horizon短？
│     │     │  ├─ 是 → MSMR (递归预测所有变量)
│     │     │  └─ 否 → MSMDR (分块递归)
│     └─ 继续
```

---

## 🎯 推荐策略

### 场景1: 电力负荷预测
- 有温度、湿度等相关变量 → **MSMD** ⭐
- 只有负荷历史数据 → **USMD**
- 短期(1-24小时) → **USMD** 或 **MSMD**
- 长期(1-7天) → **USMR** 或 **MSMR**

### 场景2: 股票价格预测
- 多个相关股票 → **MSMD**
- 单一股票 → **USMD**
- 高频交易(短期) → **USMD**
- 趋势预测(长期) → **USMR**

### 场景3: 气象预测
- 需要预测温度、湿度、气压等 → **MSMR**
- 只需要预测温度 → **MSMD**

---

## 🔧 实现checklist

### MSMD 实现要点(当前需要修复)

✅ **特征工程**
```python
# create_features 中
df_series_copy, multi_lag_features, _ = self.extend_lag_feature_multivariate(
    df=df_series_copy,
    endogenous_cols=all_endogenous_for_lags,  # 所有内生变量!
    n_lags=max(self.args.lags),
    horizon=1
)
```

✅ **训练**
```python
# 目标只包含目标变量的H步
target_output_features = [f"{target_feature}_shift_{h}" for h in range(H)]
```

✅ **预测**
```python
# 使用所有内生变量的历史构建滞后特征
relevant_history_for_lags = df_history.iloc[-max_lag:].copy()
# 确保所有内生变量都在
for endo_feat in endogenous_features:
    if endo_feat not in relevant_history_for_lags.columns:
        relevant_history_for_lags[endo_feat] = df_history[endo_feat].iloc[-max_lag:]
```

---

## 📊 性能对比(经验值)

| 方法 | 短期精度 | 长期精度 | 训练速度 | 预测速度 | 数据需求 |
|-----|---------|---------|---------|---------|---------|
| USMDO | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| USMD | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| USMR | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| USMDR | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| MSMD | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| MSMR | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| MSMDR | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**综合推荐: MSMD** - 在大多数场景下性能最佳！

# 时间序列平稳性

![img](docs/ts_stationary.png)
