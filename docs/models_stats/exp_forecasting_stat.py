# -*- coding: utf-8 -*-
"""
基于统计模型的时间序列预测框架
=================================

支持的统计模型:
1. ARIMA - 自回归积分滑动平均模型
2. SARIMA - 季节性ARIMA
3. ETS - 指数平滑
4. Prophet - Facebook Prophet
5. VAR - 向量自回归（多变量）

作者: Zhefeng Wang
日期: 2026-02-11
版本: 1.0
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


class StatisticalModel(ABC):
    """统计模型基类"""
    
    @abstractmethod
    def fit(self, y, exog=None):
        """拟合模型"""
        pass
    
    @abstractmethod
    def forecast(self, steps, exog=None):
        """预测"""
        pass


class ARIMAModel(StatisticalModel):
    """ARIMA模型"""
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.fitted_model = None
    
    def fit(self, y, exog=None):
        """拟合ARIMA模型"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(y, order=self.order, exog=exog)
            self.fitted_model = model.fit()
            return self
        except ImportError:
            raise ImportError("需要安装 statsmodels: pip install statsmodels")
    
    def forecast(self, steps, exog=None):
        """预测未来steps步"""
        if self.fitted_model is None:
            raise ValueError("模型未训练")
        return self.fitted_model.forecast(steps=steps, exog=exog)


class SARIMAModel(StatisticalModel):
    """SARIMA模型（季节性ARIMA）"""
    
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
    
    def fit(self, y, exog=None):
        """拟合SARIMA模型"""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=exog
            )
            self.fitted_model = model.fit(disp=False)
            return self
        except ImportError:
            raise ImportError("需要安装 statsmodels: pip install statsmodels")
    
    def forecast(self, steps, exog=None):
        """预测"""
        if self.fitted_model is None:
            raise ValueError("模型未训练")
        return self.fitted_model.forecast(steps=steps, exog=exog)


class ProphetModel(StatisticalModel):
    """Prophet模型（Facebook开发）"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
    
    def fit(self, y, exog=None):
        """拟合Prophet模型"""
        try:
            from prophet import Prophet
            self.model = Prophet(**self.kwargs)
            
            # 准备数据格式
            df = pd.DataFrame({
                'ds': y.index if isinstance(y, pd.Series) else range(len(y)),
                'y': y.values if isinstance(y, pd.Series) else y
            })
            
            # 添加外生变量
            if exog is not None:
                for col in exog.columns:
                    df[col] = exog[col].values
                    self.model.add_regressor(col)
            
            self.model.fit(df)
            return self
        except ImportError:
            raise ImportError("需要安装 prophet: pip install prophet")
    
    def forecast(self, steps, exog=None):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        future = self.model.make_future_dataframe(periods=steps)
        
        if exog is not None:
            for col in exog.columns:
                future[col] = exog[col].values[:len(future)]
        
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(steps).values


class ETSModel(StatisticalModel):
    """ETS模型（指数平滑）"""
    
    def __init__(self, seasonal_periods=12):
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
    
    def fit(self, y, exog=None):
        """拟合ETS模型"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.fitted_model = ExponentialSmoothing(
                y,
                seasonal_periods=self.seasonal_periods,
                seasonal='add'
            ).fit()
            return self
        except ImportError:
            raise ImportError("需要安装 statsmodels: pip install statsmodels")
    
    def forecast(self, steps, exog=None):
        """预测"""
        if self.fitted_model is None:
            raise ValueError("模型未训练")
        return self.fitted_model.forecast(steps)


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    y = pd.Series(
        np.random.randn(365).cumsum() + 100,
        index=dates,
        name='target'
    )
    
    # 测试ARIMA
    print("测试 ARIMA 模型:")
    print("=" * 50)
    arima = ARIMAModel(order=(2,1,2))
    arima.fit(y)
    forecast = arima.forecast(steps=30)
    print(f"预测未来30天: {forecast[:5]}...")
    
    # 测试SARIMA
    print("\n测试 SARIMA 模型:")
    print("=" * 50)
    sarima = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima.fit(y)
    forecast = sarima.forecast(steps=30)
    print(f"预测未来30天: {forecast[:5]}...")
