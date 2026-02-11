# -*- coding: utf-8 -*-

# ***************************************************
# * File        : advanced_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : 特征工程：
# *               基本特征：
# *                 - 内生变量特征
# *                     - 日期时间特征(小时、星期、月份、季度等)、周期性编码(sin/cos)
# *                     - 天气特征(天气数据集成)
# *                     - 节假日 标记特征(日期类型数据集成)
# *                 - 内生变量特征：
# *                     - 滞后特征：单变量(目标变量)滞后特征、多变量(目标变量、其他内生变量)滞后特征
# *               高级特征：
# *                 - 内生变量特征
# *                     - 滑动窗口统计特征 (Rolling Window Statistics)
# *                         - load_rolling_mean_3   # 最近3步平均值
# *                         - load_rolling_std_7    # 最近7步标准差
# *                         - load_rolling_min_12   # 最近12步最小值
# *                         - load_rolling_max_12   # 最近12步最大值
# *                     - 扩展窗口统计特征 (Expanding Window Statistics)
# *                         - load_expanding_mean   # 累积平均值
# *                         - load_expanding_std    # 累积标准差
# *                     - 差分特征 (Difference Features)
# *                         - load_diff_1          # 一阶差分
# *                         - load_diff_seasonal   # 季节性差分
# *                     - 时间距离特征 (Time-based Features)
# *                         - time_since_peak      # 距离峰值的时间
# *                         - time_since_low      # 距离谷值的时间
# *                 - 交叉特征
# *                     - hour_x_load_lag_1 = hour * load_lag_1     # 时间 × 滞后值
# *                     - temp_x_humidity = temperature * humidity  # 温度 × 湿度
# *                     - load_lag_1_squared = load_lag_1 ** 2      # 多项式特征
# *               目标编码
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


class EnhancedFeaturePreprocessor(FeaturePreprocessor):
    """增强的特征预处理器"""
    
    def add_lag_statistics(self, df: pd.DataFrame, column: str, windows: List[int]):
        """
        添加滞后统计特征
        
        Args:
            df: 数据框
            column: 列名
            windows: 窗口大小列表 [3, 7, 14]
        
        Returns:
            增强后的数据框
        """
        df_enhanced = df.copy()
        
        for window in windows:
            # 滑动平均
            df_enhanced[f'{column}_rolling_mean_{window}'] = (
                df[column].rolling(window=window, min_periods=1).mean()
            )
            
            # 滑动标准差
            df_enhanced[f'{column}_rolling_std_{window}'] = (
                df[column].rolling(window=window, min_periods=1).std()
            )
            
            # 滑动最小值
            df_enhanced[f'{column}_rolling_min_{window}'] = (
                df[column].rolling(window=window, min_periods=1).min()
            )
            
            # 滑动最大值
            df_enhanced[f'{column}_rolling_max_{window}'] = (
                df[column].rolling(window=window, min_periods=1).max()
            )
            
            # 滑动中位数
            df_enhanced[f'{column}_rolling_median_{window}'] = (
                df[column].rolling(window=window, min_periods=1).median()
            )
        
        return df_enhanced
    
    def add_diff_features(self, df: pd.DataFrame, column: str, periods: List[int]):
        """
        添加差分特征
        
        Args:
            df: 数据框
            column: 列名
            periods: 差分周期列表 [1, 7, 24]
        """
        df_enhanced = df.copy()
        
        for period in periods:
            df_enhanced[f'{column}_diff_{period}'] = df[column].diff(period)
        
        return df_enhanced
    
    def add_expanding_features(self, df: pd.DataFrame, column: str):
        """添加扩展窗口特征"""
        df_enhanced = df.copy()
        
        df_enhanced[f'{column}_expanding_mean'] = df[column].expanding().mean()
        df_enhanced[f'{column}_expanding_std'] = df[column].expanding().std()
        df_enhanced[f'{column}_expanding_min'] = df[column].expanding().min()
        df_enhanced[f'{column}_expanding_max'] = df[column].expanding().max()
        
        return df_enhanced
    
    def add_time_based_features(self, df: pd.DataFrame, column: str, time_column: str):
        """添加基于时间的特征"""
        df_enhanced = df.copy()
        
        # 距离峰值的时间
        peak_idx = df[column].idxmax()
        df_enhanced[f'{column}_time_since_peak'] = (df.index - peak_idx).total_seconds() / 3600
        
        # 距离谷值的时间
        low_idx = df[column].idxmin()
        df_enhanced[f'{column}_time_since_low'] = (df.index - low_idx).total_seconds() / 3600
        
        return df_enhanced





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
