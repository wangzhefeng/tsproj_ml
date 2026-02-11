# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MITSUI_CO_baseline.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-01-29
# * Version     : 1.0.012923
# * Description : description
# * Link        : 最新量化赛事方案: https://mp.weixin.qq.com/s/M40VUuuCx1BQ3EzQZEdAeA
# *             : kaggle: https://www.kaggle.com/code/yuanzhezhou/mitsui-co-baseline-train-infer/notebook
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
warnings.filterwarnings("ignore", message="DeprecationWarning")
warnings.filterwarnings("ignore", message="See the caveats")
# 屏蔽特定的 FutureWarning
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated."
)
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message="invalid value encountered in "
)

import numpy as np
import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ##############################
# evaluation metrics
# ##############################
SOLUTION_NULL_FILLER = -999999


def rank_correlation_sharpe_ratio(merged_df: pd.DataFrame) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).

    :param merged_df: DataFrame containing prediction columns (starting with 'prediction_')
                      and target columns (starting with 'target_')
    :return: Sharpe ratio of the rank correlation
    :raises ZeroDivisionError: If the standard deviation is zero
    """
    prediction_cols = [col for col in merged_df.columns if col.startswith('prediction_')]
    target_cols = [col for col in merged_df.columns if col.startswith('target_')]


    def _compute_rank_correlation(row):
        non_null_targets = [col for col in target_cols if not pd.isnull(row[col])]
        matching_predictions = [col for col in prediction_cols if col.replace('prediction', 'target') in non_null_targets]
        if not non_null_targets:
            raise ValueError('No non-null target values found')
        if row[non_null_targets].std(ddof=0) == 0 or row[matching_predictions].std(ddof=0) == 0:
            raise ZeroDivisionError('Denominator is zero, unable to compute rank correlation.')
        return np.corrcoef(row[matching_predictions].rank(method='average'), row[non_null_targets].rank(method='average'))[0, 1]

    daily_rank_corrs = merged_df.apply(_compute_rank_correlation, axis=1)
    std_dev = daily_rank_corrs.std(ddof=0)
    if std_dev == 0:
        raise ZeroDivisionError('Denominator is zero, unable to compute Sharpe ratio.')
    sharpe_ratio = daily_rank_corrs.mean() / std_dev
    return float(sharpe_ratio)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str='') -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert all(solution.columns == submission.columns)

    submission = submission.rename(columns={col: col.replace('target_', 'prediction_') for col in submission.columns})

    # Not all securities trade on all dates, but solution files cannot contain nulls.
    # The filler value allows us to handle trading halts, holidays, & delistings.
    solution = solution.replace(SOLUTION_NULL_FILLER, None)
    return rank_correlation_sharpe_ratio(pd.concat([solution.reset_index(drop=True), submission.reset_index(drop=True)], axis='columns'))
# ##############################
# data preprocessing and feature engineering
# ##############################
def preprocess(df):
    df = df.copy()
    df = df.rename(columns={'date_id': 'date'})
    
    # 创建空的结果 DataFrame
    result = pd.DataFrame(columns=['date', 'id', 'close', 'open', 'high', 'low', 'volume', 'sprice', 'interest'])
    # ------------------------------
    # 处理 LME 数据
    # ------------------------------
    lme_metals = ['AH', 'CA', 'PB', 'ZS']
    for metal in lme_metals:
        temp_df = pd.DataFrame()
        temp_df['date'] = df['date']
        temp_df['id'] = f'LME_{metal}'
        temp_df['close'] = df[f'LME_{metal}_Close']
        # LME 数据没有其他字段，设置为 NaN
        temp_df['open'] = None
        temp_df['high'] = None
        temp_df['low'] = None
        temp_df['volume'] = None
        temp_df['sprice'] = None
        temp_df['interest'] = None
        result = pd.concat([result, temp_df], ignore_index=True)
    
    # ------------------------------
    # 处理 JPX 期货数据
    # ------------------------------
    jpx_products = {
        'Gold_Mini': ['Open', 'High', 'Low', 'Close', 'Volume', 'settlement_price', 'open_interest'],
        'Gold_Rolling-Spot': ['Open', 'High', 'Low', 'Close', 'Volume', 'settlement_price', 'open_interest'],
        'Gold_Standard': ['Open', 'High', 'Low', 'Close', 'Volume', 'open_interest'],
        'Platinum_Mini': ['Open', 'High', 'Low', 'Close', 'Volume', 'settlement_price', 'open_interest'],
        'Platinum_Standard': ['Open', 'High', 'Low', 'Close', 'Volume', 'open_interest'],
        'RSS3_Rubber': ['Open', 'High', 'Low', 'Close', 'Volume', 'settlement_price', 'open_interest']
    }
    for product, columns in jpx_products.items():
        temp_df = pd.DataFrame()
        temp_df['date'] = df['date']
        temp_df['id'] = f'JPX_{product}'
        
        if 'Close' in columns:
            temp_df['close'] = df[f'JPX_{product}_Futures_Close']
        if 'Open' in columns:
            temp_df['open'] = df[f'JPX_{product}_Futures_Open']
        if 'High' in columns:
            temp_df['high'] = df[f'JPX_{product}_Futures_High']
        if 'Low' in columns:
            temp_df['low'] = df[f'JPX_{product}_Futures_Low']
        if 'Volume' in columns:
            temp_df['volume'] = df[f'JPX_{product}_Futures_Volume']
        if 'settlement_price' in columns:
            temp_df['sprice'] = df[f'JPX_{product}_Futures_settlement_price']
        if 'open_interest' in columns:
            temp_df['interest'] = df[f'JPX_{product}_Futures_open_interest']
        
        result = pd.concat([result, temp_df], ignore_index=True)
    
    # ------------------------------
    # 处理 US Stock 数据
    # ------------------------------
    us_stocks = [
        'ACWI', 'AEM', 'AG', 'AGG', 'ALB', 'AMP', 'BCS', 'BKR', 'BND', 'BNDX', 
        'BP', 'BSV', 'CAT', 'CCJ', 'CLF', 'COP', 'CVE', 'CVX', 'DE', 'DVN', 
        'EEM', 'EFA', 'EMB', 'ENB', 'EOG', 'EWJ', 'EWT', 'EWY', 'EWZ', 'FCX', 
        'FNV', 'FXI', 'GDX', 'GDXJ', 'GLD', 'GOLD', 'HAL', 'HES', 'HL', 'IAU', 
        'IEF', 'IEMG', 'IGSB', 'JNK', 'KGC', 'KMI', 'LQD', 'LYB', 'MBB', 'MPC', 
        'MS', 'NEM', 'NUE', 'NUGT', 'OIH', 'OKE', 'OXY', 'PAAS', 'RIO', 'RSP', 
        'RY', 'SCCO', 'SHEL', 'SHY', 'SLB', 'SLV', 'SPIB', 'SPTL', 'SPYV', 'STLD', 
        'TD', 'TECK', 'TIP', 'TRGP', 'URA', 'VALE', 'VCIT', 'VCSH', 'VEA', 'VGIT', 
        'VGK', 'VGLT', 'VGSH', 'VT', 'VTV', 'VWO', 'VXUS', 'VYM', 'WMB', 'WPM', 
        'X', 'XLB', 'XLE', 'XOM', 'YINN'
    ]
    for stock in us_stocks:
        temp_df = pd.DataFrame()
        temp_df['date'] = df['date']
        temp_df['id'] = f'US_Stock_{stock}'
        temp_df['close'] = df[f'US_Stock_{stock}_adj_close']
        temp_df['open'] = df[f'US_Stock_{stock}_adj_open']
        temp_df['high'] = df[f'US_Stock_{stock}_adj_high']
        temp_df['low'] = df[f'US_Stock_{stock}_adj_low']
        temp_df['volume'] = df[f'US_Stock_{stock}_adj_volume']
        # US Stock 数据没有 sprice 和 interest，设置为 NaN
        temp_df['sprice'] = None
        temp_df['interest'] = None
        result = pd.concat([result, temp_df], ignore_index=True)
    
    # ------------------------------
    # 处理 FX 数据
    # ------------------------------
    fx_pairs = [
        'AUDJPY', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURGBP', 'EURJPY', 
        'EURUSD', 'GBPAUD', 'GBPJPY', 'GBPUSD', 'NZDJPY', 'NZDUSD', 'USDCHF', 
        'USDJPY', 'ZARJPY', 'ZARUSD', 'NOKUSD', 'NOKEUR', 'CADUSD', 'AUDNZD', 
        'EURCHF', 'EURCAD', 'AUDCAD', 'GBPCHF', 'EURNZD', 'AUDCHF', 'GBPNZD', 
        'GBPCAD', 'CADCHF', 'NZDCAD', 'NZDCHF', 'ZAREUR', 'NOKGBP', 'NOKCHF', 
        'ZARCHF', 'NOKJPY', 'ZARGBP'
    ]
    for pair in fx_pairs:
        temp_df = pd.DataFrame()
        temp_df['date'] = df['date']
        temp_df['id'] = f'FX_{pair}'
        temp_df['close'] = df[f'FX_{pair}']
        # FX 数据只有收盘价，其他字段设置为 NaN
        temp_df['open'] = None
        temp_df['high'] = None
        temp_df['low'] = None
        temp_df['volume'] = None
        temp_df['sprice'] = None
        temp_df['interest'] = None
        result = pd.concat([result, temp_df], ignore_index=True)
    # ------------------------------
    # 结果处理
    # ------------------------------
    # 重置索引
    result = result.reset_index(drop=True)
    # 确保数据按日期排序
    result = result.sort_values(['id', 'date']).reset_index(drop=True)
    
    return result


def create_features(df, windows=[5, 10, 20]):
    # 按资产分组处理
    grouped = df.groupby('id')
    
    features_list = []
    for asset_id, group in grouped:
        group = group.copy()

        for col1 in ['close', 'open', 'high', 'low']:
            for col2 in ['close', 'open', 'high', 'low']:
                if col1 > col2:
                    group[f'{col1}/{col2}'] = (group[col1] - group[col2]) / (group[col1] + group[col2])

        group['open/close_shift1'] = group['open'] / group['close'].shift(1)
        
        for window in windows:
            group[f'ret_{window}'] = group['close'] / group['close'].shift(window) - 1
            group[f'vol_{window}'] = (group['close'] / group['close'].shift(1) - 1).rolling(window).std()
            group[f'volume_{window}'] = group['volume'].rolling(window).mean() / group['volume'].rolling(window * 2).mean()
            group[f'technical1_{window}'] = (group['close'] > group['high'].ffill().shift(1)).astype('float') - (group['close'] < group['low'].ffill().shift(1)).astype('float')
            group[f'technical2_{window}'] = (group['low'] > group['high'].ffill().shift(1)).astype('float') - (group['high'] < group['low'].ffill().shift(1)).astype('float')
        
        # sprice和interest相关特征
        if 'sprice' in group.columns:
            group['sprice_change'] = group['sprice'] / group['sprice'].ffill().shift(1) - 1
            group['premium_discount'] = (group['close'] - group['sprice']) / group['sprice']
        
        if 'interest' in group.columns:
            group['volume_interest_ratio'] = group['volume'] / (group['interest'] + 1)  
        
        features_list.append(group)
    
    # 合并所有资产的特征
    features_df = pd.concat(features_list, ignore_index=True)
    features_df = features_df.sort_values(by=['date', 'id'])
    # id_num 特征
    dict_ = {
        'FX_AUDCAD': 0, 'FX_AUDCHF': 1, 'FX_AUDJPY': 2, 'FX_AUDNZD': 3, 'FX_AUDUSD': 4, 'FX_CADCHF': 5, 'FX_CADJPY': 6, 'FX_CADUSD': 7, 'FX_CHFJPY': 8, 'FX_EURAUD': 9, 'FX_EURCAD': 10, 'FX_EURCHF': 11, 'FX_EURGBP': 12, 'FX_EURJPY': 13, 'FX_EURNZD': 14, 'FX_EURUSD': 15, 'FX_GBPAUD': 16, 'FX_GBPCAD': 17, 'FX_GBPCHF': 18, 'FX_GBPJPY': 19, 'FX_GBPNZD': 20, 'FX_GBPUSD': 21, 'FX_NOKCHF': 22, 'FX_NOKEUR': 23, 'FX_NOKGBP': 24, 'FX_NOKJPY': 25, 'FX_NOKUSD': 26, 'FX_NZDCAD': 27, 'FX_NZDCHF': 28, 'FX_NZDJPY': 29, 'FX_NZDUSD': 30, 'FX_USDCHF': 31, 'FX_USDJPY': 32, 'FX_ZARCHF': 33, 'FX_ZAREUR': 34, 'FX_ZARGBP': 35, 'FX_ZARJPY': 36, 'FX_ZARUSD': 37, 
        'JPX_Gold_Mini': 38, 'JPX_Gold_Rolling-Spot': 39, 'JPX_Gold_Standard': 40, 'JPX_Platinum_Mini': 41, 'JPX_Platinum_Standard': 42, 'JPX_RSS3_Rubber': 43, 
        'LME_AH': 44, 'LME_CA': 45, 'LME_PB': 46, 'LME_ZS': 47, 
        'US_Stock_ACWI': 48, 'US_Stock_AEM': 49, 'US_Stock_AG': 50, 'US_Stock_AGG': 51, 'US_Stock_ALB': 52, 'US_Stock_AMP': 53, 'US_Stock_BCS': 54, 'US_Stock_BKR': 55, 'US_Stock_BND': 56, 'US_Stock_BNDX': 57, 'US_Stock_BP': 58, 'US_Stock_BSV': 59, 'US_Stock_CAT': 60, 'US_Stock_CCJ': 61, 'US_Stock_CLF': 62, 'US_Stock_COP': 63, 'US_Stock_CVE': 64, 'US_Stock_CVX': 65, 'US_Stock_DE': 66, 'US_Stock_DVN': 67, 'US_Stock_EEM': 68, 'US_Stock_EFA': 69, 'US_Stock_EMB': 70, 'US_Stock_ENB': 71, 'US_Stock_EOG': 72, 'US_Stock_EWJ': 73, 'US_Stock_EWT': 74, 'US_Stock_EWY': 75, 'US_Stock_EWZ': 76, 'US_Stock_FCX': 77, 'US_Stock_FNV': 78, 'US_Stock_FXI': 79, 'US_Stock_GDX': 80, 'US_Stock_GDXJ': 81, 'US_Stock_GLD': 82, 'US_Stock_GOLD': 83, 'US_Stock_HAL': 84, 'US_Stock_HES': 85, 'US_Stock_HL': 86, 'US_Stock_IAU': 87, 'US_Stock_IEF': 88, 'US_Stock_IEMG': 89, 'US_Stock_IGSB': 90, 'US_Stock_JNK': 91, 'US_Stock_KGC': 92, 'US_Stock_KMI': 93, 'US_Stock_LQD': 94, 'US_Stock_LYB': 95, 'US_Stock_MBB': 96, 'US_Stock_MPC': 97, 'US_Stock_MS': 98, 'US_Stock_NEM': 99, 'US_Stock_NUE': 100, 'US_Stock_NUGT': 101, 'US_Stock_OIH': 102, 'US_Stock_OKE': 103, 'US_Stock_OXY': 104, 'US_Stock_PAAS': 105, 'US_Stock_RIO': 106, 'US_Stock_RSP': 107, 'US_Stock_RY': 108, 'US_Stock_SCCO': 109, 'US_Stock_SHEL': 110, 'US_Stock_SHY': 111, 'US_Stock_SLB': 112, 'US_Stock_SLV': 113, 'US_Stock_SPIB': 114, 'US_Stock_SPTL': 115, 'US_Stock_SPYV': 116, 'US_Stock_STLD': 117, 'US_Stock_TD': 118, 'US_Stock_TECK': 119, 'US_Stock_TIP': 120, 'US_Stock_TRGP': 121, 'US_Stock_URA': 122, 'US_Stock_VALE': 123, 'US_Stock_VCIT': 124, 'US_Stock_VCSH': 125, 'US_Stock_VEA': 126, 'US_Stock_VGIT': 127, 'US_Stock_VGK': 128, 'US_Stock_VGLT': 129, 'US_Stock_VGSH': 130, 'US_Stock_VT': 131, 'US_Stock_VTV': 132, 'US_Stock_VWO': 133, 'US_Stock_VXUS': 134, 'US_Stock_VYM': 135, 'US_Stock_WMB': 136, 'US_Stock_WPM': 137, 'US_Stock_X': 138, 'US_Stock_XLB': 139, 'US_Stock_XLE': 140, 'US_Stock_XOM': 141, 'US_Stock_YINN': 142
    }
    features_df['id_num'] = features_df['id'].map(dict_)
    
    return features_df


class CFG:
    if Path('./dataset/kaggle/mitsui-commodity-prediction-challenge').exists():
        input_path = './dataset/kaggle/mitsui-commodity-prediction-challenge/'
    else:
        input_path = '/kaggle/input/mitsui-commodity-prediction-challenge/'

    num_valid = 134
# ##############################
# model training
# ##############################
class Modelmultilabel():
    
    def __init__(self, config, input_path=None):
        self.models = []

        if input_path is not None:
            self.load_model(input_path)

        self.config = config

    def preprocess(self, df_features):
        feature_names = [
            col for col in df_features.columns 
            if (col not in ['date', 'id', 'label_d1', 'high', 'low', 'open', 'interest', 'sprice']) and ('label' not in col)
        ]
        # print(feature_names)
        
        final_features = []
        for date, df_tmp in df_features.groupby('date'):
            final_features.append(df_tmp[feature_names].values.ravel())
        
        final_features = np.vstack(final_features)
        return final_features
    
    def train(self, df_features, N_valid=134):
        final_features = self.preprocess(df_features)
        # features
        x_train = final_features[:-N_valid]
        x_test = final_features[-N_valid:]
        # labels
        df_labels = pd.read_csv(f'{self.config.input_path}/train_labels.csv').set_index('date_id')
        y_train = df_labels.head(df_labels.shape[0]-N_valid)
        y_test = df_labels.tail(N_valid)
        # model
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=2,
            boosting_type='Plain',
            loss_function='MultiRMSE',  # 多输出回归使用MultiRMSE损失
            verbose=20,
            random_state=42,
            task_type='GPU',
            reg_lambda=2000
        )
        
        # 训练模型
        model.fit(
            x_train, 
            y_train.fillna(0).values.argsort(-1).argsort(-1) / 424,
            eval_set=(
                x_test, 
                y_test.fillna(0).values.argsort(-1).argsort(-1)/424
            ),
            early_stopping_rounds=200
        )
        self.models.append(model)

        # 模型推理
        predictions = self.predict(df_features)
        pred = y_test.reset_index().copy()
        pred.iloc[:, 1:] = predictions[-len(y_test):]
        print('score:', score(y_test.reset_index().tail(N_valid), pred.tail(N_valid), 'date_id'))

    def predict(self, df_features):
        final_features = self.preprocess(df_features)
        # predict
        predictions = [
            model.predict(final_features) 
            for model in self.models
        ]
        return np.mean(predictions, 0)

    def load_model(self, input_path):
        self.models.append()
# ##############################
# 
# ##############################
class Modelmultilabel_v2():
    def __init__(self, config, input_path=None):
        self.models = []

        if input_path is not None:
            self.load_model(input_path)

        self.config = config

    def preprocess(self, df_features):
        feature_names = [
            col for col in df_features.columns 
            if (col not in ['date', 'id', 'label_d1', 'high', 'low', 'open', 'interest', 'sprice']) and ('label' not in col)
        ]
        # print(feature_names)
        
        return df_features[feature_names].values
    
    def train(self, df_features, N_valid=134):
        final_features = self.preprocess(df_features)

        id_nunique = df_features.id.nunique()
        # features
        x_train = final_features[:-N_valid * id_nunique]
        x_test = final_features[-N_valid * id_nunique:]
        # labels
        df_labels = pd.read_csv(f'{self.config.input_path}/train_labels.csv').set_index('date_id')
        # y_train = df_labels.head(df_labels.shape[0]-N_valid)
        # y_test = df_labels.tail(N_valid)
        df_features['label1'] = -np.log(df_features.groupby('id')['close'].shift(-1-1)/df_features.groupby('id')['close'].shift(-1))
        df_features['label2'] = -np.log(df_features.groupby('id')['close'].shift(-1-2)/df_features.groupby('id')['close'].shift(-1))
        df_features['label3'] = -np.log(df_features.groupby('id')['close'].shift(-1-3)/df_features.groupby('id')['close'].shift(-1))
        df_features['label4'] = -np.log(df_features.groupby('id')['close'].shift(-1-4)/df_features.groupby('id')['close'].shift(-1))
        y_train = df_features.head(df_features.shape[0]-N_valid * id_nunique)[['label1', 'label2', 'label3', 'label4']].fillna(0)
        y_test = df_features.tail(N_valid * id_nunique)[['label1', 'label2', 'label3', 'label4']].fillna(0)

        # model
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.005,
            depth=3,
            boosting_type='Plain',
            loss_function='MultiRMSE',  # 多输出回归使用MultiRMSE损失
            verbose=100,
            random_state=42,
            task_type='GPU',
            reg_lambda=2000
        )
        
        # 训练模型
        model.fit(
            x_train, y_train.values,
            eval_set=(x_test, y_test.values ),
            early_stopping_rounds=200
        )
        self.models.append(model)

        # 模型推理
        predictions = self.predict(df_features)        
        pred = df_labels.tail(N_valid).reset_index().copy()
        pred.iloc[:,1:] = predictions[-N_valid:] + np.random.random(predictions[-N_valid:].shape) * 1e-10
        print('score:', score(df_labels.tail(N_valid).reset_index(), pred.tail(N_valid), 'date_id'))

    def predict(self, df_features):
        final_features = self.preprocess(df_features)

        id_nunique = df_features.id.nunique()

        predictions = [model.predict(final_features) for model in self.models]
        predictions = np.mean(predictions, 0).reshape(-1, id_nunique, 4)

        indexlist = [
            [131, -1], [46, 131], [45, 47], [44, 47], [44, 40], [47, 42], [46, 44], [47, 135], [89, 40], [2, 46], [42, 127], [9, 45], [42, 5], [46, 33], [42, 29], [8, 44], [132, 47], [102, 40], [44, 37], [40, 32], [47, 1], [44, 113], [55, 42], [19, 45], [122, 42], [44, 126], [36, 45], [103, 47], [65, 47], [7, 44], [47, 61], [47, 82], [47, 77], [40, 75], [0, 42], [133, 40], [46, 26], [45, 128], [106, 42], [10, 44], [45, 86], [47, 119], [52, 44], [35, 44], [40, 24], [45, 23], [68, 46], [66, 45], [45, 76], [46, 117], [112, 40], [40, 53], [60, 40], [72, 44], [120, 42], [47, 98], [30, 47], [42, 91], [71, 40], [40, 134], [44, 22], [44, 57], [40, 85], [63, 40], [6, 44], [104, 47], [107, 42], [40, 84], [15, 40], [45, 28], [31, 40], [14, 40], [44, 118], [45, 108], [34, 45], [47, 139], [47, 12], [46, 123], [62, 42], [47, 67], [40, 48], [21, 40], [40, 138], [97, 46], [25, 47], [40, 64], [44, 11], [13, 47], [44, 99], [44, 20], [95, 44], [100, 44], [42, 69], [44, 140], [73, 44], [44, 121], [74, 46], [46, 136], [45, 141], [40, 93], [109, 44], [42, 4], [47, 88], [116, 45], [137, 40], [87, 45], [134, -1], [47, 134], [47, 46], [44, 40], [42, 44], [40, 47], [40, 45], [40, 0], [42, 28], [35, 40], [47, 32], [45, 34], [10, 47], [45, 5], [44, 1], [72, 40], [73, 40], [42, 141], [47, 53], [20, 45], [44, 136], [47, 11], [42, 75], [36, 46], [42, 55], [47, 127], [76, 47], [46, 22], [91, 47], [98, 46], [47, 120], [44, 69], [15, 40], [67, 47], [42, 61], [46, 24], [47, 104], [47, 117], [46, 85], [33, 40], [140, 46], [42, 52], [89, 42], [42, 93], [44, 137], [45, 123], [71, 46], [13, 40], [103, 40], [7, 44], [45, 108], [119, 45], [46, 112], [116, 42], [44, 68], [46, 4], [23, 46], [46, 122], [44, 133], [45, 121], [87, 47], [47, 132], [88, 46], [45, 30], [46, 25], [42, 12], [47, 118], [138, 46], [63, 46], [42, 48], [44, 102], [26, 44], [8, 47], [42, 113], [65, 42], [64, 44], [2, 47], [40, 100], [40, 131], [99, 40], [139, 47], [6, 42], [31, 46], [19, 45], [44, 128], [74, 44], [97, 45], [46, 62], [42, 57], [86, 44], [9, 44], [42, 37], [44, 21], [45, 95], [29, 45], [109, 44], [47, 135], [47, 60], [42, 66], [82, 46], [106, 46], [40, 107], [84, 45], [46, 77], [126, 42], [14, 46], [37, -1], [46, 37], [45, 40], [46, 40], [44, 42], [40, 44], [47, 46], [47, 26], [44, 57], [33, 42], [40, 35], [45, 29], [104, 44], [141, 40], [44, 76], [9, 46], [47, 21], [45, 19], [44, 52], [46, 32], [40, 116], [73, 47], [47, 24], [42, 121], [46, 8], [47, 137], [71, 40], [4, 42], [25, 46], [45, 138], [47, 14], [46, 132], [67, 46], [20, 42], [109, 44], [42, 64], [44, 99], [47, 123], [134, 45], [45, 15], [40, 63], [44, 139], [5, 42], [100, 45], [48, 42], [45, 61], [68, 46], [45, 74], [42, 28], [42, 77], [91, 42], [45, 103], [107, 45], [7, 45], [119, 46], [44, 62], [46, 23], [42, 72], [40, 120], [108, 45], [46, 117], [47, 87], [45, 133], [42, 6], [40, 86], [2, 46], [97, 47], [46, 126], [40, 10], [40, 22], [66, 44], [34, 47], [36, 40], [89, 42], [140, 40], [46, 112], [40, 55], [47, 82], [95, 42], [44, 69], [45, 135], [47, 93], [128, 45], [45, 1], [45, 60], [53, 47], [47, 88], [44, 13], [45, 84], [40, 11], [42, 122], [40, 131], [75, 47], [65, 45], [46, 30], [85, 46], [12, 42], [46, 127], [0, 42], [118, 47], [40, 31], [98, 44], [47, 113], [44, 102], [106, 46], [42, 136], [23, -1], [44, 23], [40, 47], [44, 47], [44, 46], [46, 45], [44, 42], [46, 52], [73, 40], [47, 76], [42, 75], [42, 62], [45, 36], [42, 55], [131, 47], [44, 109], [77, 47], [121, 45], [9, 44], [7, 45], [40, 97], [107, 46], [42, 87], [42, 134], [46, 95], [67, 47], [65, 47], [6, 46], [48, 40], [45, 57], [34, 40], [44, 133], [47, 89], [12, 40], [104, 45], [106, 44], [44, 140], [47, 53], [40, 141], [47, 84], [45, 127], [47, 91], [44, 132], [46, 119], [100, 40], [4, 42], [122, 42], [47, 66], [72, 44], [21, 46], [19, 42], [45, 137], [42, 118], [29, 42], [1, 42], [112, 45], [44, 14], [42, 8], [120, 46], [139, 45], [40, 99], [68, 44], [116, 44], [28, 46], [40, 0], [5, 45], [45, 117], [42, 71], [46, 2], [63, 42], [47, 126], [128, 45], [40, 136], [44, 60], [11, 45], [86, 44], [102, 45], [45, 113], [44, 103], [93, 45], [135, 40], [45, 85], [25, 45], [47, 69], [13, 42], [123, 40], [40, 138], [37, 44], [31, 46], [32, 42], [44, 33], [46, 35], [15, 44], [44, 64], [98, 46], [20, 40], [40, 88], [30, 46], [45, 82], [42, 10], [24, 47], [26, 44], [40, 108], [74, 44], [42, 22], [45, 61]]

        pred_1d = predictions[:, [item[0] for item in indexlist[106*0:106*1]], 0] - predictions[:, [item[1] for item in indexlist[106*0:106*1]], 0] * (np.array([item[1]!=-1 for item in indexlist[106*0:106*1]])).astype('float')
        pred_2d = predictions[:, [item[0] for item in indexlist[106*1:106*2]], 1] - predictions[:, [item[1] for item in indexlist[106*1:106*2]], 1] * (np.array([item[1]!=-1 for item in indexlist[106*1:106*2]])).astype('float')
        pred_3d = predictions[:, [item[0] for item in indexlist[106*2:106*3]], 2] - predictions[:, [item[1] for item in indexlist[106*2:106*3]], 2] * (np.array([item[1]!=-1 for item in indexlist[106*2:106*3]])).astype('float')
        pred_4d = predictions[:, [item[0] for item in indexlist[106*3:106*4]], 3] - predictions[:, [item[1] for item in indexlist[106*3:106*4]], 3] * (np.array([item[1]!=-1 for item in indexlist[106*3:106*4]])).astype('float')

        predictions = np.concatenate([pred_1d, pred_2d, pred_3d, pred_4d,], -1)
        
        return predictions
# ##############################
# 
# ##############################
class Modelmultilabel_v3():
    def __init__(self, config, input_path=None):
        self.models = []

        if input_path is not None:
            self.load_model(input_path)

        self.config = config

    def preprocess(self, df_features):
        feature_names = [
            col for col in df_features.columns 
            if (col not in ['date', 'id', 'label_d1', 'high', 'low', 'open', 'interest', 'sprice']) and ('label' not in col)
        ]
        # print(feature_names)
        
        return df_features[feature_names].values
    
    def train(self, df_features, N_valid=134):
        final_features = self.preprocess(df_features)

        id_nunique = df_features.id.nunique()
        # labels
        df_labels = pd.read_csv(f'{self.config.input_path}/train_labels.csv').set_index('date_id')
        y_train = df_labels.head(df_labels.shape[0]-N_valid).fillna(0).values.argsort(-1).argsort(-1).ravel() / 424
        y_test = df_labels.tail(N_valid).fillna(0).values.argsort(-1).argsort(-1).ravel() / 424
        # features
        final_features = final_features.reshape(-1, id_nunique, final_features.shape[-1])
        indexlist = [[131, -1], [46, 131], [45, 47], [44, 47], [44, 40], [47, 42], [46, 44], [47, 135], [89, 40], [2, 46], [42, 127], [9, 45], [42, 5], [46, 33], [42, 29], [8, 44], [132, 47], [102, 40], [44, 37], [40, 32], [47, 1], [44, 113], [55, 42], [19, 45], [122, 42], [44, 126], [36, 45], [103, 47], [65, 47], [7, 44], [47, 61], [47, 82], [47, 77], [40, 75], [0, 42], [133, 40], [46, 26], [45, 128], [106, 42], [10, 44], [45, 86], [47, 119], [52, 44], [35, 44], [40, 24], [45, 23], [68, 46], [66, 45], [45, 76], [46, 117], [112, 40], [40, 53], [60, 40], [72, 44], [120, 42], [47, 98], [30, 47], [42, 91], [71, 40], [40, 134], [44, 22], [44, 57], [40, 85], [63, 40], [6, 44], [104, 47], [107, 42], [40, 84], [15, 40], [45, 28], [31, 40], [14, 40], [44, 118], [45, 108], [34, 45], [47, 139], [47, 12], [46, 123], [62, 42], [47, 67], [40, 48], [21, 40], [40, 138], [97, 46], [25, 47], [40, 64], [44, 11], [13, 47], [44, 99], [44, 20], [95, 44], [100, 44], [42, 69], [44, 140], [73, 44], [44, 121], [74, 46], [46, 136], [45, 141], [40, 93], [109, 44], [42, 4], [47, 88], [116, 45], [137, 40], [87, 45], [134, -1], [47, 134], [47, 46], [44, 40], [42, 44], [40, 47], [40, 45], [40, 0], [42, 28], [35, 40], [47, 32], [45, 34], [10, 47], [45, 5], [44, 1], [72, 40], [73, 40], [42, 141], [47, 53], [20, 45], [44, 136], [47, 11], [42, 75], [36, 46], [42, 55], [47, 127], [76, 47], [46, 22], [91, 47], [98, 46], [47, 120], [44, 69], [15, 40], [67, 47], [42, 61], [46, 24], [47, 104], [47, 117], [46, 85], [33, 40], [140, 46], [42, 52], [89, 42], [42, 93], [44, 137], [45, 123], [71, 46], [13, 40], [103, 40], [7, 44], [45, 108], [119, 45], [46, 112], [116, 42], [44, 68], [46, 4], [23, 46], [46, 122], [44, 133], [45, 121], [87, 47], [47, 132], [88, 46], [45, 30], [46, 25], [42, 12], [47, 118], [138, 46], [63, 46], [42, 48], [44, 102], [26, 44], [8, 47], [42, 113], [65, 42], [64, 44], [2, 47], [40, 100], [40, 131], [99, 40], [139, 47], [6, 42], [31, 46], [19, 45], [44, 128], [74, 44], [97, 45], [46, 62], [42, 57], [86, 44], [9, 44], [42, 37], [44, 21], [45, 95], [29, 45], [109, 44], [47, 135], [47, 60], [42, 66], [82, 46], [106, 46], [40, 107], [84, 45], [46, 77], [126, 42], [14, 46], [37, -1], [46, 37], [45, 40], [46, 40], [44, 42], [40, 44], [47, 46], [47, 26], [44, 57], [33, 42], [40, 35], [45, 29], [104, 44], [141, 40], [44, 76], [9, 46], [47, 21], [45, 19], [44, 52], [46, 32], [40, 116], [73, 47], [47, 24], [42, 121], [46, 8], [47, 137], [71, 40], [4, 42], [25, 46], [45, 138], [47, 14], [46, 132], [67, 46], [20, 42], [109, 44], [42, 64], [44, 99], [47, 123], [134, 45], [45, 15], [40, 63], [44, 139], [5, 42], [100, 45], [48, 42], [45, 61], [68, 46], [45, 74], [42, 28], [42, 77], [91, 42], [45, 103], [107, 45], [7, 45], [119, 46], [44, 62], [46, 23], [42, 72], [40, 120], [108, 45], [46, 117], [47, 87], [45, 133], [42, 6], [40, 86], [2, 46], [97, 47], [46, 126], [40, 10], [40, 22], [66, 44], [34, 47], [36, 40], [89, 42], [140, 40], [46, 112], [40, 55], [47, 82], [95, 42], [44, 69], [45, 135], [47, 93], [128, 45], [45, 1], [45, 60], [53, 47], [47, 88], [44, 13], [45, 84], [40, 11], [42, 122], [40, 131], [75, 47], [65, 45], [46, 30], [85, 46], [12, 42], [46, 127], [0, 42], [118, 47], [40, 31], [98, 44], [47, 113], [44, 102], [106, 46], [42, 136], [23, -1], [44, 23], [40, 47], [44, 47], [44, 46], [46, 45], [44, 42], [46, 52], [73, 40], [47, 76], [42, 75], [42, 62], [45, 36], [42, 55], [131, 47], [44, 109], [77, 47], [121, 45], [9, 44], [7, 45], [40, 97], [107, 46], [42, 87], [42, 134], [46, 95], [67, 47], [65, 47], [6, 46], [48, 40], [45, 57], [34, 40], [44, 133], [47, 89], [12, 40], [104, 45], [106, 44], [44, 140], [47, 53], [40, 141], [47, 84], [45, 127], [47, 91], [44, 132], [46, 119], [100, 40], [4, 42], [122, 42], [47, 66], [72, 44], [21, 46], [19, 42], [45, 137], [42, 118], [29, 42], [1, 42], [112, 45], [44, 14], [42, 8], [120, 46], [139, 45], [40, 99], [68, 44], [116, 44], [28, 46], [40, 0], [5, 45], [45, 117], [42, 71], [46, 2], [63, 42], [47, 126], [128, 45], [40, 136], [44, 60], [11, 45], [86, 44], [102, 45], [45, 113], [44, 103], [93, 45], [135, 40], [45, 85], [25, 45], [47, 69], [13, 42], [123, 40], [40, 138], [37, 44], [31, 46], [32, 42], [44, 33], [46, 35], [15, 44], [44, 64], [98, 46], [20, 40], [40, 88], [30, 46], [45, 82], [42, 10], [24, 47], [26, 44], [40, 108], [74, 44], [42, 22], [45, 61]]
        final_features_list = np.zeros((final_features.shape[0] * len(indexlist), final_features.shape[-1] * 2), dtype='float32')
        c = 0
        from tqdm.auto import tqdm
        for i in tqdm(range(final_features.shape[0])):
            for item in indexlist:
                final_features_list[c, :final_features.shape[-1]] = final_features[i, item[0]] 
                final_features_list[c, final_features.shape[-1]:] = final_features[i, item[1]] * (item[1]!= -1)
                c += 1

        x_train = final_features_list[:-N_valid * len(indexlist)]
        x_test = final_features_list[-N_valid * len(indexlist):]

        # model
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=2,
            # boosting_type='Plain',
            # loss_function='MultiRMSE',  # 多输出回归使用MultiRMSE损失
            verbose=100,
            random_state=42,
            task_type='GPU',
            reg_lambda=200
        )
        
        # 训练模型
        model.fit(
            x_train, y_train,
            eval_set=(x_test, y_test ),
            early_stopping_rounds=200
        )
        self.models.append(model)

        # 模型推理
        predictions = self.predict(df_features)        
        pred = df_labels.tail(N_valid).reset_index().copy()
        pred.iloc[:,1:] = predictions[-N_valid:] + np.random.random(predictions[-N_valid:].shape) * 1e-10
        N_test = 500
        print('score:', score(df_labels.tail(N_valid).reset_index(), pred.tail(N_valid), 'date_id'))

    def predict(self, df_features):
        final_features = self.preprocess(df_features)
        id_nunique = df_features.id.nunique()
        final_features = final_features.reshape(-1, id_nunique, final_features.shape[-1])
        indexlist = [[131, -1], [46, 131], [45, 47], [44, 47], [44, 40], [47, 42], [46, 44], [47, 135], [89, 40], [2, 46], [42, 127], [9, 45], [42, 5], [46, 33], [42, 29], [8, 44], [132, 47], [102, 40], [44, 37], [40, 32], [47, 1], [44, 113], [55, 42], [19, 45], [122, 42], [44, 126], [36, 45], [103, 47], [65, 47], [7, 44], [47, 61], [47, 82], [47, 77], [40, 75], [0, 42], [133, 40], [46, 26], [45, 128], [106, 42], [10, 44], [45, 86], [47, 119], [52, 44], [35, 44], [40, 24], [45, 23], [68, 46], [66, 45], [45, 76], [46, 117], [112, 40], [40, 53], [60, 40], [72, 44], [120, 42], [47, 98], [30, 47], [42, 91], [71, 40], [40, 134], [44, 22], [44, 57], [40, 85], [63, 40], [6, 44], [104, 47], [107, 42], [40, 84], [15, 40], [45, 28], [31, 40], [14, 40], [44, 118], [45, 108], [34, 45], [47, 139], [47, 12], [46, 123], [62, 42], [47, 67], [40, 48], [21, 40], [40, 138], [97, 46], [25, 47], [40, 64], [44, 11], [13, 47], [44, 99], [44, 20], [95, 44], [100, 44], [42, 69], [44, 140], [73, 44], [44, 121], [74, 46], [46, 136], [45, 141], [40, 93], [109, 44], [42, 4], [47, 88], [116, 45], [137, 40], [87, 45], [134, -1], [47, 134], [47, 46], [44, 40], [42, 44], [40, 47], [40, 45], [40, 0], [42, 28], [35, 40], [47, 32], [45, 34], [10, 47], [45, 5], [44, 1], [72, 40], [73, 40], [42, 141], [47, 53], [20, 45], [44, 136], [47, 11], [42, 75], [36, 46], [42, 55], [47, 127], [76, 47], [46, 22], [91, 47], [98, 46], [47, 120], [44, 69], [15, 40], [67, 47], [42, 61], [46, 24], [47, 104], [47, 117], [46, 85], [33, 40], [140, 46], [42, 52], [89, 42], [42, 93], [44, 137], [45, 123], [71, 46], [13, 40], [103, 40], [7, 44], [45, 108], [119, 45], [46, 112], [116, 42], [44, 68], [46, 4], [23, 46], [46, 122], [44, 133], [45, 121], [87, 47], [47, 132], [88, 46], [45, 30], [46, 25], [42, 12], [47, 118], [138, 46], [63, 46], [42, 48], [44, 102], [26, 44], [8, 47], [42, 113], [65, 42], [64, 44], [2, 47], [40, 100], [40, 131], [99, 40], [139, 47], [6, 42], [31, 46], [19, 45], [44, 128], [74, 44], [97, 45], [46, 62], [42, 57], [86, 44], [9, 44], [42, 37], [44, 21], [45, 95], [29, 45], [109, 44], [47, 135], [47, 60], [42, 66], [82, 46], [106, 46], [40, 107], [84, 45], [46, 77], [126, 42], [14, 46], [37, -1], [46, 37], [45, 40], [46, 40], [44, 42], [40, 44], [47, 46], [47, 26], [44, 57], [33, 42], [40, 35], [45, 29], [104, 44], [141, 40], [44, 76], [9, 46], [47, 21], [45, 19], [44, 52], [46, 32], [40, 116], [73, 47], [47, 24], [42, 121], [46, 8], [47, 137], [71, 40], [4, 42], [25, 46], [45, 138], [47, 14], [46, 132], [67, 46], [20, 42], [109, 44], [42, 64], [44, 99], [47, 123], [134, 45], [45, 15], [40, 63], [44, 139], [5, 42], [100, 45], [48, 42], [45, 61], [68, 46], [45, 74], [42, 28], [42, 77], [91, 42], [45, 103], [107, 45], [7, 45], [119, 46], [44, 62], [46, 23], [42, 72], [40, 120], [108, 45], [46, 117], [47, 87], [45, 133], [42, 6], [40, 86], [2, 46], [97, 47], [46, 126], [40, 10], [40, 22], [66, 44], [34, 47], [36, 40], [89, 42], [140, 40], [46, 112], [40, 55], [47, 82], [95, 42], [44, 69], [45, 135], [47, 93], [128, 45], [45, 1], [45, 60], [53, 47], [47, 88], [44, 13], [45, 84], [40, 11], [42, 122], [40, 131], [75, 47], [65, 45], [46, 30], [85, 46], [12, 42], [46, 127], [0, 42], [118, 47], [40, 31], [98, 44], [47, 113], [44, 102], [106, 46], [42, 136], [23, -1], [44, 23], [40, 47], [44, 47], [44, 46], [46, 45], [44, 42], [46, 52], [73, 40], [47, 76], [42, 75], [42, 62], [45, 36], [42, 55], [131, 47], [44, 109], [77, 47], [121, 45], [9, 44], [7, 45], [40, 97], [107, 46], [42, 87], [42, 134], [46, 95], [67, 47], [65, 47], [6, 46], [48, 40], [45, 57], [34, 40], [44, 133], [47, 89], [12, 40], [104, 45], [106, 44], [44, 140], [47, 53], [40, 141], [47, 84], [45, 127], [47, 91], [44, 132], [46, 119], [100, 40], [4, 42], [122, 42], [47, 66], [72, 44], [21, 46], [19, 42], [45, 137], [42, 118], [29, 42], [1, 42], [112, 45], [44, 14], [42, 8], [120, 46], [139, 45], [40, 99], [68, 44], [116, 44], [28, 46], [40, 0], [5, 45], [45, 117], [42, 71], [46, 2], [63, 42], [47, 126], [128, 45], [40, 136], [44, 60], [11, 45], [86, 44], [102, 45], [45, 113], [44, 103], [93, 45], [135, 40], [45, 85], [25, 45], [47, 69], [13, 42], [123, 40], [40, 138], [37, 44], [31, 46], [32, 42], [44, 33], [46, 35], [15, 44], [44, 64], [98, 46], [20, 40], [40, 88], [30, 46], [45, 82], [42, 10], [24, 47], [26, 44], [40, 108], [74, 44], [42, 22], [45, 61]]

        final_features_list = np.zeros((final_features.shape[0] * len(indexlist), final_features.shape[-1] * 2), dtype='float32')
        c = 0
        from tqdm.auto import tqdm
        for i in tqdm(range(final_features.shape[0])):
            for item in indexlist:
                final_features_list[c, :final_features.shape[-1]] = final_features[i, item[0]] 
                final_features_list[c, final_features.shape[-1]:] = final_features[i, item[1]] * (item[1]!= -1)
            
                c += 1
        
        id_nunique = df_features.id.nunique()

        predictions = [model.predict(final_features_list) for model in self.models]
        predictions = np.mean(predictions, 0)
        predictions = predictions.reshape(-1, 424)
        
        return predictions




# 测试代码 main 函数
def main():
    # ------------------------------
    # 数据配置
    # ------------------------------
    config = CFG()
    # ------------------------------
    # 数据读取
    # ------------------------------
    df = pd.read_csv(f'{config.input_path}/train.csv')
    print(df)
    df_pairs = pd.read_csv(f'{config.input_path}/target_pairs.csv')
    print(df_pairs)
    # ------------------------------
    # 特征工程
    # ------------------------------
    df_processed = preprocess(df)
    print(df_processed)
    df_features = create_features(df_processed)
    print(df_features)
    print(df_features.columns)
    """
    # ------------------------------
    # 
    # ------------------------------
    N_valid = 134
    model1 = Modelmultilabel(config, None)
    model1.train(df_features, N_valid)
    # ------------------------------
    # 
    # ------------------------------
    N_valid = 134
    model2 = Modelmultilabel_v2(config, None)
    model2.train(df_features, N_valid)
    # ------------------------------
    # 
    # ------------------------------
    N_valid = 134
    model3 = Modelmultilabel_v3(config, None)
    model3.train(df_features, N_valid)
    # ------------------------------
    # 
    # ------------------------------
    prediction1 = model1.predict(df_features).reshape(-1, 424)
    prediction2 = model2.predict(df_features).reshape(-1, 424)
    prediction3 = model3.predict(df_features).reshape(-1, 424)

    pred_ensemble = prediction1 + prediction2 + prediction3 * 3
    """

if __name__ == "__main__":
    main()
