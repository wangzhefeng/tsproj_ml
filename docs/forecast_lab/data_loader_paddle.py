# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader_paddle.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-01-05
# * Version     : 0.1.010517
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
import glob
import pickle
import re
from loguru import logger

from pandas.tseries import to_offset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import paddle

from timestamp_utils import to_unix_time

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TSDataset(paddle.io.Dataset):
    """
    时序 DataSet
    划分数据集、适配 dataloader 所需的 dataset 格式
    ref: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
    proj: https://aistudio.baidu.com/aistudio/projectdetail/5911966
    """
    def __init__(self, 
                 data, 
                 ts_col = 'DATATIME',
                 use_cols = [
                    'WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 
                    'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 
                    'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15',
                    'month', 'day', 'weekday', 'hour', 'minute'
                 ],
                 labels = ['ROUND(A.POWER,0)', 'YD15'], 
                 input_len = 24*4*5, 
                 pred_len = 24*4, 
                 stride = 19*4, 
                 data_type = 'train',
                 train_ratio = 0.7, 
                 val_ratio = 0.15):
        super(TSDataset, self).__init__()
        self.ts_col = ts_col  # 时间戳列
        self.use_cols = use_cols  # 训练时使用的特征列
        self.labels = labels  # 待预测的标签列
        self.input_len = input_len  # 模型输入数据的样本点长度，15分钟间隔，一个小时14个点，近5天的数据就是24*4*5
        self.pred_len = pred_len  # 预测长度，预测次日00:00至23:45实际功率，即1天：24*4
        self.data_type = data_type  # 需要加载的数据类型
        self.scale = True  # 是否需要标准化
        self.train_ratio = train_ratio  # 训练集划分比例
        self.val_ratio = val_ratio  # 验证集划分比例
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率，所以x和label要间隔19*4个点
        self.stride = stride
        assert data_type in ['train', 'val', 'test']  # 确保data_type输入符合要求
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.data_type]
        # data transform
        self.transform(data)

    def transform(self, df):
        # 获取 unix 时间戳、输入特征和预测标签
        time_stamps = df[self.ts_col].apply(lambda x:to_unix_time(x)).values
        x_values = df[self.use_cols].values
        y_values = df[self.labels].values
        # 划分数据集
        num_train = int(len(df) * self.train_ratio)
        num_vali = int(len(df) * self.val_ratio)
        num_test = len(df) - num_train - num_vali
        border1s = [0, num_train - self.input_len - self.stride, len(df) - num_test - self.input_len - self.stride]
        border2s = [num_train, num_train + num_vali, len(df)]
        # 获取 data_type 下的左右数据截取边界
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]    
        # 标准化
        self.scaler = StandardScaler()
        if self.scale:
            # 使用训练集得到 scaler 对象
            train_data = x_values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(x_values)
            # 保存 scaler
            pickle.dump(self.scaler, open('/home/aistudio/submission/model/scaler.pkl', 'wb'))
        else:
            data = x_values
        # array to paddle tensor
        self.time_stamps = paddle.to_tensor(time_stamps[border1:border2], dtype = 'int64')
        self.data_x = paddle.to_tensor(data[border1:border2], dtype = 'float32')
        self.data_y = paddle.to_tensor(y_values[border1:border2], dtype = 'float32')  

    def __getitem__(self, index):
        """
        实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率，所以x和label要间隔19*4个点
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end + self.stride
        r_end = r_begin + self.pred_len
        # TODO 可以增加对未来可见数据的获取
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        ts_x = self.time_stamps[s_begin:s_end]
        ts_y = self.time_stamps[r_begin:r_end]
        return seq_x, seq_y, ts_x, ts_y

    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.data_x) - self.input_len - self.stride - self.pred_len  + 1


class TSPredDataset(paddle.io.Dataset):
    """
    时序 Pred DataSet
    划分数据集、适配dataloader所需的dataset格式
    ref: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
    proj: https://aistudio.baidu.com/aistudio/projectdetail/5911966
    """
    def __init__(self, 
                 data, 
                 ts_col = 'DATATIME',
                 use_cols =[
                     'WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 
                     'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 
                     'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15',
                     'month', 'day', 'weekday', 'hour', 'minute'
                 ],
                 labels = ['ROUND(A.POWER,0)', 'YD15'],  
                 input_len = 24*4*5, 
                 pred_len = 24*4, 
                 stride = 19*4):
        super(TSPredDataset, self).__init__()
        self.ts_col = ts_col  # 时间戳列
        self.use_cols = use_cols  # 训练时使用的特征列
        self.labels = labels  # 待预测的标签列
        self.input_len = input_len  # 模型输入数据的样本点长度，15分钟间隔，一个小时14个点，近5天的数据就是24*4*5
        self.pred_len = pred_len  # 预测长度，预测次日00:00至23:45实际功率，即1天：24*4
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率，所以x和label要间隔19*4个点
        self.stride = stride        
        self.scale = True  # 是否需要标准化
        # data transform
        self.transform(data)

    def transform(self, df):
        # 获取 unix 时间戳、输入特征和预测标签
        time_stamps = df[self.ts_col].apply(lambda x:to_unix_time(x)).values
        x_values = df[self.use_cols].values
        y_values = df[self.labels].values
        # 截取边界
        border1 = len(df) - self.input_len - self.stride - self.pred_len
        border2 = len(df)   
        # 标准化
        self.scaler = StandardScaler()
        if self.scale:
            # 读取预训练好的 scaler
            self.scaler = pickle.load(open('/home/aistudio/submission/model/scaler.pkl', 'rb'))
            data = self.scaler.transform(x_values)
        else:
            data = x_values
        # array to paddle tensor
        self.time_stamps = paddle.to_tensor(time_stamps[border1:border2], dtype = 'int64')
        self.data_x = paddle.to_tensor(data[border1:border2], dtype = 'float32')
        self.data_y = paddle.to_tensor(y_values[border1:border2], dtype = 'float32')  

    def __getitem__(self, index):
        """
        实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率，所以x和label要间隔19*4个点
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end + self.stride
        r_end = r_begin + self.pred_len
        # TODO 可以增加对未来可见数据的获取
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        ts_x = self.time_stamps[s_begin:s_end]
        ts_y = self.time_stamps[r_begin:r_end]
        return seq_x, seq_y, ts_x, ts_y

    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.data_x) - self.input_len - self.stride - self.pred_len  + 1


def data_preprocess(df):
    """
    数据预处理
    1. 数据排序
    2. 去除重复值
    3. 重采样（ 可选）
    4. 缺失值处理
    5. 异常值处理
    proj: https://aistudio.baidu.com/aistudio/projectdetail/5911966
    """
    # 排序
    df = df.sort_values(by = "DATATIME", ascending = True)
    logger.info(f"df.shape: {df.shape}")
    logger.info(f"Time range from {df['DATATIME'].values[0]} to {df['DATATIME'].values[-1]}")
    # 去除重复值
    df = df.drop_duplicates(subset = "DATATIME", keep = "first")
    logger.info(f"After dropping dulicates: {df.shape}")
    # 重采样（可选）+ 缺失值处(理线性插值)：比如 04 风机缺少 2022-04-10 和 2022-07-25 两天的数据，重采样会把这两天数据补充进来
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.set_index("DATATIME")
    df = df.resample(rule = to_offset('15T').freqstr, label = 'right', closed = 'right')
    df = df.interpolate(method = 'linear', limit_direction = 'both').reset_index()
    # 异常值处理
    # 当实际风速为 0 时，功率设置为 0
    df.loc[df["ROUND(A.WS,1)"] == 0, "YD15"] = 0
    # TODO 风速过大但功率为 0 的异常：先设计函数拟合出：实际功率=f(风速)，然后代入异常功率的风速获取理想功率，替换原异常功率
    # TODO 对于在特定风速下的离群功率（同时刻用 IQR 检测出来），做功率修正（如均值修正）

    return df




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
