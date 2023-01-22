import os

import torch

from model import train_config as trainer_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import tensor, from_numpy
from torch.utils.data import Dataset


class CSVIterator:
    def __init__(self) -> None:
        self.feature_files_list = os.listdir(trainer_config.feature_data_directory_path)
        self.label_files_list = os.listdir(trainer_config.label_data_directory_path)
        self.feature_files_path = []
        self.label_files_path = []
        for f in self.feature_files_list:
            if 'csv' in f:
                self.feature_files_path.append(os.path.join(trainer_config.feature_data_directory_path, f))
                # label 与 feature文件同名
                self.label_files_path.append(os.path.join(trainer_config.label_data_directory_path, f))

    def __len__(self):
        return len(self.feature_files_path)

    def __getitem__(self, index):
        print(self.feature_files_path[index])
        feature_data = pd.read_csv(self.feature_files_path[index], header=0)
        label_data = pd.read_csv(self.label_files_path[index], header=0)
        code = "'" + self.feature_files_path[index][-10:-4]
        return code, feature_data, label_data  # 特征为列向量，沿行方向为时间轴


class DataFormer(Dataset):
    def __init__(self, stream_length: int):
        self.__stream_length = stream_length  # 一个样本的图片层数。可理解为一个单元的视频祯中包含多少张图片
        self.__feature_stream, self.__label_stream = self.__set_data()

    def __scale_and_to_tensor(self, np_data: np.array) -> np.array:
        scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化
        scaler = scaler.fit(np_data)  # fit，在这里本质是生成min(x)和max(x)
        scaled_np_data = scaler.transform(np_data)  # 通过接口导出结果
        return scaled_np_data

    def __pic_to_flat_stream(self, np_data: np.array) -> np.array:  # 按日生成前N日内的"视频"流
        num_of_vector = np_data.shape[0]
        flat_stream = np_data[0:self.__stream_length]
        circles = num_of_vector - self.__stream_length
        for i in range(circles):
            if i > 0:
                flat_stream = np.append(flat_stream, np_data[i:i+self.__stream_length], axis=0)
        return flat_stream

    def __set_data(self) -> (tensor, tensor):
        csv_iterator = CSVIterator()
        first_flag = True
        for _, feature_data, label_data in csv_iterator:
            # 可能刚好出现feature数据少甚至数据为空的feature csv，需要作出判断和筛选
            # 如果数据长度小于视频流长度，则放弃该只share数据
            if len(feature_data) < self.__stream_length + 1:
                continue
            # -------每一csv下的数据--------
            if first_flag:  # 为了识别第一次运算。便于后续结果直接赋值给np_feature_flat_stream和np_label_stream
                first_flag = False
                # 取得归一化数据：np,此处df数据的列为特征，每一列为一个特征
                scaled_np_feature_data = self.__scale_and_to_tensor(feature_data.values)
                # scaled_np_label_data = self.__scale(label_data.values)  # 因目标数据已经为0或1，不再进行归一化转换
                # 转换为时序"视频流"数据 （将每日数据转成平面矩阵，再将指定数量的上述平面向第3维方向堆叠，即动画原理）
                np_feature_flat_stream = self.__pic_to_flat_stream(scaled_np_feature_data)
                np_label_flat_stream = self.__pic_to_flat_stream(label_data)
            else:  # 以下语句的注释同上
                scaled_np_feature_data = self.__scale_and_to_tensor(feature_data.values)
                np_feature_flat_stream = np.append(np_feature_flat_stream,
                                                   self.__pic_to_flat_stream(scaled_np_feature_data), axis=0)
                np_label_flat_stream = np.append(np_label_flat_stream, self.__pic_to_flat_stream(label_data), axis=0)
        feature_stream = from_numpy(np_feature_flat_stream)
        label_stream = from_numpy(np_label_flat_stream)
        feature_stream = feature_stream.view(-1, self.__stream_length, 5, 5)  # 转换成若干份3d流
        feature_stream = torch.unsqueeze(feature_stream, 1).to(torch.float32)  # 按conv3d输入格式，在3维前填充 channel为1
        label_stream = label_stream.view(-1, self.__stream_length, 1, 2).to(torch.float32)  # 转换成若干份3d流
        return feature_stream, label_stream

    def __len__(self):
        return len(self.__feature_stream)

    def __getitem__(self, index):
        # 对data期望是，每次返回feature和label的steam一组，即deep个5x5的pic（每pic是单日特征值）
        data = (self.__feature_stream[index], self.__label_stream[index])
        return data



