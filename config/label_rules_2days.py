import pandas as pd
import os


class Positive2Days:
    def __init__(self):
        self.numbers_of_prediction = 2
        self.model_file_name = 'cnn_net_2day_positive.pth'

    def create_label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 初始化label
        label_list = [0.0, 0.0]
        closed_price_list = df['close'].tolist()
        # 当天股价大于明天,且升幅大于1%，则标记为1
        if closed_price_list[1] >= closed_price_list[0] * 1.01:
            label_list[0] = 1.0
        # 后天收盘价不比明天小，则标记为1
        if closed_price_list[2] >= closed_price_list[1]:
            label_list[1] = 1.0
        label_df = pd.DataFrame(label_list)
        return label_df

'''
class Negative10Days:
    def __init__(self):
        self.numbers_of_prediction = 10  # 2 weeks
        self.model_file_name = 'cnn_net_10day_negative.pth'

    def create_label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 初始化label
        label_list = [0.0, 0.0]
        closed_price_list = df['close'].tolist()
        # 当天股价大于明天,且升幅大于1%，则标记为1
        if closed_price_list[1] < closed_price_list[0]:  # 明天低于昨天
            label_list[0] = 1.0
        # 后天收盘价不比明天小，则标记为1
        if closed_price_list[10] < closed_price_list[1]:  # 10天后也小于明天
            label_list[1] = 1.0
        label_df = pd.DataFrame(label_list)
        return label_df
'''

class Negative2Days:
    def __init__(self):
        self.numbers_of_prediction = 2  # 2 day
        self.model_file_name = 'cnn_net_2day_negative.pth'

    def create_label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 初始化label
        label_list = [0.0, 0.0]
        closed_price_list = df['close'].tolist()
        # 当天股价大于明天,且升幅大于1%，则标记为1
        if closed_price_list[1] < closed_price_list[0]:  # 明天低于昨天
            label_list[0] = 1.0
        # 后天收盘价不比明天小，则标记为1
        if closed_price_list[2] < closed_price_list[1]:  # 10天后也小于明天
            label_list[1] = 1.0
        label_df = pd.DataFrame(label_list)
        return label_df
