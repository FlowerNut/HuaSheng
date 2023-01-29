from model import predict_config as app_config
import torch
from collector import share_history_iterator
import pandas as pd
from scipy import fft
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import predict_config
import os


class PredictByCNN:
    def __init__(self, label_rule, predict_result_file_path):
        self.label_rule = label_rule
        self.predict_result_file_path = predict_result_file_path
        # ----------初始化存贮应用数据文件夹---------
        app_config.build_dir(app_config.application_data_directory_path)
        # --------以下数据需与训练模型/fft生成参数一致-------
        self.__stream_length = 10
        self.taken_fft_channel_numbers = 8  # 分解成8个周期波型
        self.data_buffer_days = 618  # 数据决定参与计算fft的日数据量，与cnn的计算深度stream len可以为不同长度
        self.numbers_of_prediction = self.label_rule.numbers_of_prediction  # 预测明天，后天，共两天； self.__creating_label_df的标签计算需要根据该值调整
        # -------------加载训练模型----------------
        model_file_path = os.path.join(predict_config.prediction_model_directory_path, self.label_rule.model_file_name)
        self.cnn = torch.load(model_file_path)
        # -------------预测数据-------------------
        self.__predict()

    def __single_clean_share_clean_data_to_feature(self, clean_share_df: pd.DataFrame) -> (bool, pd.DataFrame):
        # 指示share数据长度是否足够，如为false，表示数据量不足，返回空的feature data
        feature_available = False
        # 创建空df用于存贮结果
        feature_df = pd.DataFrame()
        if clean_share_df.shape[0] - self.data_buffer_days > 10:   # 数据有g一定余量再参与运算
            feature_available = True
            # 样本量要大于self.data_buff_days，有足够数据计算一次fft，单只股票数据才参与计算
            last_row_index = clean_share_df.__len__() - 1  # 索引第1个为0，最后一位索引应为长度-1
            since_row_index = last_row_index - self.__stream_length + 1  # 因需要取stream len日数据作为一组cnn输入，该数据计算截取开始索引
            # 提取需要的数据列
            clean_share_df = clean_share_df[['open', 'high', 'low','close', 'pre_close', 'change',
                                             'pct_chg', 'vol', 'amount']]
            for current_row_index in range(since_row_index, last_row_index + 1):   # +1为了尾索引可以包括 last_row_index所在数据
                if current_row_index > self.data_buffer_days - 1:
                    # 取当日数据
                    current_day_df = clean_share_df.iloc[current_row_index, :].copy()  # df为引用型数据，防止原数据被修改影响下一循环结果
                    # 过程变量，历史转换为列向量df
                    pic_df = pd.DataFrame(current_day_df.tolist())
                    # 获取fft周期数据
                    # 根据clean数据，计算fft;其中输入数据取stream len长度个日期数据，计算fft。其中每日数据计算fft自索引0的clean数据开始，确保数据量足够以达到最大精度
                    # 其中current_row_index + 1是因为索引后一位只取到current_row_index前一位，需要加一，确保fft计算到当日
                    fft_composed_periods_prices_df = self.__closed_price_fft_into_multiple_periods_prices_df \
                        (clean_share_df.iloc[0:current_row_index+1, :].copy(), self.taken_fft_channel_numbers)
                    # 只取当日+1天的fft结果（获得数据当天已收盘，取下一天数据作预测），以及当日+self.numbers_of_prediction天后fft结果
                    # 提取fft的当日后一日和self.numbers_of_prediction后一日数据(注：如用其它数据，需要注意周六日无数据，一周为5天）
                    section_fft_composed_periods_prices_df = pd.concat([fft_composed_periods_prices_df.iloc[current_row_index + 1, :]
                            , fft_composed_periods_prices_df.iloc[current_row_index + self.numbers_of_prediction, :]],
                        axis=1, ignore_index=True)  # 指示具体索引而非范围尾数，需要指针-1
                    # 周期数据按列遍历，首尾相接， 将周期数据合并至pic_df列向量（作为图片识别）
                    pic_df = pd.concat([pic_df, self.__reshape_df_into_one_row(section_fft_composed_periods_prices_df.T)],
                                       ignore_index=True, sort=False)  # .T转为按频率的列向量，保持同一频率数值相近
                    # 保存至feature_df
                    feature_df = pd.concat([feature_df, pic_df], axis=1, ignore_index=True)
            # 转置，行为样本数（日期），列为特征
            feature_df = feature_df.T
            # 更改列名，防止后续读取时列名为纯数字带来的问题
            feature_df.columns = ["F{0}".format(x) for x in range(feature_df.shape[1])]
        return feature_available, feature_df

    def __closed_price_fft_into_multiple_periods_prices_df(self,share_history_df:pd.DataFrame,taken_fft_channel_numbers:int)->pd.DataFrame:
        price_list = share_history_df['close'].to_numpy()
        column_names = ["F{0}".format(i) for i in range(taken_fft_channel_numbers)]
        fft_df = pd.DataFrame(columns=column_names) # 结果容器
        # 将价格移至平均值，变为ac信号
        price_list -= np.average(price_list)
        # 生成价格的fft
        fft_y = fft.rfft(price_list)
        #fft_x=fft.fftfreq(len(price_list),1/len(price_list))
        # 从fft_y中，寻找｜绝对值｜较大的频率的List index
        fft_y_value_index_descending = np.argsort(np.abs(fft_y))[::-1][0:taken_fft_channel_numbers]
        # 按从大到小的fft_y，逆变生成对应波形
        for i,v in enumerate(fft_y_value_index_descending):
            y = np.zeros(len(fft_y),dtype=complex) # 与fft_y同长度的全0 array
            y[v] = fft_y[v] #按从大到小fft_y替换，用于提取波形
            ifft_data_length = len(price_list) + self.numbers_of_prediction
            irfft_y = fft.irfft(y,ifft_data_length)
            column_name = "F{0}".format(i)
            fft_df[column_name] = irfft_y
        return fft_df

    def __reshape_df_into_one_row(self,df:pd.DataFrame)->pd.DataFrame: #输出为多行单列的“向量” df
        column_count = df.shape[1]
        data = []
        for i in range(column_count):
            data.append(df.iloc[:,i].tolist())
        # 存入pic_df
        flatten_data = [y for x in data for y in x]
        pic_df = pd.DataFrame(flatten_data)
        return pic_df

    def __feature_df_to_stream(self, feature_df: pd.DataFrame):
        # 数据归一化，用于输入模型
        scaled_np_feature_data = self.__scale_and_to_tensor(feature_df.values)
        # 转换为时序"视频流"数据 （将每日数据转成平面矩阵，再将指定数量的上述平面向第3维方向堆叠，即动画原理）
        np_feature_flat_stream = self.__pic_to_flat_stream(scaled_np_feature_data)
        # 转换为tensor
        feature_stream = torch.from_numpy(np_feature_flat_stream)
        # 转换为模型输入格式
        feature_stream = feature_stream.view(-1, self.__stream_length, 5, 5)  # 转换成若干份3d流
        feature_stream = torch.unsqueeze(feature_stream, 1).to(torch.float32)  # 按conv3d输入格式，在3维前填充 channel为1
        return feature_stream

    def __scale_and_to_tensor(self, np_data: np.array) -> np.array:
        scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化
        scaler = scaler.fit(np_data)  # fit，在这里本质是生成min(x)和max(x)
        scaled_np_data = scaler.transform(np_data)  # 通过接口导出结果
        return scaled_np_data

    def __pic_to_flat_stream(self, np_data: np.array) -> np.array: # 按日生成前N日内的"视频"流
        flat_stream = np_data[0:self.__stream_length]
        num_of_vector = np_data.shape[0]
        circles = num_of_vector - self.__stream_length
        for i in range(circles):
            if i > 0:
                flat_stream = np.append(flat_stream, np_data[i:i+self.__stream_length], axis=0)
        return flat_stream

    def __predict(self):
        # 输入数据（clean share）更新，由外部程序完成
        predict_pd_table = pd.DataFrame(columns=['code', 'date', 'value1', 'value2'])
        # 遍历clean share
        for code, cleaned_share_df in share_history_iterator.HistoryCSVIterator():
            # 历史数据小于预测要求，路过该股票
            if len(cleaned_share_df) < self.__stream_length:
                continue
            latest_date = cleaned_share_df.tail(1).index.values[0]  # 最后一行的第一个值
        # 按最新日期，生成最新一次fft数据
            feature_available, feature_df = self.__single_clean_share_clean_data_to_feature(cleaned_share_df)
            if feature_available:
                with torch.no_grad():
                    # 输入数据整形为cnn输入格式
                    feature_stream = self.__feature_df_to_stream(feature_df)
                    # 当日数据及fft数据整合成为一次cnn输入
                    prediction = self.cnn(feature_stream)
                    prediction = torch.flatten(prediction, 1)
                    prediction = prediction.view(-1, 1).numpy()
                    prediction_len = len(prediction)
                    single_predict_df = pd.DataFrame([[code, latest_date, prediction[prediction_len - 2][0],
                                                       prediction[prediction_len - 1][0]]],
                                                     columns=['code', 'date', 'value1', 'value2'])
                    print(single_predict_df)
                    predict_pd_table = pd.concat([predict_pd_table, single_predict_df])
        # 按条件过滤、排序再输出
        #predict_pd_table = predict_pd_table.loc[(predict_pd_table['value1'] > 0.618) & (predict_pd_table['value2'] > 0.618)]
        #predict_pd_table = predict_pd_table.sort_values(by='value1', ascending=False)
        predict_pd_table.to_csv(self.predict_result_file_path, index=False)
