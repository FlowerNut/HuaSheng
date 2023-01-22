from scipy import fft
from cleaner import share_history
from cleaner import config as cleaner_config
import numpy as np
import pandas as pd
import os
from model import train_config as trainer_config


class TrainingDataPreparer:
    def __init__(self, label_rule, be_continue=False) -> None:
        self.stream_len = 10
        self.label_rule = label_rule  # 输入类, 用于改变训练数据生成的规则；
        self.be_continue = be_continue  # 用于给出指令：重新生成数据，或续上次完成生成数据；
        self.share_history_class = share_history.ShareHistoryCleaner()
        self.taken_fft_channel_numbers = 8  # 分解成8个周期波型
        self.numbers_of_prediction = self.label_rule.numbers_of_prediction  # 预测明天，后天，共两天； self.__creating_label_df的标签计算需要根据该值调整
        self.data_buffer_days = 618  # 数据决定参与计算fft的日数据量，与cnn的计算深度stream len可以为不同长度
        self.__prepare()

    # 单个股票生成特征数据太大，可能要改结构，每次生成数据不存贮，直接训练 ==> 逐个生成训练数据，不存，直接训练出结果，并存贮模型。
    # 对比历史数据文件中股票代码，和训练过的股票代码。找出未训练的训练。
    def __prepare(self):
        cleaned_share_history_file_list = os.listdir(cleaner_config.cleaned_share_history_directory_path)
        if not self.be_continue:
            # 如果文件夹不存在，创建:用于存贮训练数据文件
            trainer_config.build_or_clear_dir(trainer_config.training_data_directory_path)
            trainer_config.build_or_clear_dir(trainer_config.feature_data_directory_path)
            trainer_config.build_or_clear_dir(trainer_config.label_data_directory_path)
            # trainer_config.build_or_clear_dir(trainer_config.mapminmax_record_data_path)
            # 遍历历史数据文件夹中文件
            for file_name in cleaned_share_history_file_list:
                code = file_name[0:6]
                # 因创业版数据长度有限，影响模型精度，此处跳过3字头的创业版数据，不生成模型数据
                #if code[0] == '3':
                    #continue
                # 如果code不是纯数字，跳过
                if code.isdigit():
                    self.__prepare_single_share_data(file_name)
        else:
            created_training_data_list = os.listdir(trainer_config.feature_data_directory_path)
            todo_file_name_list = []
            for file_name in cleaned_share_history_file_list:
                if file_name in created_training_data_list:
                    continue
                else:
                    todo_file_name_list.append(file_name)
            for file_name in todo_file_name_list:
                code = file_name[0:6]
                # 因创业版数据长度有限，影响模型精度，此处跳过3字头的创业版数据，不生成模型数据
                #if code[0] == '3':
                    #continue
                # 如果code不是纯数字，跳过
                if code.isdigit():
                    self.__prepare_single_share_data(file_name)

            
    def __prepare_single_share_data(self, file_name: str):
        code = file_name[0:6]
        # 读取历史数据
        cleaned_share_history_df = self.share_history_class.read_cleaned_share_history_csv(code, datetime_index=True)
        # 每一股票数据保存一个特征的csv和一个标签的csv
        # 样本量要大于self.data_buff_days，有足够数据计算一次fft，单只股票数据才参与计算
        if cleaned_share_history_df.shape[0] - self.data_buffer_days > self.stream_len:
            # 每一股票:获取数据列的最大值和最小值，并存贮至“auxiliary”文件夹对应的code.csv
            #self.__get_max_min_value_to_csv(code,cleaned_share_history_df)
            # 提取需要的数据列
            cleaned_share_history_df = cleaned_share_history_df[['closed_price', 'highest_price', 'lowest_price',
                                                                 'opened_price', 'change_rate', 'traded_volume',
                                                                 'traded_amount', 'total_share_capital',
                                                                 'flow_share_capital']]
            # 列数据归一化
            #cleaned_share_history_df = cleaned_share_history_df.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)))
            # 创建空df用于存贮结果
            feature_df = pd.DataFrame()
            label_df = pd.DataFrame()
            for current_row_index in range(cleaned_share_history_df.shape[0]-self.numbers_of_prediction-1): # 从self.data_buffer_days开始，至（历史数据长度-self.numbers_of_prediction) (作为索引，需要再-1)
                # 跳过self.data_buffer_days日，以使得fft波形完整
                if current_row_index > self.data_buffer_days:
                    # 开始生成当前股票数据
                    # ---------生成特征数据--------
                    # 取当日数据
                    current_day_df = cleaned_share_history_df.iloc[current_row_index, :].copy() #df为引用型数据，防止原数据被修改影响下一循环结果
                    # 过程变量，历史转换为列向量df
                    pic_df = pd.DataFrame(current_day_df.tolist())
                    # 获取fft周期数据
                    ''' 以下内容按固定长度计算，速度更快，但精度不足
                    # 根据clean数据，计算fft;其中输入数据只取 当前日期至日期前的self.data_buffer_days之间，减少计算量
                    # fft计算输入是一个移动的框，选取自self.data_buff_day天前至当天。而生成的fft则是比上述框多出self.numbers_of_prediction（天）
                    # 此处取的cleaned_share_history_df的行实际为current index - 1，索引取值问题
                    fft_composed_periods_prices_df = self.__closed_price_fft_into_multiple_periods_prices_df\
                        (cleaned_share_history_df.iloc[(current_row_index - self.data_buffer_days-1):current_row_index,:].copy(),self.taken_fft_channel_numbers)
                    # 只取当日+1天的fft结果（获得数据当天已收盘，取下一天数据作预测），以及当日+self.numbers_of_prediction天后fft结果
                    # 因fft生成是固定长度，取值也是固定位置，而非移动,当前日+1 = self.data_buffer_days+1，此处取除当日之外的，以后的两个预测日的fft
                    section_fft_composed_periods_prices_df = pd.concat([fft_composed_periods_prices_df.iloc[self.data_buffer_days+1,:]\
                        ,fft_composed_periods_prices_df.iloc[self.data_buffer_days+self.numbers_of_prediction, :]],
                                                                       axis=1, ignore_index=True) #指示具体索引而非范围尾数，需要指针-1
                    '''
                    # 根据clean数据，计算fft;其中输入数据取自0索引开始到当天，因索引范围后面的数值为当前数据前一位，要取到当天数据需要current_row_index +1
                    fft_composed_periods_prices_df = self.__closed_price_fft_into_multiple_periods_prices_df\
                        (cleaned_share_history_df.iloc[0:current_row_index+1, :].copy(), self.taken_fft_channel_numbers)
                    # 只取当日+1天的fft结果（获得数据当天已收盘，取下一天数据作预测），以及当日+self.numbers_of_prediction天后fft结果
                    # 因fft生成是固定长度，取值也是固定位置，而非移动,当前日+1 = self.data_buffer_days+1，此处取除当日之外的，以后的两个预测日的fft
                    section_fft_composed_periods_prices_df = pd.concat([fft_composed_periods_prices_df.iloc[current_row_index + 1, :],
                                                                        fft_composed_periods_prices_df.iloc[current_row_index + self.numbers_of_prediction, :]],
                                                                       axis=1, ignore_index=True)  # 指示具体索引而非范围尾数，需要指针-1
                    # 周期数据按列遍历，首尾相接， 将周期数据合并至pic_df列向量（作为图片识别）
                    pic_df = pd.concat([pic_df,self.__reshape_df_into_one_row(section_fft_composed_periods_prices_df.T)],
                                       ignore_index=True, sort=False)  # .T转为按频率的列向量，保持同一频率数值相近
                    # 保存至feature_df
                    feature_df = pd.concat([feature_df, pic_df], axis=1, ignore_index=True)
                    # ----------生成标签数据----------
                    # 当天至预测范围内的数据，按行截取
                    # 索引原因，需要再+1
                    section_df = cleaned_share_history_df.iloc[current_row_index:current_row_index+self.numbers_of_prediction+1, :].copy()
                    label_df = pd.concat([label_df, self.label_rule.create_label_df(section_df)], axis=1, ignore_index=True)
            # 转置，行为样本数（日期），列为特征
            feature_df = feature_df.T
            label_df = label_df.T
            # 更改列名，防止后续读取时列名为纯数字带来的问题
            feature_df.columns = ["F{0}".format(x) for x in range(feature_df.shape[1])] 
            label_df.columns = ["L{0}".format(x) for x in range(label_df.shape[1])] 
            # 保存至csv
            feature_csv_name = os.path.join(trainer_config.feature_data_directory_path,code+".csv")
            feature_df.to_csv(feature_csv_name, index=False)
            print("{0} feature ====> ok!".format(code))
            label_csv_name = os.path.join(trainer_config.label_data_directory_path,code+".csv")
            label_df.to_csv(label_csv_name, index=False)
            print("{0} label ====> ok!".format(code))

    '''
    def __get_max_min_value_to_csv(self,code:str,df:pd.DataFrame)->None:
        auxiliary_df = pd.DataFrame(columns=[['code','closed_price_min','closed_price_max','highest_price_min','highest_price_max',\
            'lowest_price_min','lowest_price_max','opened_price_min','opened_price_max','change_rate_min','change_rate_max',\
                'traded_volume_min','traded_volume_max','traded_amount_min','traded_amount_max','total_share_capital_min','total_share_capital_max',\
                    'flow_share_capital_min','flow_share_capital_max']])
        auxiliary_df.loc[0,'code'] = code
        auxiliary_df.loc[0,'closed_price_min'] =df.loc[:,'closed_price'].min()
        auxiliary_df.loc[0,'closed_price_max'] =df.loc[:,'closed_price'].max()
        auxiliary_df.loc[0,'highest_price_min'] =df.loc[:,'highest_price'].min()
        auxiliary_df.loc[0,'highest_price_max'] =df.loc[:,'highest_price'].max()
        auxiliary_df.loc[0,'lowest_price_min'] =df.loc[:,'lowest_price'].min()
        auxiliary_df.loc[0,'lowest_price_max'] =df.loc[:,'lowest_price'].max()
        auxiliary_df.loc[0,'opened_price_min'] =df.loc[:,'opened_price'].min()
        auxiliary_df.loc[0,'opened_price_max'] =df.loc[:,'opened_price'].max()
        auxiliary_df.loc[0,'change_rate_min'] =df.loc[:,'change_rate'].min()
        auxiliary_df.loc[0,'change_rate_max'] =df.loc[:,'change_rate'].max()
        auxiliary_df.loc[0,'traded_volume_min'] =df.loc[:,'traded_volume'].min()
        auxiliary_df.loc[0,'traded_volume_max'] =df.loc[:,'traded_volume'].max()
        auxiliary_df.loc[0,'traded_amount_min'] =df.loc[:,'traded_amount'].min()
        auxiliary_df.loc[0,'traded_amount_max'] =df.loc[:,'traded_amount'].max()
        auxiliary_df.loc[0,'total_share_capital_min'] =df.loc[:,'total_share_capital'].min()
        auxiliary_df.loc[0,'total_share_capital_max'] =df.loc[:,'total_share_capital'].max()
        auxiliary_df.loc[0,'flow_share_capital_min'] =df.loc[:,'flow_share_capital'].min()
        auxiliary_df.loc[0,'flow_share_capital_max'] =df.loc[:,'flow_share_capital'].max()
        auxiliary_df.set_index('code',inplace=True)
        auxiliary_csv_path = os.path.join(trainer_config.mapminmax_record_data_path,code+".csv")
        auxiliary_df.to_csv(auxiliary_csv_path)
    '''

    def __reshape_df_into_one_row(self,df:pd.DataFrame)->pd.DataFrame: #输出为多行单列的“向量” df
        column_count = df.shape[1]
        data = []
        for i in range(column_count):
            data.append(df.iloc[:,i].tolist())
        # 存入pic_df
        flatten_data = [y for x in data for y in x]
        pic_df = pd.DataFrame(flatten_data)
        return pic_df

    def __closed_price_fft_into_multiple_periods_prices_df(self,share_history_df:pd.DataFrame,taken_fft_channel_numbers:int)->pd.DataFrame:
        price_list = share_history_df['closed_price'].to_numpy()
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
'''
    def __creating_label_df(self, df: pd.DataFrame)-> pd.DataFrame:
        # 初始化label
        label_list = [0.0, 0.0]
        closed_price_list = df['closed_price'].tolist()
        # 当天股价大于明天,且升幅大于1%，则标记为1
        if closed_price_list[1] >= closed_price_list[0] * 1.01:
            label_list[0] = 1.0
        # 后天收盘价不比明天小，则标记为1
        if closed_price_list[2] >= closed_price_list[1]:
            label_list[1] = 1.0
        label_df = pd.DataFrame(label_list)
        return label_df
'''

'''   
    def read_feature_data_csv(self,code:str)->pd.DataFrame:
        file_path = os.path.join(trainer_config.feature_data_directory_path,code+".csv")
        return pd.read_csv(file_path)

    def read_label_data_csv(self,code:str)->pd.DataFrame:
        file_path = os.path.join(trainer_config.label_data_directory_path,code+".csv")
        return pd.read_csv(file_path)   

    def temp_problem_fix(self):
        for file_name in os.listdir(trainer_config.feature_data_directory_path):
            code=file_name[0:6]
            df = self.read_feature_data_csv(code)
            if df.iloc[0,:].isnull().any():
                print(code)

    # to show the irfft result for testing
    def plot_rfft_and_irfft_figure(self,share_history_df:pd.DataFrame,taken_fft_channel_numbers:int)->None:
        price_list = share_history_df['closed_price'].to_numpy()
        figure_num = taken_fft_channel_numbers + 1
        numbers_of_prediction = 10
        irfft_df = self.closed_price_fft_into_multiple_periods_prices_df(share_history_df,taken_fft_channel_numbers,numbers_of_prediction)
        # plot
        # 1st share price
        plt.subplot(figure_num,2,1)
        plt.grid(ls='--')
        plt.plot(range(len(price_list)),price_list)
        plt.title('share')
        plt.ylabel('price')
        plt.xlabel('x')
        # 2nd irfft summary
        irfft_df['sum']=irfft_df.loc[:,:].apply(lambda x:x.sum(),axis=1)
        irfft_y = irfft_df['sum'].tolist()
        plt.subplot(figure_num,2,2)
        plt.grid(ls='--')
        plt.plot(range(len(irfft_y)),irfft_y)
        plt.title('irfft_summary')
        plt.ylabel('irfft_sum')
        plt.xlabel('x')
        # divided irfft figure
        figure_index = 3
        for c in irfft_df.columns:
            irfft_y = irfft_df[c].tolist()
            plt.subplot(figure_num,2,figure_index)
            plt.grid(ls='--')
            plt.plot(range(len(irfft_y)),irfft_y)
            plt.title("F{0}".format(figure_index-2))
            plt.ylabel('price')
            plt.xlabel('x')
            figure_index+=1
        plt.show()
'''