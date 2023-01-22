import pandas as pd
from collector import config as collector_config
from cleaner import config as cleaner_config
import os
import numpy as np

# note: temp and cleaned csv shall share same file name format: code.csv
class ShareHistoryCleaner:
    def __init__(self) -> None:
        cleaner_config.build_dir(cleaner_config.cleaned_share_history_directory_path)
        self.temp_share_history_directory_path = collector_config.temp_share_history_directory_path
        self.cleaned_share_history_directory_path = cleaner_config.cleaned_share_history_directory_path

    def __get_file_list(self,dictionary_path) -> list:
        return os.listdir(dictionary_path)

    def __read_temp_share_history_csv(self,code:str) -> pd.DataFrame:
        file_path = os.path.join(self.temp_share_history_directory_path, code+".csv")
        return pd.read_csv(file_path, encoding='gbk')

    def __change_temp_share_history_df_title(self,df:pd.DataFrame)->pd.DataFrame:
        # replace title
        df = df.rename(columns={"日期":'date',"股票代码":'code',"名称":'company_name',"收盘价":'closed_price',\
            "最高价":'highest_price',"最低价":'lowest_price',"开盘价":'opened_price',"换手率":'change_rate',\
                "成交量":'traded_volume',"成交金额":'traded_amount',"总市值":'total_share_capital',\
                    "流通市值":'flow_share_capital'})
        df.set_index('date',inplace=True)
        return df
    
    def __clean_concat_share_history_df(self,df:pd.DataFrame)->pd.DataFrame:
        df.replace(0,np.nan,inplace=True)
        # fill na by data of the day of yesterday
        # 如果首行有空，则删除（循环直到首行不空）
        for row_index,row_content in df.iterrows():
            if row_content.isnull().any():
                df.drop(index=row_index,inplace=True)
            else:
                break
        # 首行不为空，则向下填充
        df.fillna(method='ffill',inplace=True)
        return df
        
    def read_cleaned_share_history_csv(self, code: str, datetime_index=True) -> pd.DataFrame:
        file_path = os.path.join(self.cleaned_share_history_directory_path, code+".csv")
        clean_share_history_df = pd.read_csv(file_path, encoding='utf-8', index_col='date')
        if datetime_index == True:
            clean_share_history_df.index = pd.to_datetime(clean_share_history_df.index)
        return clean_share_history_df.sort_index(ascending=True)

    def concat_into_clean_share_history_csv(self):
        temp_share_history_file_list = os.listdir(self.temp_share_history_directory_path)
        for file_name in temp_share_history_file_list:
            code = file_name[0:6]
            # 如果code不是纯数字，跳过
            if not code.isdigit():
                continue
            print(code)
            temp_share_history_df = self.__read_temp_share_history_csv(code)
            if not temp_share_history_df.empty:
                temp_share_history_df = self.__change_temp_share_history_df_title(temp_share_history_df)
                cleaned_share_csv_path = os.path.join(self.cleaned_share_history_directory_path,file_name)
                if os.path.exists(cleaned_share_csv_path):
                    # read, than merge
                    clean_share_history_df = self.read_cleaned_share_history_csv(code, datetime_index=False)
                    clean_share_history_df = pd.concat([clean_share_history_df, temp_share_history_df])
                    clean_share_history_df = self.__clean_concat_share_history_df(clean_share_history_df)
                    clean_share_history_df.to_csv(cleaned_share_csv_path)
                else:
                    self.__clean_concat_share_history_df(temp_share_history_df)
                    temp_share_history_df.to_csv(cleaned_share_csv_path)

'''
    def temp_fix_problem(self):
        data_directory_path = cleaner_config.data_directory_path
        training_data_directory_path = os.path.join(data_directory_path,"Train")
        feature_data_directory_path = os.path.join(training_data_directory_path,"Feature")
        label_data_directory_path = os.path.join(training_data_directory_path,"Label")
        mapminmax_record_data_path = os.path.join(data_directory_path,"Auxiliary")
        list_tobe_fix  = []
        for file_name in os.listdir(self.cleaned_share_history_directory_path):
            code = file_name[0:6]
            clean_share_history_df = self.read_cleaned_share_history_csv(code, datetime_index=True)
            if clean_share_history_df.iloc[0,:].isnull().any():
                list_tobe_fix.append(code)
                os.remove(os.path.join(self.cleaned_share_history_directory_path,code+".csv"))
        
        for file_name in os.listdir(feature_data_directory_path):
            if file_name in list_tobe_fix:
                os.remove(os.path.join(feature_data_directory_path,code+".csv"))
                os.remove(os.path.join(label_data_directory_path,code+".csv"))
                os.remove(os.path.join(mapminmax_record_data_path,code+".csv"))
                print("{0} files removes".format(code))

'''
