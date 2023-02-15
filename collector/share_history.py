import time
import pandas as pd
import tushare as ts
from collector import config, share_list
import datetime
import os


class ShareHistoryDownloader:
    def __init__(self):
        self.download_counting_per_minute = config.download_counting_per_minute
        config.build_or_clear_dir(config.temp_share_history_directory_path)
        self.share_list_class = share_list.ShareListDownloader()
        config.build_dir(config.cleaned_share_history_directory_path)
        self.__download_share_history()
        self.__concat_and_update_cleaned_share_history()

    def __download_share_history(self):
        ts.set_token('e8c995e8571eb7a82cd0b12f561da39bef1f6ac6c96195812ab0107b')  # tusahre key
        pro = ts.pro_api()   # tushare api setting
        the_share_list = self.__create_share_date_list()  # 生成初始的表格 【股票代码（用于tushare查询的），config日期】
        the_share_list = self.__update_share_latest_date_if_exist(the_share_list)  # 查询已下载历史数据，按最新数据日期更新
        today_date = datetime.date.today().strftime('%Y%m%d')
        #  程序计数计时，tushare复权日行情接口限制 200次/分钟
        count = 0
        t1 = time.perf_counter()
        for index, row_data in the_share_list.iterrows():
            ts_code = row_data.loc['ts_code'][1::]  # 去除之前加注的"'"
            print("{0} :: downloading.".format(ts_code))
            since_date = row_data.loc['since_date']
            # 日数据（非复权）
            #data_df = pro.daily(ts_code=ts_code, start_date=since_date, end_date=today_date)
            # 后复权日数据
            data_df = ts.pro_bar(ts_code=ts_code, adj='hfq', start_date=since_date, end_date=today_date)
            data_csv_path = os.path.join(config.temp_share_history_directory_path, ts_code+".csv")
            data_df.to_csv(data_csv_path, index=False)
            count += 1
            t2 = time.perf_counter() - t1
            #  在接近60秒期间监测连接次数是否大于限制连接次数，如大于499次，则休眠至1分钟+1秒后
            if 57 < t2 < 60:
                if count >= self.download_counting_per_minute:
                    time.sleep(61 - t2)  # 休眠至1分钟+1秒计时后
                    print("连接次数超499次/分钟，休眠（s）：", 61-t2)
                    t2 = time.perf_counter() - t1
            # 如果计时超过一分钟，重置计数和计时
            if t2 > 60:
                count = 0
                t1 = time.perf_counter()

    def __create_share_date_list(self) -> pd.DataFrame:
        sl = self.share_list_class.read_share_list()
        the_share_list = pd.DataFrame(sl.loc[:, 'ts_code'], columns=['ts_code'])
        the_share_list['since_date'] = config.start_date
        return the_share_list

    def __update_share_latest_date_if_exist(self, the_share_list: pd.DataFrame) -> pd.DataFrame:
        files_list = os.listdir(config.cleaned_share_history_directory_path)
        if len(files_list):  # 如果列表不为空
            for f in files_list:
                code = f[0:6]
                file_path = os.path.join(config.cleaned_share_history_directory_path, f)
                # 如果code不是纯数字，跳过
                if not code.isdigit():
                    continue
                ts_code = f[0:9]
                df = pd.read_csv(file_path, index_col='trade_date')
                df = df.sort_index(ascending=True)
                if not df.empty:
                    latest_date = datetime.datetime.strptime(str(df.index[-1]), "%Y%m%d")
                    since_date = (latest_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
                    for _, row_content in the_share_list.iterrows():
                        if row_content['ts_code'] == "'" + ts_code:
                            row_content['since_date'] = since_date
                            print("{0} :: {1}".format(row_content['ts_code'], since_date))
                            break
        return the_share_list

    def __concat_and_update_cleaned_share_history(self):
        config.build_dir(config.cleaned_share_history_directory_path)
        # 如果clean中对应share文件为空，则复制
        temp_history_csv_list = os.listdir(config.temp_share_history_directory_path)
        cleaned_history_csv_list = os.listdir(config.cleaned_share_history_directory_path)
        for temp_csv in temp_history_csv_list:
            temp_csv_file_path = os.path.join(config.temp_share_history_directory_path, temp_csv)
            cleaned_csv_file_path = os.path.join(config.cleaned_share_history_directory_path, temp_csv)
            if temp_csv not in cleaned_history_csv_list:
                os.system('cp {0} {1}'.format(temp_csv_file_path, cleaned_csv_file_path))
            else:  # 股票已有文件存在clean文件
                temp_history_df = pd.read_csv(temp_csv_file_path)
                if not temp_history_df.empty:
                    clean_history_df = pd.read_csv(cleaned_csv_file_path)
                    clean_history_df = pd.concat([clean_history_df, temp_history_df])
                    clean_history_df.set_index('trade_date', inplace=True)
                    clean_history_df.sort_index(ascending=True, inplace=True)
                    clean_history_df.to_csv(cleaned_csv_file_path)

