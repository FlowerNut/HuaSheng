import pandas as pd
import tushare as ts
from collector import config, share_list
import datetime
import os


class ShareHistoryDownloader:
    def __init__(self):
        config.build_or_clear_dir(config.temp_share_history_directory_path)
        self.share_list_class = share_list.ShareListDownloader()

    def download_share_history(self):
        ts.set_token('e8c995e8571eb7a82cd0b12f561da39bef1f6ac6c96195812ab0107b')  # tusahre key
        pro = ts.pro_api()   # tushare api setting
        the_share_list = self.__create_share_date_list()  # 生成初始的表格 【股票代码（用于tushare查询的），config日期】
        the_share_list = self.__update_share_latest_date_if_exist(the_share_list)  # 查询已下载历史数据，按最新数据日期更新
        today_date = datetime.date.today().strftime('%Y%m%d')
        for index, row_data in the_share_list.iterrows():
            ts_code = row_data.loc['ts_code'][1::]  # 去除之前加注的"'"
            print("{0} :: downloading.", ts_code)
            since_date = row_data.loc['since_date']
            data_df = pro.daily(ts_code=ts_code, start_date=since_date, end_date=today_date)
            data_csv_path = os.path.join(config.temp_share_history_directory_path, ts_code+".csv")
            data_df.to_csv(data_csv_path, index=False)

    def __create_share_date_list(self) -> pd.DataFrame:
        sl = self.share_list_class.read_share_list()
        the_share_list = pd.DataFrame(sl.loc[:, 'ts_code'], columns=['ts_code'])
        the_share_list['since_date'] = config.start_date

        return the_share_list

    def __update_share_latest_date_if_exist(self, the_share_list: pd.DataFrame) -> pd.DataFrame:
        files_list = os.listdir(config.cleaned_share_list_directory_path)
        if len(files_list):  # 如果列表不为空
            for f in files_list:
                code = f[0:6]
                # 如果code不是纯数字，跳过
                if not code.isdigit():
                    continue
                ts_code = f[0:9]
                df = pd.read_csv(f, index_col='trade_date')
                df = df.sort_index(ascending=True)
                if not df.empty:
                    since_date = (df.index[-1] + datetime.timedelta(days=1)).strftime('%Y%m%d')
                    the_share_list.loc[(df.index == ts_code)]['since_date'] = since_date
        return the_share_list

    def __concat_and_update_cleaned_share_history(self):
        pass
