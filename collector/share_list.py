import os.path
import pandas as pd
from collector import config
from os import path
import tushare as ts


class ShareListDownloader:
    def __init__(self) -> None:
        # =================build dir============================
        self.file_path = path.join(config.cleaned_share_list_directory_path, 'share_list.csv')

    def download_share_list(self):
        ts.set_token('e8c995e8571eb7a82cd0b12f561da39bef1f6ac6c96195812ab0107b')  # tusahre key
        pro = ts.pro_api()   # tushare api setting
        config.build_or_clear_dir(config.cleaned_share_list_directory_path)   # clean and rebuild dir for the list
        data_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        data_df['ts_code'] = "'" + data_df['ts_code']
        data_df['symbol'] = "'" + data_df['symbol']
        data_df.to_csv(self.file_path, index=False)

    def read_share_list(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)


