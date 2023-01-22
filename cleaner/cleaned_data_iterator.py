from cleaner import config as cleaner_config
from cleaner import share_history
import os
import pandas as pd


class HistoryCSVIterator:
    def __init__(self) -> None:
        self.share_history_cleaner = share_history.ShareHistoryCleaner()
        self.cleaned_csv_files_list = os.listdir(cleaner_config.cleaned_share_history_directory_path)
        self.cleaned_csv_files_path = []
        for f in self.cleaned_csv_files_list:
            if 'csv' in f:
                self.cleaned_csv_files_path.append(os.path.join(cleaner_config.cleaned_share_history_directory_path, f))

    def __len__(self):
        return len(self.cleaned_csv_files_path)

    def __getitem__(self, index):
        print(self.cleaned_csv_files_path[index])
        code = self.cleaned_csv_files_path[index][-10:-4]
        cleaned_share_history_df = self.share_history_cleaner.read_cleaned_share_history_csv(code, datetime_index=True)
        code = "'" + code
        return code, cleaned_share_history_df   # 特征为列向量，沿行方向为时间轴

