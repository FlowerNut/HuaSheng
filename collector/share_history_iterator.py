from collector import config
import os
import pandas as pd


class HistoryCSVIterator:
    def __init__(self) -> None:
        self.cleaned_csv_files_list = os.listdir(config.cleaned_share_history_directory_path)
        self.cleaned_csv_files_path = []
        for f in self.cleaned_csv_files_list:
            if 'csv' in f:
                self.cleaned_csv_files_path.append(os.path.join(config.cleaned_share_history_directory_path, f))

    def __len__(self):
        return len(self.cleaned_csv_files_path)

    def __getitem__(self, index):
        print(self.cleaned_csv_files_path[index])
        code = self.cleaned_csv_files_path[index][-13:-7]
        cleaned_share_history_df = pd.read_csv(self.cleaned_csv_files_path[index])
        cleaned_share_history_df.set_index('trade_date', inplace=True)
        cleaned_share_history_df.sort_index(ascending=True, inplace=True)
        code = "'" + code
        return code, cleaned_share_history_df   # 特征为列向量，沿行方向为时间轴

