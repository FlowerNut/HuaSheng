import pandas as pd
from model import predict_config as app_config
from collector import share_list

class Combination:
    def __init__(self, base_csv_path, second_csv_path, plus_t_minus_f=True):
        self.base_csv_path = base_csv_path
        self.second_csv_path = second_csv_path
        self.plus_or_minus = plus_t_minus_f
        self.__combine()

    def __combine(self):
        base_df = pd.read_csv(self.base_csv_path, encoding='utf-8')
        second_df = pd.read_csv(self.second_csv_path, encoding='utf-8')
        predict_df = pd.DataFrame(columns=['code', 'date', 'value1', 'value2', 'score'])
        # 表格中有不参与计算的"日期"内容，用笨办法遍历
        if self.plus_or_minus:  # plus
            for base_df_row_index, base_df_row_content in base_df.iterrows():
                for second_df_row_index, second_df_row_content in second_df.iterrows():
                    if base_df_row_content['code'] == second_df_row_content['code']:
                        value1 = base_df_row_content['value1'] + second_df_row_content['value1']
                        value2 = base_df_row_content['value2'] + second_df_row_content['value2']
                        score = value1 * 0.618 + value2 * (1 - 0.618)
                        temp_df = pd.DataFrame([[base_df_row_content['code'], base_df_row_content['date'],
                                                 value1, value2, score]],
                                               columns=['code', 'date', 'value1', 'value2', 'score'])
                        predict_df = pd.concat([predict_df, temp_df], axis=1, ignore_index=True)
                        continue
        else:  # minus
            for base_df_row_index, base_df_row_content in base_df.iterrows():
                for second_df_row_index, second_df_row_content in second_df.iterrows():
                    if base_df_row_content['code'] == second_df_row_content['code']:
                        value1 = base_df_row_content['value1'] - second_df_row_content['value1']
                        value2 = base_df_row_content['value2'] - second_df_row_content['value2']
                        score = value1 * 0.618 + value2 * (1 - 0.618)
                        temp_df = pd.DataFrame([[base_df_row_content['code'], base_df_row_content['date'],
                                                 value1, value2, score]],
                                               columns=['code', 'date', 'value1', 'value2', 'score'])
                        predict_df = pd.concat([predict_df, temp_df], axis=0, ignore_index=True)
                        break
        predict_df = predict_df.loc[(predict_df['value1'] > 0.9) & (predict_df['value2'] > 0.9)]
        predict_df = predict_df.sort_values(by='score', ascending=False)
        result_df = self.__fill_base_info(predict_df)
        result_df.to_excel(app_config.prediction_table_path, index=False)

    def __fill_base_info(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        # 读取share list
        share_list_downloader = share_list.ShareListDownloader()
        share_list_df = pd.read_csv(share_list_downloader.file_path)
        # 向predict df添加空列
        result_df = pd.DataFrame(columns=['code', 'name', 'industry', 'date', 'value1', 'value2', 'score'])
        for predict_row_index, predict_row_content in predict_df.iterrows():
            for share_list_row_index, share_list_row_content in share_list_df.iterrows():
                if share_list_row_content['symbol'] == predict_row_content['code']:
                    temp_df = pd.DataFrame([[predict_row_content['code'], share_list_row_content['name'],
                                             share_list_row_content['industry'], predict_row_content['date'],
                                             predict_row_content['value1'], predict_row_content['value2'],
                                             predict_row_content['score']]],
                                           columns=['code', 'name', 'industry', 'date', 'value1', 'value2', 'score'])
                    result_df = pd.concat([result_df, temp_df], axis=0, ignore_index=True)
                    continue
        return result_df


