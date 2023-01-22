import pandas as pd
from model import predict_config as app_config


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
                        score = value1 * (1-0.618) + value2 * 0.618
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
                        score = value1 * (1-0.618) + value2 * 0.618
                        temp_df = pd.DataFrame([[base_df_row_content['code'], base_df_row_content['date'],
                                                 value1, value2, score]],
                                               columns=['code', 'date', 'value1', 'value2', 'score'])
                        predict_df = pd.concat([predict_df, temp_df], axis=0, ignore_index=True)
                        continue
        predict_df = predict_df.loc[(predict_df['value1'] > 0.618) & (predict_df['value2'] > 0.618)]
        predict_df = predict_df.sort_values(by='score', ascending=False)
        predict_df.to_csv(app_config.prediction_table_path, index=False)

