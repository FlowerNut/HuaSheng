import pandas as pd
from cleaner import config as cleaner_config
from collector import config as collector_config
import os

class ShareListCleaner:
    def __init__(self) -> None:
        cleaner_config.build_dir(cleaner_config.cleaned_share_list_directory_path)
        self.temp_share_list_csv_path = os.path.join(collector_config.temp_share_list_directory_path,'share_list.xlsx')
        self.cleaned_share_list_excel_path = os.path.join(cleaner_config.cleaned_share_list_directory_path,'share_list.xlsx')

    def __read_temp_excel(self)->pd.DataFrame:
        df = pd.read_excel(self.temp_share_list_csv_path, dtype={'A股代码':str})
        df = df[["A股代码","A股简称","公司全称","板块","A股总股本","A股流通股本","省    份","所属行业"]]\
            .rename(columns={"A股代码":'code',"A股简称":'company_name',"公司全称":'company_full_name',"板块":'field',"A股总股本":'total_share_capital',"A股流通股本":'flow_share_capital',"省    份":'province',"所属行业":'industry'})
        return df

    def read_cleaned_excel(self)->pd.DataFrame:
        cleaned_df = pd.read_excel(self.cleaned_share_list_excel_path,dtype={'code':str})
        return cleaned_df

    def update_cleaned_share_list_csv(self):
        temp_df = self.__read_temp_excel()
        if os.path.exists(self.cleaned_share_list_excel_path):
            cleaned_df = self.read_cleaned_excel()
            cleaned_df.update(temp_df)
        else:
            temp_df.to_excel(self.cleaned_share_list_excel_path)
        