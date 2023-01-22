import datetime
import pandas as pd
from collector import config as collector_config
from cleaner import share_list
from cleaner import config as cleaner_config
from cleaner import share_history
from requests import get as requests_get
import os


"""Through:
Action: Download csv from url;
Input: pd.DataFrame[share_code, since_date]
"""


class ShareHistoryDownloader:
    def __init__(self) -> None:
        collector_config.build_or_clear_dir(collector_config.temp_share_history_directory_path)
        self.share_list_cleaner = share_list.ShareListCleaner()
        self.share_history_cleaner = share_history.ShareHistoryCleaner()
        self.__prepare_for_downloading()
        self.download_share_history(self.__prepare_for_downloading())
        
    def __prepare_for_downloading(self)->pd.DataFrame:
        list_df = self.share_list_cleaner.read_cleaned_excel()
        code_date_df = list_df[['code']]
        # since date : start date defined in config
        code_date_df.loc[:, 'since_date'] = datetime.datetime.strptime(collector_config.start_date, '%Y-%m-%d').strftime('%Y%m%d')
        code_date_df.set_index('code', inplace=True)
        # if clean share dir exists!!
        latest_date_df = self.__get_latest_date_from_cleaned_share_history_csv()
        if not latest_date_df.empty:
        # loop latest_date_df, modify code_date_df which fill with today with str format
            for row_index, row_series in latest_date_df.iterrows():
                code = row_index
                if code in code_date_df.index.values:
                    code_date_df.at[code, 'since_date'] = row_series['since_date']
        return code_date_df

    def download_share_history(self,df:pd.DataFrame):
        today_date = datetime.date.today().strftime('%Y%m%d')
        for row_index, row_series in df.iterrows():
            code = row_index
            url = self.__get_sz_history_flow_url(code, row_series['since_date'],today_date)
            file_path = os.path.join(collector_config.temp_share_history_directory_path,code+'.csv')
            self.__download_file(url, file_path)
            print("{0} =====> downloaded.".format(code))
            
    def __get_sz_history_flow_url(self,code:str,since_date:str,to_date:str):
        # code = 1 for sz, 0 for hs
        url = "http://quotes.money.163.com/service/chddata.html?code=1{code}&start={theStartDate}&end={theEndDate}&fields=TCLOSE;HIGH;LOW;TOPEN;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP".format(code=code,theStartDate=since_date,theEndDate=to_date)
        print(url)
        return url

    def __download_file(self,url:str,file_path:str)->None:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4146.4 Safari/537.36'}
        r = requests_get(url, headers=headers)
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(r.content)
                f.close()

    def __get_latest_date_from_cleaned_share_history_csv(self) -> pd.DataFrame:
        latest_date_df = pd.DataFrame(columns=['code', 'since_date'])
        if os.path.exists(cleaner_config.cleaned_share_history_directory_path):
            files = os.listdir(cleaner_config.cleaned_share_history_directory_path)
            for i, f in enumerate(files):
                code = f[0:6]
                # 如果code不是纯数字，跳过
                if not code.isdigit():
                    continue
                df = self.share_history_cleaner.read_cleaned_share_history_csv(code)  # sorted df by date ascending
                # if df (clean share history) not empty ???
                if not df.empty:
                    since_date = (df.index[-1] + datetime.timedelta(days=1)).strftime('%Y%m%d')
                    latest_date_df.loc[i] = {'code': code, 'since_date': since_date}
        latest_date_df.set_index('code', inplace=True)
        # if not exist, return empty df
        return latest_date_df

