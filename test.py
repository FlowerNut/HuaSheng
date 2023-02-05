import time
import tushare as ts

#pro = ts.pro_api()  # tushare api setting

df = ts.pro_bar(ts_code='000001.SZ', adj='hfq', start_date='20180101', end_date='20181011')

print(df)

