import datetime
import os
import shutil
from config import root_config

"""Thoughts:
1. downloads = list, history, industry files in csv
2. upper files in different directory
3. every time download action will delete the directory and recreate
end => no data process in this module
"""

start_date = "20120220"  # 当天后，行业数据完整
download_counting_per_minute = 199  # 复权行情接口200次/分钟

# 原始数据存贮目录分配
data_directory_path = root_config.data_directory_path
cleaned_data_directory_path = os.path.join(data_directory_path, "Cleaned")
temp_data_directory_path = os.path.join(data_directory_path, "Temp")
# ========== share list =============================
cleaned_share_list_directory_path = os.path.join(cleaned_data_directory_path, "ShareList")
cleaned_share_history_directory_path = os.path.join(cleaned_data_directory_path, "ShareHistory")
# ========== share history =============================
#  share list下载版本已处理，不再另行处理
temp_share_history_directory_path = os.path.join(temp_data_directory_path, "ShareHistory")


# 文件夹创建
def build_dir(dir_path) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# 文件夹清空&创建
def build_or_clear_dir(dir_path)->None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

'''
# 工作日生成 yield [2012-02-21 ...]
def get_workdays(date_pointer:datetime.date = datetime.date.today()):
    #date pointer init as present date as default if no input
    while date_pointer.isoformat() != start_date:
        weekday = date_pointer.isoweekday()
        if weekday == 6 or weekday == 7:  # if weekend ,then pass
            date_pointer = date_pointer - datetime.timedelta(1)
            continue
        else:
            yield date_pointer.isoformat()
            date_pointer = date_pointer - datetime.timedelta(1)
'''