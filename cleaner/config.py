import os
from config import root_config

data_directory_path = root_config.data_directory_path
cleaned_data_directory_path = os.path.join(data_directory_path, "Cleaned")
cleaned_share_list_directory_path = os.path.join(cleaned_data_directory_path, "ShareList")
cleaned_share_history_directory_path = os.path.join(cleaned_data_directory_path, "ShareHistory")
cleaned_share_industry_directory_path = os.path.join(cleaned_data_directory_path, "ShareIndustry")


# 文件夹创建
def build_dir(dir_path) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

