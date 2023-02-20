import os
from config import root_config
import shutil

# 训练数据存贮目录
data_directory_path = root_config.data_directory_path
model_directory_path = root_config.model_directory_path
training_data_directory_path = os.path.join(data_directory_path, "Train")
feature_data_directory_path = os.path.join(training_data_directory_path, "Feature")
label_data_directory_path = os.path.join(training_data_directory_path, "Label")

steam_length = 10
taken_fft_channel_numbers = 8

# 文件夹清空&创建
def build_or_clear_dir(dir_path) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


# 文件夹创建
def build__dir(dir_path) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

