import os
from config import root_config

data_directory_path = root_config.data_directory_path
application_data_directory_path = os.path.join(data_directory_path, "Application")
accuracy_table_path = os.path.join(application_data_directory_path, 'accuracy_table.csv')
prediction_table_path = os.path.join(application_data_directory_path, 'predict_table.xlsx')
prediction_model_directory_path = root_config.model_directory_path


def build_dir(dir_path) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
