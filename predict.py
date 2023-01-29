from config import label_rules_2days
from model import predict_config as app_config, combine_prediction, predict_using_cnn
import os

if __name__ == '__main__':
    # ----------positive predict----------------------
    positive_2day_predict_result_file_path = os.path.join(app_config.application_data_directory_path, 'positive_2day_prediction.csv')
    positive_2day_label_rule = label_rules_2days.Positive2Days()
    predict_using_cnn.PredictByCNN(positive_2day_label_rule, positive_2day_predict_result_file_path)
    # ----------negative predict----------------------
    #negative_predict_result_file_path = os.path.join(app_config.application_data_directory_path, 'negative_10day_prediction.csv')
    #negative_10day_label_rule = label_rules_2days.Negative10Days()
    #predict_using_cnn.PredictByCNN(negative_10day_label_rule, negative_predict_result_file_path)
    # ----------negative predict----------------------
    negative_2day_predict_result_file_path = os.path.join(app_config.application_data_directory_path, 'negative_2day_prediction.csv')
    negative_2day_label_rule = label_rules_2days.Negative2Days()
    predict_using_cnn.PredictByCNN(negative_2day_label_rule, negative_2day_predict_result_file_path)
    # ---------- combine p and n results --------------
    combine_prediction.Combination(positive_2day_predict_result_file_path, negative_2day_predict_result_file_path, plus_t_minus_f=False)

