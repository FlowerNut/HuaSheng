from config import label_rules_2days
from model import predict_config as app_config, combine_prediction, predict_using_cnn
import os

if __name__ == '__main__':
    # ----------positive predict----------------------
    positive_3day_predict_result_file_path = os.path.join(app_config.application_data_directory_path, 'positive_3day_prediction.csv')
    positive_3day_label_rule = label_rules_2days.Positive3Days1Percent()
    predict_using_cnn.PredictByCNN(positive_3day_label_rule, positive_3day_predict_result_file_path)
    # ----------negative predict----------------------
    negative_5day_predict_result_file_path = os.path.join(app_config.application_data_directory_path, 'negative_5day_prediction.csv')
    negative_5day_label_rule = label_rules_2days.Negative5Days1percent()
    predict_using_cnn.PredictByCNN(negative_5day_label_rule, negative_5day_predict_result_file_path)

    # ---------- combine p and n results --------------
    #combine_prediction.Combination(positive_2day_predict_result_file_path, negative_2day_predict_result_file_path, plus_t_minus_f=False)
    combine_prediction.Combination(positive_3day_predict_result_file_path, negative_5day_predict_result_file_path, plus_t_minus_f=False)

