from model import train_data_preparer, cnn_model_classifer_2days
from config import label_rules_2days

if __name__ == '__main__':
    #------------- positive model -------------------
    #positive_2day_label_rule = label_rules_2days.Positive2Days()
    #train_data_preparer.TrainingDataPreparer(positive_2day_label_rule, be_continue=True)
    #cnn_model_classifer_2days.Classifier(positive_2day_label_rule.model_file_name)
    #------------- negative 2 day model -------------------
    #negative_2day_label_rule = label_rules_2days.Negative2Days()
    #train_data_preparer.TrainingDataPreparer(negative_2day_label_rule, be_continue=True)
    #cnn_model_classifer_2days.Classifier(negative_2day_label_rule.model_file_name)
    #------------- negative 5 day model -------------------
    #negative_5day_label_rule = label_rules_2days.Negative5Days()
    #train_data_preparer.TrainingDataPreparer(negative_5day_label_rule, be_continue=True)
    #cnn_model_classifer_2days.Classifier(negative_5day_label_rule.model_file_name)
    #------------- positive model -------------------
    positive_3day_label_rule = label_rules_2days.Positive3Days()
    train_data_preparer.TrainingDataPreparer(positive_3day_label_rule, be_continue=False)
    cnn_model_classifer_2days.Classifier(positive_3day_label_rule.model_file_name)

