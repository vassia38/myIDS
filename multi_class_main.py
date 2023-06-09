from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

import sklearn.model_selection as skms
import sklearn.metrics as skmetric
import sklearn.preprocessing as skpre
import scikitplot as skplot
import matplotlib.pyplot as plt
import pandas as pd
import utils
from plotting import *
from imblearn.over_sampling import SMOTE


def prepare_data(benign_size=10000, benign_skip_n=0, malicious_size=2500, malicious_skip_n=0):
    df = utils.load_data("NF-UQ-NIDS-v2_benign.csv", "NF-UQ-NIDS-v2_malicious.csv",
                         benign_size, benign_skip_n,
                         malicious_size, malicious_skip_n)
    # print("input sparsity ratio:{:.3f}".format(utils.get_sparsity_ratio(df)))
    
    X = df.drop('Attack', axis="columns")
    X_feature_names = X.columns
    X = X.values
    y = pd.DataFrame(df['Attack'])
    y, y_labels_map = utils.encode_labels(y['Attack'].values)
    print("\n")
    print(pd.DataFrame(y).value_counts())
    print(pd.DataFrame(y).count())
    print(y_labels_map)
    print("\n")

    return X, X_feature_names, y, y_labels_map

def get_best_model_from_cross_val(benign_size=10000, benign_skip_n=0, 
                                  malicious_size=2500, malicious_skip_n=0):
    X, X_feature_names, y, y_label_map = prepare_data(benign_size, benign_skip_n,
                       malicious_size, malicious_skip_n)
    rfc = RandomForestClassifier()
    result = cross_validate(rfc, X, y, cv=5, return_estimator=True, return_train_score=True)
    print(result)
    
    best_estim_index = 0
    for i in range(len(result['test_score'])):
        if result['test_score'][i] > result['test_score'][best_estim_index]:
            best_estim_index = i
    
    return result['estimator'][best_estim_index]

def evaluate_model(model, benign_size=10000, benign_skip_n=0, 
                   malicious_size=2500, malicious_skip_n=0):
    X_test, X_feature_names, y_test, y_labels_map = prepare_data(benign_size, benign_skip_n,
                                                                malicious_size, malicious_skip_n)
    
    y_predicted = model.predict(X_test)
    acc = skmetric.accuracy_score(y_predicted, y_test)
    print("{:.5f}".format(acc))
    
    show_feature_importance(rfc, X_feature_names)
    show_confusion_matrix(y_test, y_predicted, y_labels_map.keys())

benign_train_size = 2000000
malicious_train_size = 2000000
# rfc = get_best_model_from_cross_val(benign_train_size, 0, malicious_train_size, 0)
# utils.save_model(rfc)

rfc = utils.load_model()
evaluate_model(rfc, 1000, benign_train_size, 100000, malicious_train_size)