from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import utils
from plotting import *
import numpy as np


def prepare_data(benign_size=10000, benign_skip_n=0, malicious_size=2500, malicious_skip_n=0, binary_classif=False):
    df = utils.load_data(r"data\NF-UQ-NIDS-v2_Benign.csv", r"data\malicious",
                         benign_size, benign_skip_n,
                         malicious_size, malicious_skip_n,
                         binary_classif)
    # print("input sparsity ratio:{:.3f}".format(utils.get_sparsity_ratio(df)))
    class_column_name = 'Attack'
    if binary_classif:
        class_column_name = 'Label'
    
    X = df.drop(class_column_name, axis="columns")
    X_feature_names = X.columns
    X = X.values

    y = pd.DataFrame(df[class_column_name])
    if binary_classif:
        y_labels_map = {'Benign': 0, 'Malicious': 1}
        y = np.ravel(y)
    else:
        y, y_labels_map = utils.encode_labels(y[class_column_name].values)
    
    print(pd.DataFrame(y).value_counts())
    print(y_labels_map)
    print("\n")

    return X, X_feature_names, y, y_labels_map


def get_best_model_from_cross_val(benign_size=10000, benign_skip_n=0, 
                                  malicious_size=2500, malicious_skip_n=0,
                                  binary_classif=False, resample=False):
    X, X_feature_names, y, y_label_map = prepare_data(benign_size, benign_skip_n,
                                                      malicious_size, malicious_skip_n,
                                                      binary_classif)
    if resample:
        X,y = SMOTE().fit_resample(X,y)
        print(pd.DataFrame(y).value_counts())

    rfc = RandomForestClassifier()
    result = cross_validate(rfc, X, y, cv=5, return_estimator=True, return_train_score=True)
    print(result)
    
    best_estim_index = 0
    for i in range(len(result['test_score'])):
        if result['test_score'][i] > result['test_score'][best_estim_index]:
            best_estim_index = i
    
    return result['estimator'][best_estim_index]


def evaluate_model(model, benign_size=40, benign_skip_n=0, 
                   malicious_size=60, malicious_skip_n=0,
                   binary_classif=False):
    X_test, X_feature_names, y_test, y_labels_map = prepare_data(benign_size, benign_skip_n,
                                                                malicious_size, malicious_skip_n,
                                                                binary_classif)
    
    y_predicted = model.predict(X_test)
    acc = accuracy_score(y_predicted, y_test)
    print("{:.5f}".format(acc))
    
    #show_feature_importance(rfc, X_feature_names)
    show_confusion_matrix(y_test, y_predicted)
    return X_test, y_predicted, y_labels_map


def make_binary_rfc():
    benign_train_size = 20000
    malicious_train_size = 2500
    binary = True
    resample = False
    rfc = get_best_model_from_cross_val(benign_train_size, 0, malicious_train_size, 0, binary, resample)
    utils.save_model(rfc, "RF_SMOTE_bin.sav")
    evaluate_model(rfc, 50, benign_train_size, 50, malicious_train_size, binary)


def make_multic_rfc():
    benign_train_size = 0
    malicious_train_size = 2500
    binary = False
    resample = True
    rfc = get_best_model_from_cross_val(benign_train_size, 0, malicious_train_size, 0, binary, resample)
    utils.save_model(rfc, "RF_SMOTE_multi.sav")
    evaluate_model(rfc, 0, 0, 1000, 0, binary)

# make_binary_rfc()
# make_multic_rfc()

rfc_bin = utils.load_model("RF_SMOTE_bin.sav")
rfc_multi = utils.load_model("RF_SMOTE_multi.sav")
X_test, y_pred, y_label_map = evaluate_model(rfc_bin, 100, 0, 10, 0, binary_classif=True)