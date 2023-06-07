from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

import sklearn.model_selection as skms
import sklearn.metrics as skmetric
import sklearn.preprocessing as skpre
from sklearn.pipeline import make_pipeline
import scikitplot as skplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import utils

def prepare_data(benign_size=10000, benign_skip_n=0, malicious_size=2500, malicious_skip_n=0):
    df = utils.load_data("NF-UQ-NIDS-v2_benign.csv", "NF-UQ-NIDS-v2_malicious.csv",
                         benign_size, benign_skip_n,
                         malicious_size, malicious_skip_n)
    df.drop(inplace=True, columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "Dataset", "Label"])
    # print("input sparsity ratio:{:.3f}".format(utils.get_sparsity_ratio(df)))
    
    X = df.drop('Attack', axis="columns")
    y = pd.DataFrame(df['Attack'])
    y_values, y_label_map = utils.encode_values(y['Attack'].values)
    y['Attack'] = y_values
    y = y['Attack']
    
    print("\n")
    print(pd.DataFrame(y)['Attack'].value_counts())
    print(y_label_map)
    print("\n")
    return X, y, y_label_map

def get_best_model_from_cross_val(benign_size=10000, benign_skip_n=0, 
                                  malicious_size=2500, malicious_skip_n=0):
    X,y, y_label_map = prepare_data(benign_size, benign_skip_n,
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
    X_test, y_test, y_label_map = prepare_data(benign_size, benign_skip_n,
                       malicious_size, malicious_skip_n)
    y_predicted = model.predict(X_test)
    acc = skmetric.accuracy_score(y_predicted, y_test)
    print(acc)
    # fig_matrix = plt.figure(figsize=(15,10))
    # ax_matrix = fig_matrix.add_subplot()
    # skplot.metrics.plot(y_test, y_predicted, 
    #                     labels=list(y_label_map.values()),
    #                     ax=ax_matrix)
    # plt.show()

def big_data_train():
    benign_train_size = 1000000
    malicious_train_size = 500000
    rfc = get_best_model_from_cross_val(benign_train_size, 0, malicious_train_size, 0)
    utils.save_model(rfc)
    
benign_train_size = 1000000
malicious_train_size = 2000000

rfc = utils.load_model()
evaluate_model(rfc, 25, benign_train_size, 1000, malicious_train_size)
