from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

import utils
import sklearn.model_selection as skms
import sklearn.metrics as skmetric
import sklearn.preprocessing as skpre
from sklearn.pipeline import make_pipeline
import scikitplot as skplot
import matplotlib.pyplot as plt
import pandas as pd
import time

def evaluateModel(X, y, model, scaler=skpre.StandardScaler(with_mean=False, with_std=False)):
    k = 5
    kf = skms.KFold(n_splits=k, random_state=None)

    acc_score = []
    fpr_score = []
    tpr_score = []
    
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        scale = scaler.fit(X_train)
        X_train = scale.transform(X_train)
        X_test = scale.transform(X_test)
        
        model.fit(X_train,y_train)
        y_predicted = model.predict(X_test)
        
        # acc = skmetric.accuracy_score(y_predicted, y_test)
        TN, FP, FN, TP = skmetric.confusion_matrix(y_test, y_predicted, labels=[0, 1]).ravel()
        acc = (TP+TN)/(TP+FP+FN+TN)
        fpr = FP/(FP+TN)
        tpr = TP/(TP+FN)
        acc_score.append(acc)
        fpr_score.append(fpr)
        tpr_score.append(tpr)
        
    # print('accuracy of each fold - {}'.format(acc_score))
    return {
        "model":    model,
        "scaler":   scaler,
        "acc":      sum(acc_score)/k,
        "tpr":      sum(tpr_score)/k,
        "fpr":      sum(fpr_score)/k
        }

df = utils.load_data("NF-UQ-NIDS-v2_benign.csv", "NF-UQ-NIDS-v2_malicious.csv", 10000, 2500)
df.drop(inplace=True, columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "Dataset", "Attack"])
# print("input sparsity ratio:{:.3f}".format(utils.get_sparsity_ratio(df)))
X = df.iloc[:,:-1]
y = df.iloc[:, -1]

scalers = {
    "raw": skpre.StandardScaler(with_mean=False, with_std=False), # adica nu se aplica
    "standardized": skpre.StandardScaler(),
    "normalized": skpre.MinMaxScaler(),
}
models_scalers = [
    [RandomForestClassifier(),
     scalers["raw"]],
]   

fig_matrix = plt.figure(figsize=(15,10))
fig_learning = plt.figure(figsize=(15,10))
index = 1

for [m,s] in models_scalers:
    print("evaluating {} with {}".format(m, s))
    estim = make_pipeline(s, m)
    
    start = time.time()
    # result = evaluateModel(X, y, m, s)
    # print("ACC: {} TPR: {} FPR: {}".format( result["acc"], result["tpr"], result["fpr"]))
    ax_learning = fig_learning.add_subplot(2,2,index)
    skplot.estimators.plot_learning_curve(estim, X,y, cv=5, scoring="accuracy", shuffle=True, 
                                            n_jobs=-1, figsize=(6,4), title="{} {}".format(m,s),
                                            ax=ax_learning)
    stop = time.time()
    print("elapsed time: {}".format(stop - start))
    
    predictions = cross_val_predict(estim, X, y)
    ax_matrix = fig_matrix.add_subplot(2,2,index)
    skplot.metrics.plot_confusion_matrix(y, predictions, normalize=True, 
                                        figsize=(6,4), title="{} {}".format(m,s),
                                        ax=ax_matrix)
    
    index += 1
    
plt.show()