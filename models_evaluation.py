from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import utils
import sklearn.model_selection as skms
import sklearn.metrics as skmetric
import sklearn.preprocessing as skpre
from sklearn.pipeline import make_pipeline
import scikitplot as skplot
import matplotlib.pyplot as plt
import pandas as pd


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

def evaluate_models(models_scalers, X, y):
    fig, ax = plt.subplots(2,2, figsize=(30, 10))
    i = j = 0

    for [m,s] in models_scalers:
        print("\nevaluating {} model with {}".format(m, s))
        result = evaluateModel(X, y, m, s)
        m = result["model"]
        s = result["scaler"]
        acc = result["acc"]
        tpr = result["tpr"]
        fpr = result["fpr"]
        print('Avg accuracy : {:.3f}'.format(acc))
        print('Avg TPR : {:.3f}'.format(tpr))
        print('Avg FPR : {:.3f}'.format(fpr))
        
        if "Logistic" in "{}".format(m):

            
            ax[i, j].bar(X.columns, pd.Series(m.coef_[0]))
            ax[i, j].set_title("{}".format(m))
            ax[i, j].spines["bottom"].set_position("zero")
            
        j += j == 0
        i += (j == 1 and i == 0)

    plt.show()

df = utils.load_data(r"data\NF-UQ-NIDS-v2_Benign.csv", r"data\malicious", 10000, 0, 1000, 0, True)
# print("input sparsity ratio:{:.3f}".format(utils.get_sparsity_ratio(df)))

X = df.iloc[:,:-1]
y = df.iloc[:, -1]

scalers = {
    "raw": skpre.StandardScaler(with_mean=False, with_std=False), # adica nu se aplica
    "standardized": skpre.StandardScaler(),
    "normalized": skpre.MinMaxScaler(),
}

models_scalers = [
    [LogisticRegression(solver="saga", penalty='l1', max_iter=600, n_jobs=-1),
     scalers["normalized"]],
    
    [BernoulliNB(),
     scalers["standardized"]],
    
    [DecisionTreeClassifier(),
     scalers["raw"]],

    [RandomForestClassifier(n_jobs=-1),
     scalers["raw"]],

    [SVC(),
     scalers["normalized"]],
]   

fig_learning = plt.figure(figsize=(15,12))
fig_importance = plt.figure(figsize=(15,12))
index = 1
for [m,s] in models_scalers:
    print("\nevaluating {} model with {}".format(m, s))
    estim = make_pipeline(s, m)
    ax_learning = fig_learning.add_subplot(3,3,index)
    skplot.estimators.plot_learning_curve(estim, X,y, cv=5, scoring="accuracy", shuffle=True, 
                                          n_jobs=-1, figsize=(6,4), title="{} {}".format(m,s), ax=ax_learning)
    try:
        ax_importance = fig_importance.add_subplot(3,3,index)
        skplot.estimators.plot_feature_importances(estim, feature_names=X.columns,
                                         title="{} with {} Feature Importance".format(m,s),
                                         x_tick_rotation=90, order="ascending",
                                         ax=ax_importance)
    except:
        pass
    index += 1
plt.tight_layout()
plt.show()