from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import utils
from plotting import *
import numpy as np
from config import *


def prepare_data(benign_size=10000, benign_skip_n=0, malicious_size=2500, malicious_skip_n=0, binary_classif=False):
    df = utils.load_data(r"data\NF-UQ-NIDS-v2_Benign.csv", r"data\malicious",
                         benign_size, benign_skip_n,
                         malicious_size, malicious_skip_n,
                         binary_classif)
    
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
                                  malicious_size=1000, malicious_skip_n=0,
                                  binary_classif=False, resample=False, tuning=False):
    X, feature_names, y, label_map = prepare_data(benign_size, benign_skip_n,
                                                  malicious_size, malicious_skip_n,
                                                  binary_classif)
    if resample:
        X, y = resample_dataset(X, y, binary_classif, label_map, benign_size, malicious_size)
        print(pd.DataFrame(y).value_counts())

    rfc = RandomForestClassifier(n_jobs=-1)

    if tuning:
    #  Hyperparameters tuning
        random_grid = utils.get_RandomForestHyperparams_Grid()
        rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, 
                                       n_iter = 100, cv = 5, verbose=2, random_state=42, 
                                       n_jobs = -1, return_train_score=True)
        result = rf_random.fit(X,y)
        print(result)
        return rf_random.best_estimator_

    result = cross_validate(rfc, X, y, cv=5, return_estimator=True, return_train_score=True)
    print(result)
    
    best_estim_index = 0
    for i in range(len(result['test_score'])):
        if result['test_score'][i] > result['test_score'][best_estim_index]:
            best_estim_index = i
    
    return result['estimator'][best_estim_index]


def evaluate_model(model, benign_size=40, benign_skip_n=0, 
                   malicious_size=60, malicious_skip_n=0,
                   binary_classif=False, resample=False):
    X, feature_names, y, label_map = prepare_data(benign_size, benign_skip_n,
                                                                malicious_size, malicious_skip_n,
                                                                binary_classif)
    
    if resample:
        X, y = resample_dataset(X, y, binary_classif, label_map, benign_size, malicious_size)

    y_predicted = model.predict(X)
    acc = accuracy_score(y_predicted, y)
    print("{:.5f}".format(acc))

    print_top_features(model, feature_names)
    show_feature_importance(model, feature_names)
    show_confusion_matrix(y, y_predicted)
    return X, y_predicted, label_map


def print_top_features(estimator, feature_names, top_n=15):
    if not hasattr(estimator, 'feature_importances_'):
        print('Top features cannot be displayed: estimator has no feature_importances_.')
        return

    importances = estimator.feature_importances_
    feature_names = np.array(feature_names)
    indices = np.argsort(importances)[::-1][:top_n]

    print('\nTop features (by importance):')
    for rank, idx in enumerate(indices, start=1):
        print(f"{rank:2d}. {feature_names[idx]} = {importances[idx]:.6f}")


def resample_dataset(X, y, binary_classif, label_map, benign_size, malicious_size):
    if binary_classif:
        return SMOTE(sampling_strategy='auto').fit_resample(X, y)

    counts = pd.Series(y).value_counts().to_dict()
    samp_str = {}
    for key in label_map.keys():
        target = benign_size if label_map[key] == 'Benign' else malicious_size
        current = counts.get(key, 0)
        samp_str[key] = max(current, target)

    return SMOTE(sampling_strategy=samp_str).fit_resample(X, y)


def search_lightweight_models(train_sizes=None, param_grid=None, resample=True, cv_splits=3):
    if train_sizes is None:
        train_sizes = [
            (500, 20),
            (1000, 30),
            (2000, 50),
        ]

    if param_grid is None:
        param_grid = [
            {'n_estimators': 50, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1},
            {'n_estimators': 50, 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 2},
            {'n_estimators': 100, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1},
            {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2},
            {'n_estimators': 150, 'max_depth': 30, 'max_features': 'log2', 'min_samples_leaf': 1},
        ]

    best = {
        'score': 0.0,
        'std': 0.0,
        'params': None,
        'train_size': None,
    }

    for benign_size, malicious_size in train_sizes:
        print(f"\nSearching with benign={benign_size}, malicious_per_file={malicious_size}")
        X, feature_names, y, label_map = prepare_data(benign_size, 0, malicious_size, 0, False)

        if resample:
            X, y = resample_dataset(X, y, False, label_map, benign_size, malicious_size)

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        for params in param_grid:
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                max_features=params['max_features'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                n_jobs=-1,
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"params={params} -> accuracy={mean_score:.4f} (+/- {std_score:.4f})")
            if mean_score > best['score']:
                best.update({
                    'score': mean_score,
                    'std': std_score,
                    'params': params,
                    'train_size': (benign_size, malicious_size),
                })

    print("\nBest lightweight model:")
    print(best)
    return best


def save_best_lightweight_model(best, filename='models/best_lightweight_multiclass_rfc.sav'):
    if best is None or best['params'] is None:
        raise ValueError('No best model found to save.')

    benign_size, malicious_size = best['train_size']
    X, feature_names, y, label_map = prepare_data(benign_size, 0, malicious_size, 0, False)
    X, y = resample_dataset(X, y, False, label_map, benign_size, malicious_size)

    best_model = RandomForestClassifier(
        n_estimators=best['params']['n_estimators'],
        max_depth=best['params']['max_depth'],
        max_features=best['params']['max_features'],
        min_samples_leaf=best['params']['min_samples_leaf'],
        random_state=42,
        n_jobs=-1,
    )
    best_model.fit(X, y)
    utils.save_model(best_model, filename)
    return best_model


def make_binary_rfc():
    benign_train_size = 20000
    malicious_train_size = 2500
    binary = True
    resample = malicious_train_size / benign_train_size

    rfc = get_best_model_from_cross_val(benign_train_size, 0, malicious_train_size, 0, binary, resample)
    utils.save_model(rfc, "models/RF_SMOTE_bin.sav")
    evaluate_model(rfc, 50, benign_train_size, 50, malicious_train_size, binary)


def make_multic_rfc():
    benign_train_size = 10000
    malicious_train_size = 1000
    resample = True
    rfc = get_best_model_from_cross_val(benign_train_size, 0, malicious_train_size, 0, False, resample, tuning=False)
    #utils.save_model(rfc, "models/RF_SMOTE_multi_tuned.sav")
    evaluate_model(rfc, 1000, benign_train_size, 1000, malicious_train_size, False, resample)

#make_binary_rfc()
#make_multic_rfc()

if __name__ == '__main__':
    make_multic_rfc()
    # To run the lightweight search and save the best model, uncomment these lines:
    best = search_lightweight_models()
    save_best_lightweight_model(best)

# rfc_bin = utils.load_model("models/RF_SMOTE_bin.sav")
# rfc_multi = utils.load_model("models/RF_SMOTE_multi.sav")
#X_test, y_pred, y_label_map = evaluate_model(rfc_multi, 10000, 10000, 1000, 1000, binary_classif=False)