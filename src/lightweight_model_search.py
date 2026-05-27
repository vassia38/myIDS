import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import utils


def load_dataset(benign_nrows, malicious_nrows, binary=False):
    df = utils.load_data(
        r"data\NF-UQ-NIDS-v2_Benign.csv",
        r"data\malicious",
        benign_nrows,
        0,
        malicious_nrows,
        0,
        binary
    )
    if binary:
        X = df.drop('Label', axis=1).values
        y = df['Label'].values
    else:
        X = df.drop('Attack', axis=1).values
        y, label_map = utils.encode_labels(df['Attack'].values)
        y = np.array(y)
    return X, y


def evaluate_parameters(X, y, params, resample=False):
    if resample:
        samp_str = 'auto'
        X, y = SMOTE(sampling_strategy=samp_str).fit_resample(X, y)

    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features=params['max_features'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean(), scores.std(), model


if __name__ == '__main__':
    sizes = [
        (500, 20),
        (1000, 30),
        (2000, 50),
    ]
    param_grid = [
        {'n_estimators': 50, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1},
        {'n_estimators': 50, 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 2},
        {'n_estimators': 100, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2},
        {'n_estimators': 150, 'max_depth': 30, 'max_features': 'log2', 'min_samples_leaf': 1},
    ]

    best = None
    best_score = 0.0
    print('Running lightweight model search...')

    for benign_nrows, malicious_nrows in sizes:
        print(f'\nTraining sizes: benign={benign_nrows}, malicious per file={malicious_nrows}')
        X, y = load_dataset(benign_nrows, malicious_nrows, binary=False)
        print(f'Loaded dataset shape: {X.shape}, labels: {np.unique(y, return_counts=True)}')

        for params in param_grid:
            mean_score, std_score, model = evaluate_parameters(X, y, params, resample=True)
            print(f"params={params} -> accuracy={mean_score:.4f} (+/- {std_score:.4f})")
            if mean_score > best_score:
                best_score = mean_score
                best = {
                    'params': params,
                    'score': mean_score,
                    'std': std_score,
                    'train_size': (benign_nrows, malicious_nrows),
                }

    print('\nBest model:')
    print(best)

    if best is not None:
        best_X, best_y = load_dataset(best['train_size'][0], best['train_size'][1], binary=False)
        best_model = RandomForestClassifier(
            n_estimators=best['params']['n_estimators'],
            max_depth=best['params']['max_depth'],
            max_features=best['params']['max_features'],
            min_samples_leaf=best['params']['min_samples_leaf'],
            random_state=42,
            n_jobs=-1,
        )
        best_model.fit(*SMOTE(sampling_strategy='auto').fit_resample(best_X, best_y))
        filename = 'models/best_lightweight_multiclass_rfc.sav'
        utils.save_model(best_model, filename)
        print(f'Saved best model to {filename}')
