"""
Lightweight multiclass attack classification model optimized for Go transpilation via m2cgen.
Uses reduced tree depth and fewer estimators for smaller compiled output.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import utils

SUSPICIOUS_MODEL_FILENAME = "models/NDR_suspicious_classifier_lightweight.sav"


def load_multiclass_dataset(benign_nrows=10000, benign_skip_n=0,
                            malicious_nrows=2000, malicious_skip_n=0):
    """Load benign and malicious data for multiclass attack classification."""
    df = utils.load_data(
        r"data\NF-UQ-NIDS-v2_Benign.csv",
        r"data\malicious",
        benign_nrows=benign_nrows,
        benign_skip_n=benign_skip_n,
        malicious_nrows=malicious_nrows,
        malicious_skip_n=malicious_skip_n,
        binary_classif=False,
    )
    X = df.drop(columns=["Attack"])
    y_raw = df["Attack"].values
    y, label_decoder = utils.encode_labels(y_raw)
    return X, np.asarray(y), label_decoder


def balance_multiclass_dataset(X, y, random_state=42):
    sampler = SMOTE(random_state=random_state)
    return sampler.fit_resample(X, y)


def build_lightweight_multiclass_model(random_state=42):
    """Build a balanced RF model for Go transpilation (130 trees, depth 16)."""
    return RandomForestClassifier(
        n_estimators=130,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def cross_validate_multiclass(X, y, model=None, cv=5):
    if model is None:
        model = build_lightweight_multiclass_model()
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    result = cross_validate(model, X, y, cv=cv_split, scoring=scoring, n_jobs=-1)
    return {k.replace("test_", ""): np.mean(v) for k, v in result.items() if k.startswith("test_")}


def train_multiclass_classifier_lightweight(benign_nrows=10000, malicious_nrows=2000,
                                            resample=True, save_path=SUSPICIOUS_MODEL_FILENAME):
    X, y, label_decoder = load_multiclass_dataset(benign_nrows, 0, malicious_nrows, 0)
    if resample:
        X, y = balance_multiclass_dataset(X, y)
    model = build_lightweight_multiclass_model()
    model.fit(X, y)
    utils.save_model(model, save_path)
    return model, label_decoder


def predict_attack_labels(model, X, label_decoder):
    label_ids = model.predict(X)
    return [label_decoder[int(label_id)] for label_id in label_ids]


def evaluate_multiclass_model(model, X, y, label_decoder):
    y_true = [label_decoder[int(label)] for label in y]
    y_pred = predict_attack_labels(model, X, label_decoder)
    report = classification_report(y_true, y_pred, zero_division=0)
    return report


def build_label_decoder():
    return {i: label for i, label in enumerate(utils.LABELS)}


def load_multiclass_model(path=SUSPICIOUS_MODEL_FILENAME):
    return utils.load_model(path)


def classify_batch(X, binary_model, multiclass_model, label_decoder):
    if hasattr(X, "iloc"):
        X_df = X
    else:
        X_df = X
    binary_pred = binary_model.predict(X_df)
    results = ["Benign"] * len(X_df)
    suspicious_indexes = np.where(binary_pred == 1)[0]
    if len(suspicious_indexes) == 0:
        return results
    X_suspicious = X_df.iloc[suspicious_indexes]
    suspicious_labels = predict_attack_labels(multiclass_model, X_suspicious, label_decoder)
    for idx, label in zip(suspicious_indexes, suspicious_labels):
        results[idx] = label
    return results


if __name__ == "__main__":
    X, y, label_decoder = load_multiclass_dataset()
    print("Loaded multiclass dataset shape:", X.shape)
    print("Unique attack classes:", len(set(y)))

    X_resampled, y_resampled = balance_multiclass_dataset(X, y)
    print("Resampled shape:", X_resampled.shape)

    print("Performing cross-validation on multiclass data...")
    cv_scores = cross_validate_multiclass(X_resampled, y_resampled)
    print("CV scores:", cv_scores)

    print("Training lightweight multiclass classifier and saving to", SUSPICIOUS_MODEL_FILENAME)
    model, label_decoder = train_multiclass_classifier_lightweight()

    report = evaluate_multiclass_model(model, X_resampled, y_resampled, label_decoder)
    print(report)
