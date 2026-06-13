"""
Lightweight binary classification model optimized for Go transpilation via m2cgen.
Uses reduced tree depth and fewer estimators for smaller compiled output.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import utils

BINARY_MODEL_FILENAME = "models/NDR_binary_filter_lightweight.sav"


def load_binary_dataset(benign_nrows=10000, malicious_nrows=2500,
                        benign_skip_n=0, malicious_skip_n=0):
    """Load binary dataset: 0 = Benign, 1 = Malicious/Suspicious."""
    df = utils.load_data(
        r"data\NF-UQ-NIDS-v2_Benign.csv",
        r"data\malicious",
        benign_nrows,
        benign_skip_n,
        malicious_nrows,
        malicious_skip_n,
        True,
    )
    X = df.drop(columns=["Label"])
    y = df["Label"].astype(int).values
    return X, y


def balance_binary_dataset(X, y, random_state=42):
    """Resample the binary dataset with SMOTE to reduce class imbalance."""
    sampler = SMOTE(random_state=random_state)
    return sampler.fit_resample(X, y)


def build_lightweight_binary_model(random_state=42):
    """Build a balanced RF model for Go transpilation (65 trees, depth 16)."""
    return RandomForestClassifier(
        n_estimators=65,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def cross_validate_binary(X, y, model=None, cv=5):
    if model is None:
        model = build_lightweight_binary_model()
    scoring = ["accuracy", "precision", "recall", "f1"]
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    result = cross_validate(model, X, y, cv=cv_split, scoring=scoring, n_jobs=-1)
    return {k.replace("test_", ""): np.mean(v) for k, v in result.items() if k.startswith("test_")}


def evaluate_binary_model(model, X, y):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=["Benign", "Malicious"], zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return report, cm


def train_binary_filter_lightweight(benign_nrows=10000, malicious_nrows=2500,
                                    resample=True, save_path=BINARY_MODEL_FILENAME):
    X, y = load_binary_dataset(benign_nrows, malicious_nrows)
    if resample:
        X, y = balance_binary_dataset(X, y)
    model = build_lightweight_binary_model()
    model.fit(X, y)
    utils.save_model(model, save_path)
    return model


def predict_binary_filter(model, X):
    return model.predict(X)


def predict_binary_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError("Model does not support predict_proba")


if __name__ == "__main__":
    X, y = load_binary_dataset()
    print("Loaded binary dataset shape:", X.shape)
    print("Class counts:", np.bincount(y))

    X_resampled, y_resampled = balance_binary_dataset(X, y)
    print("Resampled shape:", X_resampled.shape)
    print("Resampled class counts:", np.bincount(y_resampled))

    print("Performing cross-validation on the balanced dataset...")
    cv_scores = cross_validate_binary(X_resampled, y_resampled)
    print("CV scores:", cv_scores)

    print("Training lightweight binary filter model and saving to", BINARY_MODEL_FILENAME)
    model = train_binary_filter_lightweight()

    report, cm = evaluate_binary_model(model, X_resampled, y_resampled)
    print(report)
    print("Confusion matrix:\n", cm)
