"""
RF multiclass classifier: WITH SMOTE vs. WITHOUT SMOTE
=======================================================
Replicates the training setup from ndr_suspicious_classifier_lightweight.py
(same RF hyperparameters, same data sizes) but compares the two SMOTE choices
on a held-out test set drawn from the original (non-synthetic) data.

Key difference from the original script: the original evaluates on the
SMOTE-resampled training data, which gives an optimistic read because
synthetic samples are easy to classify. This script evaluates both models
on real held-out data so the comparison is fair.

Models saved:
  models/NDR_suspicious_classifier_lightweight_with_smote.sav
  models/NDR_suspicious_classifier_lightweight_no_smote.sav
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---- Paths (same relative layout as the original project) --------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_BENIGN  = REPO_ROOT / "data" / "NF-UQ-NIDS-v2_Benign.csv"
DATA_MAL_DIR = REPO_ROOT / "data" / "malicious"
MODELS_DIR   = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---- Exactly the 9 features used in production ------------------------------
FEATURES = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
]

# ---- Data sizes: identical to ndr_suspicious_classifier_lightweight.py ------
BENIGN_NROWS    = 10_000
MALICIOUS_NROWS = 2_000
WORMS_NROWS     = 100   # Worms has ~164 rows total; original script caps at 100


# ---- Data loading ------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load benign + per-class malicious CSVs, same row caps as the original script.
    Worms gets a hard cap of 100 rows (matches utils.py special-case logic).
    """
    parts = []

    df_b = pd.read_csv(DATA_BENIGN, nrows=BENIGN_NROWS, usecols=FEATURES + ["Attack"])
    parts.append(df_b)

    for path in sorted(DATA_MAL_DIR.iterdir()):
        if path.suffix.lower() != ".csv":
            continue
        nrows = WORMS_NROWS if "worms" in path.name.lower() else MALICIOUS_NROWS
        df_m = pd.read_csv(path, nrows=nrows, usecols=FEATURES + ["Attack"])
        parts.append(df_m)

    df = (pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[FEATURES] = df[FEATURES].astype(np.float32)
    return df


# ---- RF model: identical hyperparameters to ndr_suspicious_classifier_lightweight.py ---

def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=130,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


# ---- Evaluation on real (non-synthetic) test data ----------------------------

def evaluate(name: str, model, X_test, y_test) -> dict:
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_s = time.time() - t0

    acc        = accuracy_score(y_test, y_pred)
    macro_f1   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_test, y_pred, average="weighted", zero_division=0)
    us_per     = 1e6 * infer_s / len(X_test)

    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(f"  Accuracy       {acc:.4f}")
    print(f"  Macro F1       {macro_f1:.4f}")
    print(f"  Weighted F1    {weighted_f1:.4f}")
    print(f"  Inference      {us_per:.2f} us/sample")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "infer_us": us_per,
    }


# ---- Main --------------------------------------------------------------------

def main():
    print("Loading data...")
    df = load_data()
    X = df[FEATURES].values.astype(np.float32)
    y = df["Attack"].values

    print(f"  Total: {len(df):,} samples, {len(set(y))} classes")
    print("\n  Class distribution (train+test combined):")
    vc = pd.Series(y).value_counts().sort_values(ascending=False)
    for cls, cnt in vc.items():
        print(f"    {cls:<28} {cnt:>6,}")

    # Hold out 20 % as test set BEFORE any SMOTE — this stays real data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
    print("(Test set is never touched by SMOTE)")

    results = []

    # ------------------------------------------------------------------ WITH SMOTE
    print("\n" + "-"*62)
    print("  Training: RF WITH SMOTE  (replicates original script)")
    print("-"*62)

    smote = SMOTE(random_state=42)
    t0 = time.time()
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    smote_s = time.time() - t0

    print(f"  SMOTE: {len(X_train):,} -> {len(X_sm):,} samples  ({smote_s:.1f}s)")
    print(f"  Majority class size (= SMOTE target): {int(np.bincount(np.unique(y_sm, return_inverse=True)[1]).max()):,}")

    model_smote = build_model()
    t0 = time.time()
    model_smote.fit(X_sm, y_sm)
    train_s = time.time() - t0
    print(f"  RF training: {train_s:.1f}s")

    save_path = MODELS_DIR / "NDR_suspicious_classifier_lightweight_with_smote.sav"
    joblib.dump(model_smote, save_path)
    print(f"  Saved: {save_path.relative_to(REPO_ROOT)}")

    r = evaluate("RF + SMOTE (replicates original)", model_smote, X_test, y_test)
    r["train_s"] = train_s + smote_s
    results.append(r)

    # -------------------------------------------------------------- WITHOUT SMOTE
    print("\n" + "-"*62)
    print("  Training: RF WITHOUT SMOTE  (class_weight='balanced_subsample' only)")
    print("-"*62)

    model_nosmote = build_model()
    t0 = time.time()
    model_nosmote.fit(X_train, y_train)
    train_s = time.time() - t0
    print(f"  RF training: {train_s:.1f}s")

    save_path = MODELS_DIR / "NDR_suspicious_classifier_lightweight_no_smote.sav"
    joblib.dump(model_nosmote, save_path)
    print(f"  Saved: {save_path.relative_to(REPO_ROOT)}")

    r = evaluate("RF no SMOTE (balanced_subsample only)", model_nosmote, X_test, y_test)
    r["train_s"] = train_s
    results.append(r)

    # -------------------------------------------------------------------- Summary
    print("\n\n" + "="*72)
    print("  SUMMARY  (evaluated on real held-out test data, no synthetic samples)")
    print("="*72)
    print(f"  {'Model':<42} {'Macro F1':>9} {'Wt F1':>8} {'Acc':>7} {'Train(s)':>9}")
    print("  " + "-"*70)
    for r in results:
        print(f"  {r['name']:<42} {r['macro_f1']:>9.4f} {r['weighted_f1']:>8.4f} "
              f"{r['accuracy']:>7.4f} {r['train_s']:>9.1f}")
    print("="*72)

    print("\nInterpretation notes:")
    print("  - Macro F1 equally weights all 20 classes (penalises Worms, Analysis,")
    print("    Shellcode misses as much as DDoS misses).")
    print("  - Weighted F1 is dominated by the large classes (Benign, DDoS, DoS).")
    print("  - SMOTE upsamples ALL minority classes to the majority class count,")
    print("    including Worms (100 real samples -> ~10K synthetic ones).")
    print("  - class_weight='balanced_subsample' adjusts the RF split criterion")
    print("    per-tree bootstrap sample but does NOT create new data points.")


if __name__ == "__main__":
    main()
