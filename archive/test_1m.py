
from pathlib import Path
from shutil import copy2
import json

import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix
)

from src.preprocess import prepare_data

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
EXPERIMENTS = MODELS / "experiments"

model = joblib.load(
    EXPERIMENTS / "xgboost_1m.pkl"
)

scaler_backup = (
    MODELS / "baseline_backup" / "scaler.pkl"
)

with open(
    EXPERIMENTS / "xgboost_1m_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

threshold = metadata["threshold"]

print("Loading test dataset...")

if not scaler_backup.exists():
    raise FileNotFoundError(
        "Baseline scaler backup not found."
    )

# prepare_data overwrites models/scaler.pkl.
# Restore the original after preparing the split.
try:
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = prepare_data()
finally:
    copy2(scaler_backup, MODELS / "scaler.pkl")

del X_train, X_val, y_train, y_val

# Ensure preprocessing matches saved experiment.
experiment_scaler = joblib.load(
    EXPERIMENTS / "scaler_1m.pkl"
)

# X_test was scaled by prepare_data().
# Its scaler should match the experimental scaler.
import numpy as np
original_scaler = joblib.load(scaler_backup)

if not (
    np.allclose(
        original_scaler.mean_,
        experiment_scaler.mean_
    )
    and np.allclose(
        original_scaler.scale_,
        experiment_scaler.scale_
    )
):
    raise ValueError(
        "Scaler mismatch. Test cannot be trusted."
    )

X_test = X_test[
    metadata["feature_names"]
]

print("Testing 1M model...")
print("Test rows:", len(X_test))
print("Threshold:", threshold)

scores = model.predict_proba(X_test)[:, 1]
predictions = (scores >= threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
f2 = fbeta_score(y_test, predictions, beta=2)
ap = average_precision_score(y_test, scores)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

print("\n==============================")
print("1M MODEL TEST RESULTS")
print("==============================")

print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"F2 Score: {f2:.4f}")
print(f"Average Precision: {ap:.4f}")

print("\nConfusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

results = {
    "model": "XGBoost 1M",
    "threshold": float(threshold),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "f2_score": float(f2),
    "average_precision": float(ap),
    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    },
    "evaluation_note": (
        "Random-split test set previously inspected "
        "during project development."
    )
}

with open(
    EXPERIMENTS / "xgboost_1m_test_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

print("\nTest results saved.")
print("Existing deployed model unchanged.")
