
from pathlib import Path
from shutil import copy2

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.preprocess import prepare_data

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models/experiments/xgboost_1m.pkl"
BASELINE_SCALER = ROOT / "models/baseline_backup/scaler.pkl"
CURRENT_SCALER = ROOT / "models/scaler.pkl"
EXPERIMENT_SCALER = ROOT / "models/experiments/scaler_1m.pkl"

if not BASELINE_SCALER.exists():
    raise FileNotFoundError(
        "Baseline scaler backup is missing."
    )

print("Loading validation data...")

# Existing prepare_data() overwrites scaler.pkl.
# Restore the original scaler even if loading fails.
try:
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = prepare_data()
finally:
    copy2(BASELINE_SCALER, CURRENT_SCALER)

del X_train, X_test, y_train, y_test

# Confirm scaler compatibility.
baseline = joblib.load(BASELINE_SCALER)
experiment = joblib.load(EXPERIMENT_SCALER)

if not (
    np.allclose(baseline.mean_, experiment.mean_)
    and np.allclose(baseline.scale_, experiment.scale_)
):
    raise ValueError(
        "Baseline and experimental scalers differ."
    )

model = joblib.load(MODEL_PATH)

scores = model.predict_proba(X_val)[:, 1]

print("\nTHRESHOLD AUDIT")
print("-" * 65)

for threshold in [0.95, 0.99, 0.995, 0.999, 0.9999]:
    predictions = (scores >= threshold).astype(int)

    print(
        f"Threshold={threshold:.4f} | "
        f"Precision={precision_score(y_val, predictions):.4f} | "
        f"Recall={recall_score(y_val, predictions):.4f} | "
        f"F1={f1_score(y_val, predictions):.4f}"
    )

# Use original selected threshold for error analysis.
predictions = (scores >= 0.99).astype(int)

audit = pd.DataFrame({
    "actual": y_val.to_numpy(),
    "predicted": predictions,
    "score": scores,
    "type": "OTHER"
})

for transaction_type in ["CASH_OUT", "TRANSFER"]:
    column = f"type_{transaction_type}"

    audit.loc[
        X_val[column].to_numpy() == 1, "type"
    ] = transaction_type

print("\nERROR ANALYSIS BY TRANSACTION TYPE")
print("-" * 65)

for transaction_type in ["CASH_OUT", "TRANSFER", "OTHER"]:

    subset = audit[audit["type"] == transaction_type]

    tn, fp, fn, tp = confusion_matrix(
        subset["actual"],
        subset["predicted"],
        labels=[0, 1]
    ).ravel()

    print(
        f"{transaction_type}: "
        f"TP={tp}, FN={fn}, FP={fp}, TN={tn}"
    )

print("\nMISSED FRAUD CASES")
print(
    audit[
        (audit["actual"] == 1) &
        (audit["predicted"] == 0)
    ].to_string(index=False)
)

print("\nAudit completed.")
print("Baseline model artifacts preserved.")
