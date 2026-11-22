
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" /
    "PS_20174392719_1491204439457_log.csv"
)

print("Loading required PaySim columns...")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "isFraud"
    ]
)

# Previously established chronological
# validation period.
validation = df[
    (df["step"] > 281) &
    (df["step"] <= 355)
].copy()

del df

print("Validation rows:", len(validation))

# =========================================
# SIMPLE RULE-BASED CLASSIFIER
# =========================================

predictions = (
    validation["type"].isin(
        ["TRANSFER", "CASH_OUT"]
    )
    &
    (
        validation["amount"]
        == validation["oldbalanceOrg"]
    )
    &
    (
        validation["oldbalanceOrg"] > 0
    )
).astype(int)

actual = validation["isFraud"]

precision = precision_score(
    actual,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual,
    predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    actual,
    predictions,
    labels=[0, 1]
).ravel()

print("\n================================")
print("SIMPLE RULE VS XGBOOST")
print("================================")

print("\nSimple Rule:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))

print("\nConfusion Matrix:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nExisting XGBoost Validation:")
print("Precision: 0.9747")
print("Recall:    1.0000")
print("F1:        0.9872")

print("\nRule audit completed.")
print("No model files modified.")
