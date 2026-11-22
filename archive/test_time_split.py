
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix
)

from src.preprocess import (
    encode_features,
    NUMERIC_FEATURES
)


# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" /
    "PS_20174392719_1491204439457_log.csv"
)

MODEL_DIR = (
    ROOT / "models" /
    "experiments" / "time_split"
)

with open(
    MODEL_DIR / "metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

model = joblib.load(
    MODEL_DIR / "xgboost_time_1m.pkl"
)

scaler = joblib.load(
    MODEL_DIR / "scaler_time_1m.pkl"
)

threshold = float(metadata["threshold"])
test_start_step = int(metadata["validation_end_step"])

print("Testing time-based XGBoost model...")
print("Fixed threshold:", threshold)
print("Test transactions have step >", test_start_step)


# =====================================
# LOAD RESERVED TEST TRANSACTIONS
# =====================================

columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud"
]

df = pd.read_csv(
    DATA_PATH,
    usecols=columns,
    dtype={
        "step": "int32",
        "type": "category",
        "amount": "float64",
        "oldbalanceOrg": "float64",
        "oldbalanceDest": "float64",
        "isFraud": "int8"
    }
)

test_df = df.loc[
    df["step"] > test_start_step
].copy()

del df

if test_df.empty:
    raise ValueError("Reserved test set is empty.")

print("Test transactions:", len(test_df))
print("Actual fraud:", int(test_df["isFraud"].sum()))
print("Actual normal:", int((test_df["isFraud"] == 0).sum()))

# =====================================
# PREPROCESS WITH SAVED TRAINING SCALER
# =====================================

y_test = test_df["isFraud"].copy()

X_test = encode_features(test_df)

X_test = X_test.reindex(
    columns=metadata["feature_names"],
    fill_value=0
)

X_test[NUMERIC_FEATURES] = scaler.transform(
    X_test[NUMERIC_FEATURES]
)

# =====================================
# TEST PREDICTIONS
# =====================================

scores = model.predict_proba(X_test)[:, 1]

predictions = (
    scores >= threshold
).astype(int)

# =====================================
# PERFORMANCE METRICS
# =====================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

f2 = fbeta_score(
    y_test, predictions,
    beta=2,
    zero_division=0
)

ap = average_precision_score(
    y_test, scores
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

normal_baseline_accuracy = (
    (y_test == 0).mean()
)

print("\n====================================")
print("CHRONOLOGICAL TEST RESULTS")
print("====================================")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-Score:          {f1:.4f}")
print(f"F2-Score:          {f2:.4f}")
print(f"Average Precision: {ap:.4f}")

print(
    "Always-NORMAL accuracy:",
    f"{normal_baseline_accuracy:.4f}"
)

print("\nCONFUSION MATRIX")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

# =====================================
# ERRORS BY TRANSACTION TYPE
# =====================================

print("\nERROR ANALYSIS BY TYPE")
print("------------------------------------")

for transaction_type in [
    "CASH_OUT",
    "TRANSFER",
    "PAYMENT",
    "CASH_IN",
    "DEBIT"
]:
    mask = (
        test_df["type"].to_numpy()
        == transaction_type
    )

    type_tn, type_fp, type_fn, type_tp = (
        confusion_matrix(
            y_test.to_numpy()[mask],
            predictions[mask],
            labels=[0, 1]
        ).ravel()
    )

    print(
        f"{transaction_type}: "
        f"TP={type_tp}, "
        f"FN={type_fn}, "
        f"FP={type_fp}, "
        f"TN={type_tn}"
    )

# =====================================
# SAVE SEPARATE TEST RESULTS
# =====================================

results = {
    "evaluation": "Chronological test",
    "test_set_size": len(y_test),
    "threshold": threshold,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "f2_score": float(f2),
    "average_precision": float(ap),
    "always_normal_accuracy": float(
        normal_baseline_accuracy
    ),
    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
}

with open(
    MODEL_DIR / "test_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

print("\nResults saved to:")
print(MODEL_DIR / "test_results.json")
print("\nBaseline deployment unchanged.")
