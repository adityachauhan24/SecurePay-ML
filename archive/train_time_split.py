
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from src.preprocess import (
    encode_features,
    NUMERIC_FEATURES
)

# =========================================
# CONFIGURATION
# =========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" /
    "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = (
    ROOT / "models" /
    "experiments" / "time_split"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42

# Read only features needed by the model.
COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud"
]

print("Loading PaySim data...")

df = pd.read_csv(
    DATA_PATH,
    usecols=COLUMNS,
    dtype={
        "step": "int32",
        "type": "category",
        "amount": "float64",
        "oldbalanceOrg": "float64",
        "oldbalanceDest": "float64",
        "isFraud": "int8"
    }
)

# =========================================
# CHRONOLOGICAL SPLIT
# =========================================

print("Creating chronological partitions...")

counts = df["step"].value_counts().sort_index()
cumulative = counts.cumsum().to_numpy()
steps = counts.index.to_numpy()

n = len(df)

cut1 = steps[np.searchsorted(cumulative, n * 0.60)]
cut2 = steps[np.searchsorted(cumulative, n * 0.80)]

# Entire time steps remain in one partition.
train_df = df.loc[df["step"] <= cut1]
val_df = df.loc[
    (df["step"] > cut1) &
    (df["step"] <= cut2)
]

test_count = int((df["step"] > cut2).sum())

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Reserved test rows:", test_count)

print("Train ends at step:", cut1)
print("Validation ends at step:", cut2)

print("Training fraud cases:", int(train_df["isFraud"].sum()))
print("Validation fraud cases:", int(val_df["isFraud"].sum()))

# Avoid processing the reserved test set.
del df

if len(train_df) < SAMPLE_SIZE:
    raise ValueError(
        "Not enough training rows for 1M sample."
    )

# =========================================
# STRATIFIED TRAINING SAMPLE
# =========================================

print("\nSampling 1M training records...")

train_sample, _ = train_test_split(
    train_df,
    train_size=SAMPLE_SIZE,
    stratify=train_df["isFraud"],
    random_state=RANDOM_STATE
)

del train_df

y_train = train_sample["isFraud"].copy()
y_val = val_df["isFraud"].copy()

X_train = encode_features(train_sample)
X_val = encode_features(val_df)

del train_sample, val_df

# =========================================
# TRAIN-ONLY SCALER
# =========================================

print("Fitting scaler on training sample...")

scaler = StandardScaler()

X_train[NUMERIC_FEATURES] = scaler.fit_transform(
    X_train[NUMERIC_FEATURES]
)

X_val[NUMERIC_FEATURES] = scaler.transform(
    X_val[NUMERIC_FEATURES]
)

# =========================================
# XGBOOST TRAINING
# =========================================

fraud_count = int(y_train.sum())
normal_count = len(y_train) - fraud_count

scale_weight = normal_count / fraud_count

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="logloss",
    scale_pos_weight=scale_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining time-based XGBoost...")

model.fit(X_train, y_train)

print("Training completed.")

# =========================================
# VALIDATION EVALUATION
# =========================================

scores = model.predict_proba(X_val)[:, 1]

thresholds = [
    0.50, 0.70, 0.80,
    0.90, 0.95, 0.99, 0.995
]

best_threshold = None
best_f1 = -1
best_precision = 0
best_recall = 0

for threshold in thresholds:
    preds = (scores >= threshold).astype(int)

    precision = precision_score(
        y_val, preds, zero_division=0
    )
    recall = recall_score(
        y_val, preds, zero_division=0
    )
    f1 = f1_score(
        y_val, preds, zero_division=0
    )

    print(
        f"Threshold={threshold:.3f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )

    if f1 > best_f1:
        best_threshold = threshold
        best_f1 = float(f1)
        best_precision = float(precision)
        best_recall = float(recall)

best_preds = (scores >= best_threshold).astype(int)

ap = average_precision_score(y_val, scores)

cm = confusion_matrix(
    y_val, best_preds, labels=[0, 1]
)

print("\n====================================")
print("TIME-BASED VALIDATION RESULTS")
print("====================================")

print("Best threshold:", best_threshold)
print("Precision:", round(best_precision, 4))
print("Recall:", round(best_recall, 4))
print("F1:", round(best_f1, 4))
print("Average Precision:", round(ap, 4))
print("Confusion Matrix:")
print(cm)

# =========================================
# SAVE EXPERIMENT
# =========================================

joblib.dump(
    model,
    OUTPUT_DIR / "xgboost_time_1m.pkl"
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "scaler_time_1m.pkl"
)

metadata = {
    "model_name": "XGBoost Time-Based 1M",
    "training_sample_size": SAMPLE_SIZE,
    "training_fraud_cases": fraud_count,
    "train_end_step": int(cut1),
    "validation_end_step": int(cut2),
    "threshold": float(best_threshold),
    "validation_precision": best_precision,
    "validation_recall": best_recall,
    "validation_f1": best_f1,
    "validation_average_precision": float(ap),
    "feature_names": X_train.columns.tolist(),
    "test_evaluated": False
}

with open(
    OUTPUT_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\nTime-based experiment saved.")
print("Current deployed model remains unchanged.")
