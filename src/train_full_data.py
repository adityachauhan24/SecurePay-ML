
from pathlib import Path
from shutil import copy2
import json

import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from src.preprocess import prepare_data


# ======================================
# CONFIGURATION
# ======================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"

EXPERIMENT_DIR = MODEL_DIR / "experiments"

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BASELINE_SCALER = (
    MODEL_DIR / "baseline_backup" / "scaler.pkl"
)

CURRENT_SCALER = MODEL_DIR / "scaler.pkl"

NEW_SCALER = EXPERIMENT_DIR / "scaler_full.pkl"

MODEL_PATH = (
    EXPERIMENT_DIR / "xgboost_full.pkl"
)

METADATA_PATH = (
    EXPERIMENT_DIR / "xgboost_full_metadata.json"
)


# ======================================
# CHECK BASELINE BACKUP
# ======================================

if not BASELINE_SCALER.exists():
    raise FileNotFoundError(
        "Baseline scaler backup missing. "
        "Please back up models/scaler.pkl first."
    )


# ======================================
# LOAD AND PREPROCESS FULL DATA
# ======================================

print("\nLoading PaySim dataset...")

# prepare_data() generates scaler.pkl.
# Preserve the scaler used by the dashboard.

try:
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_data()

    copy2(CURRENT_SCALER, NEW_SCALER)

finally:
    copy2(BASELINE_SCALER, CURRENT_SCALER)


print("\n================================")
print("DATASET DETAILS")
print("================================")

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Reserved test rows:", len(X_test))

# Keep the test set out of this experiment.
del X_test, y_test


# ======================================
# CLASS IMBALANCE
# ======================================

fraud_count = int(y_train.sum())

normal_count = len(y_train) - fraud_count

scale_weight = normal_count / fraud_count

print("\nFraud transactions:", fraud_count)
print("Normal transactions:", normal_count)
print("Scale pos weight:", scale_weight)


# ======================================
# TRAIN XGBOOST ON ALL TRAINING ROWS
# ======================================

print("\nTraining XGBoost on full training split...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="logloss",
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nTraining completed successfully!")


# ======================================
# VALIDATION PREDICTIONS
# ======================================

print("\nEvaluating on validation data...")

probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [
    0.30,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99
]

best_threshold = None
best_f1 = -1
best_precision = 0
best_recall = 0

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    })

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = float(f1)
        best_threshold = float(threshold)
        best_precision = float(precision)
        best_recall = float(recall)


# ======================================
# FINAL VALIDATION RESULTS
# ======================================

best_predictions = (
    probabilities >= best_threshold
).astype(int)

ap = average_precision_score(
    y_val,
    probabilities
)

cm = confusion_matrix(
    y_val,
    best_predictions,
    labels=[0, 1]
)

print("\n================================")
print("FULL XGBOOST RESULTS")
print("================================")

print("Training rows:", len(X_train))
print("Best threshold:", best_threshold)
print("Precision:", round(best_precision, 4))
print("Recall:", round(best_recall, 4))
print("F1 Score:", round(best_f1, 4))
print("Average Precision:", round(ap, 4))

print("\nConfusion Matrix:")
print(cm)

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print("300K validation F1: 0.8666")
print("1M validation F1:   0.9546")
print("Full validation F1:", round(best_f1, 4))


# ======================================
# SAVE NEW MODEL
# ======================================

print("\nSaving trained model...")

joblib.dump(
    model,
    MODEL_PATH
)

metadata = {
    "model_name": "SecurePay ML Full XGBoost",
    "training_sample_size": len(X_train),
    "fraud_training_rows": fraud_count,
    "normal_training_rows": normal_count,
    "scale_pos_weight": float(scale_weight),
    "threshold": best_threshold,
    "feature_names": X_train.columns.tolist(),
    "validation_precision": best_precision,
    "validation_recall": best_recall,
    "validation_f1": best_f1,
    "validation_average_precision": float(ap),
    "threshold_results": results,
    "test_set_evaluated": False
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )

print("\nModel saved successfully!")
print("Model:", MODEL_PATH)
print("Scaler:", NEW_SCALER)
print("Metadata:", METADATA_PATH)

print("\nOriginal dashboard model unchanged.")
