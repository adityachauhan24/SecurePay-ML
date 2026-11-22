
from pathlib import Path
import json
import joblib
from shutil import copy2

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from src.preprocess import prepare_data


# =====================================
# CONFIGURATION
# =====================================

SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

EXPERIMENT_DIR = MODEL_DIR / "experiments"
BACKUP_DIR = MODEL_DIR / "baseline_backup"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = MODEL_DIR / "scaler.pkl"
BACKUP_SCALER = BACKUP_DIR / "scaler.pkl"
EXPERIMENT_SCALER = EXPERIMENT_DIR / "scaler_1m.pkl"


# =====================================
# VERIFY BASELINE BACKUP
# =====================================

if not BACKUP_SCALER.exists():
    raise FileNotFoundError(
        "Baseline scaler backup missing. "
        "Complete Step 1 before training."
    )


# =====================================
# LOAD AND PREPROCESS DATA
# =====================================

print("\nLoading PaySim dataset...")

# prepare_data() writes models/scaler.pkl.
# Always restore the backed-up baseline scaler
# so that the current API keeps using its
# original preprocessing artifact.

try:
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_data()

    # Save scaler created for this experiment.
    copy2(SCALER_PATH, EXPERIMENT_SCALER)

finally:
    copy2(BACKUP_SCALER, SCALER_PATH)


print("\nFull training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows reserved:", len(X_test))

# We deliberately do not evaluate on the
# test set during this experiment.

del X_test, y_test


# =====================================
# SAMPLE ONE MILLION TRAINING ROWS
# =====================================

if SAMPLE_SIZE > len(X_train):
    raise ValueError(
        "Sample size exceeds training split size."
    )

print("\nCreating stratified training sample...")

X_sample, _, y_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=SAMPLE_SIZE,
    stratify=y_train,
    random_state=RANDOM_STATE
)

fraud_count = int(y_sample.sum())
normal_count = len(y_sample) - fraud_count

scale_weight = normal_count / fraud_count

print("Training sample:", len(X_sample))
print("Fraud cases:", fraud_count)
print("Normal cases:", normal_count)
print("Scale pos weight:", scale_weight)

# Release the original full training split
# after sampling to reduce memory usage.
del X_train, y_train


# =====================================
# TRAIN XGBOOST
# =====================================

print("\nTraining XGBoost with 1M transactions...")

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

model.fit(X_sample, y_sample)

print("\nTraining completed.")


# =====================================
# VALIDATION PREDICTIONS
# =====================================

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

threshold_results = []

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

    threshold_results.append({
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    })

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = float(f1)
        best_threshold = float(threshold)
        best_precision = float(precision)
        best_recall = float(recall)


# =====================================
# FINAL VALIDATION SUMMARY
# =====================================

best_predictions = (
    probabilities >= best_threshold
).astype(int)

average_precision = average_precision_score(
    y_val,
    probabilities
)

cm = confusion_matrix(
    y_val,
    best_predictions,
    labels=[0, 1]
)

print("\n===================================")
print("1 MILLION TRANSACTION EXPERIMENT")
print("===================================")

print("Best threshold:", best_threshold)
print("Precision:", round(best_precision, 4))
print("Recall:", round(best_recall, 4))
print("F1 Score:", round(best_f1, 4))
print(
    "Average Precision:",
    round(average_precision, 4)
)

print("\nConfusion Matrix:")
print(cm)

print("\nBASELINE VALIDATION F1: 0.8666")
print("NEW VALIDATION F1:", round(best_f1, 4))


# =====================================
# SAVE EXPERIMENT ARTIFACTS
# =====================================

joblib.dump(
    model,
    EXPERIMENT_DIR / "xgboost_1m.pkl"
)

metadata = {
    "model_name": "XGBoost 1M Experiment",
    "training_sample_size": SAMPLE_SIZE,
    "fraud_training_rows": fraud_count,
    "normal_training_rows": normal_count,
    "scale_pos_weight": float(scale_weight),
    "feature_names": X_sample.columns.tolist(),
    "threshold": best_threshold,
    "validation_precision": best_precision,
    "validation_recall": best_recall,
    "validation_f1": best_f1,
    "validation_average_precision": float(
        average_precision
    ),
    "threshold_results": threshold_results,
    "test_set_evaluated": False
}

with open(
    EXPERIMENT_DIR / "xgboost_1m_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\nExperiment model saved successfully!")
print("Location: models/experiments/")
print("Baseline model remains unchanged.")
