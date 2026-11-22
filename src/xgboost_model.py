
import os
import json
import joblib

import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix
)

from preprocess import prepare_data


# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================

print("\nLoading preprocessed data...")

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

print("Training Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)


# ==========================================
# 2. CREATE STRATIFIED TRAINING SAMPLE
# ==========================================

X_sample, _, y_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=300000,
    random_state=42,
    stratify=y_train
)

print("\nSample Shape:", X_sample.shape)
print("Fraud Cases in Sample:", int(y_sample.sum()))


# ==========================================
# 3. HANDLE CLASS IMBALANCE
# ==========================================

fraud_count = int(y_sample.sum())
normal_count = len(y_sample) - fraud_count

if fraud_count == 0:
    raise ValueError("Training sample contains no fraud cases.")

scale_weight = normal_count / fraud_count

print("Scale Pos Weight:", scale_weight)


# ==========================================
# 4. CREATE XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    tree_method="hist",
    scale_pos_weight=scale_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nXGBoost training started...")

model.fit(X_sample, y_sample)

print("XGBoost training completed!")


# ==========================================
# 6. REUSABLE EVALUATION FUNCTION
# ==========================================

def evaluate_model(y_true, scores, threshold, title):

    predictions = (scores >= threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    f2 = fbeta_score(
        y_true, predictions, beta=2, zero_division=0
    )

    ap = average_precision_score(y_true, scores)

    print(f"\n===== {title} =====")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print(f"Average Precision: {ap:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    return predictions


# ==========================================
# 7. VALIDATION EVALUATION
# ==========================================

val_scores = model.predict_proba(X_val)[:, 1]

evaluate_model(
    y_val,
    val_scores,
    threshold=0.5,
    title="XGBOOST VALIDATION RESULTS"
)


# ==========================================
# 8. THRESHOLD TUNING
# ==========================================

thresholds = [
    0.3, 0.5, 0.6, 0.7,
    0.8, 0.9, 0.95, 0.99
]

best_threshold = None
best_f1 = -1

print("\n===== XGBOOST THRESHOLD TUNING =====")

for threshold in thresholds:

    predictions = (val_scores >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    f2 = fbeta_score(
        y_val, predictions, beta=2, zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\n===== BEST VALIDATION THRESHOLD =====")
print("Best Tested Threshold:", best_threshold)
print(f"Best Validation F1: {best_f1:.4f}")


# ==========================================
# 9. BEST THRESHOLD EVALUATION
# ==========================================

best_predictions = evaluate_model(
    y_val,
    val_scores,
    threshold=best_threshold,
    title="BEST THRESHOLD EVALUATION"
)


# ==========================================
# 10. MISSED FRAUD ANALYSIS
# ==========================================

# Convert predictions into a Series with
# the same index as the validation dataset.
val_predictions = pd.Series(
    best_predictions,
    index=X_val.index
)

missed_mask = (
    (y_val == 1) &
    (val_predictions == 0)
)

missed_frauds = X_val.loc[missed_mask]

print("\n===== MISSED FRAUD ANALYSIS =====")

print("Total Missed Frauds:", len(missed_frauds))

print("\nFirst 5 Missed Frauds:")
print(missed_frauds.head())

print("\nMissed Fraud Feature Summary:")
print(missed_frauds.describe())


# ==========================================
# 11. FRAUD ANALYSIS BY TRANSACTION TYPE
# ==========================================

print("\n===== TRANSACTION TYPE ANALYSIS =====")

for transaction_type in ["CASH_OUT", "TRANSFER"]:

    column = f"type_{transaction_type}"

    type_mask = X_val[column] == 1

    actual_fraud_mask = (
        type_mask & (y_val == 1)
    )

    missed_fraud_mask = (
        actual_fraud_mask & (val_predictions == 0)
    )

    actual_frauds = int(actual_fraud_mask.sum())
    missed_count = int(missed_fraud_mask.sum())

    detected_count = actual_frauds - missed_count

    miss_rate = (
        missed_count / actual_frauds * 100
        if actual_frauds > 0 else 0
    )

    recall = (
        detected_count / actual_frauds * 100
        if actual_frauds > 0 else 0
    )

    print(f"\nTransaction Type: {transaction_type}")
    print("Actual Frauds:", actual_frauds)
    print("Detected Frauds:", detected_count)
    print("Missed Frauds:", missed_count)
    print(f"Miss Rate: {miss_rate:.2f}%")
    print(f"Recall: {recall:.2f}%")


# ==========================================
# 12. TEST SET EVALUATION
# ==========================================

test_scores = model.predict_proba(X_test)[:, 1]

test_predictions = evaluate_model(
    y_test,
    test_scores,
    threshold=best_threshold,
    title="XGBOOST TEST RESULTS"
)


# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model_path = "models/xgboost_model.pkl"
metadata_path = "models/model_metadata.json"

joblib.dump(model, model_path)

# Save the threshold and exact feature order.
metadata = {
    "model_name": "SecurePay ML XGBoost",
    "threshold": float(best_threshold),
    "feature_names": X_train.columns.tolist(),
    "training_sample_size": len(X_sample),
    "scale_pos_weight": float(scale_weight),
    "best_validation_f1": float(best_f1)
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\n===== MODEL SAVED SUCCESSFULLY =====")
print("Model Path:", model_path)
print("Metadata Path:", metadata_path)
print("Decision Threshold:", best_threshold)
print("Number of Features:", len(X_train.columns))
