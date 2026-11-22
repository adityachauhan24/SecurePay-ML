
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix
)

import pandas as pd
import matplotlib.pyplot as plt

from preprocess import prepare_data


# =====================================
# STEP 1: LOAD PREPROCESSED DATA
# =====================================

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

print("Data preprocessing completed!")


# =====================================
# STEP 2: CREATE TRAINING SAMPLE
# =====================================

X_sample, _, y_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=300000,
    random_state=42,
    stratify=y_train
)

print("\nSample Shape:", X_sample.shape)
print("Fraud Cases in Sample:", y_sample.sum())


# =====================================
# STEP 3: CREATE RANDOM FOREST
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)


# =====================================
# STEP 4: TRAIN MODEL
# =====================================

print("\nRandom Forest training started...")

model.fit(X_sample, y_sample)

print("Random Forest training completed!")


# =====================================
# STEP 5: VALIDATION PREDICTIONS
# =====================================

y_pred = model.predict(X_val)

fraud_scores = model.predict_proba(X_val)[:, 1]


# =====================================
# STEP 6: MODEL EVALUATION
# =====================================

precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)

ap = average_precision_score(y_val, fraud_scores)

print("\n===== RANDOM FOREST VALIDATION =====")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"F2 Score: {f2:.4f}")
print(f"Average Precision: {ap:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))


# =====================================
# STEP 7: THRESHOLD TUNING
# =====================================

thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

best_f1 = -1
best_threshold = None

print("\n===== THRESHOLD EXPERIMENT =====")

for threshold in thresholds:

    predictions = (fraud_scores >= threshold).astype(int)

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


print("\n===== BEST THRESHOLD =====")
print("Best Tested Threshold:", best_threshold)
print("Best Validation F1:", round(best_f1, 4))


# =====================================
# STEP 8: FEATURE IMPORTANCE
# =====================================

importance = pd.Series(
    model.feature_importances_,
    index=X_sample.columns
)

importance = importance.sort_values(ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)


# =====================================
# STEP 9: FEATURE IMPORTANCE GRAPH
# =====================================

# plt.figure(figsize=(10, 5))

# importance.plot(kind="bar")

# plt.title("Random Forest Feature Importance")
# plt.xlabel("Features")
# plt.ylabel("Importance Score")

# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()

# plt.show()
