from sklearn.linear_model import LogisticRegression
from preprocess import prepare_data
from sklearn.metrics import fbeta_score
# Get processed data
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
# Create model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

#  Training model
print("Training started...")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print("First 10 Predictions:", y_pred[:10])
# print("First 10 Actual Labels:", y_test.iloc[:10].values)


# print("First 10 Probabilities:")
# probabilities = model.predict_proba(X_test.iloc[:10])
# print(probabilities)

# print("Model training completed!")

# # check learned parameters
# print("Coefficients Shape:", model.coef_.shape)
# print("Intercept:", model.intercept_)



from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix:")
# print(cm)


# print()

# tn, fp, fn, tp = cm.ravel()

# print("True Negatives:", tn)
# print("False Positives:", fp)
# print("False Negatives:", fn)
# print("True Positives:", tp)



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))



from sklearn.metrics import precision_score, recall_score, f1_score

# Predict fraud scores on validation data
val_scores = model.predict_proba(X_val)[:, 1]

thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]

best_threshold = None
best_f1 = -1

for threshold in thresholds:

    val_pred = (val_scores >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )

    recall = recall_score(
        y_val, val_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# print("\nBest Threshold:", best_threshold)
# print("Best Validation F1:", best_f1)


f2 = fbeta_score(
    y_val,
    val_pred,
    beta=2,
    zero_division=0
)

print(f"F2 Score: {f2:.4f}")