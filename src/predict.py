
import json
import joblib
import pandas as pd

from preprocess import preprocess_transaction


# Step 1: Load saved model
model = joblib.load("models/xgboost_model.pkl")

# Step 2: Load model metadata
with open("models/model_metadata.json", "r") as file:
    metadata = json.load(file)

threshold = metadata["threshold"]
feature_names = metadata["feature_names"]


# Step 3: Create an example transaction
transaction = {
    "step": 100,
    "type": "TRANSFER",
    "amount": 50000,
    "oldbalanceOrg": 100000,
    "oldbalanceDest": 20000
}

df = pd.DataFrame([transaction])


# Step 4: Apply saved preprocessing
X = preprocess_transaction(df)

# Ensure correct feature order
X = X[feature_names]


# Step 5: Predict fraud score
fraud_score = model.predict_proba(X)[0, 1]

# Step 6: Apply saved threshold
prediction = int(fraud_score >= threshold)


# Step 7: Display result
print("\n===== SECUREPAY ML PREDICTION =====")
print(f"Fraud Score: {fraud_score:.4f}")
print(f"Decision Threshold: {threshold}")

if prediction == 1:
    print("Prediction: FRAUD")
else:
    print("Prediction: NORMAL")
