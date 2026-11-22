
import os
import json
import joblib
import pandas as pd

from preprocess import preprocess_transaction


# ==========================================
# STEP 1: FILE PATHS
# ==========================================

INPUT_PATH = "data/sample_transactions.csv"
OUTPUT_PATH = "data/batch_predictions.csv"

MODEL_PATH = "models/xgboost_model.pkl"
METADATA_PATH = "models/model_metadata.json"


# ==========================================
# STEP 2: LOAD TRAINED MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = metadata["threshold"]
feature_names = metadata["feature_names"]


# ==========================================
# STEP 3: READ CSV FILE
# ==========================================

df = pd.read_csv(INPUT_PATH)

print("Total Transactions:", len(df))

if df.empty:
    raise ValueError("Input CSV has no transactions.")


# ==========================================
# STEP 4: PREPROCESS TRANSACTIONS
# ==========================================

X = preprocess_transaction(df)

X = X[feature_names]


# ==========================================
# STEP 5: GENERATE PREDICTIONS
# ==========================================

fraud_scores = model.predict_proba(X)[:, 1]

predictions = (fraud_scores >= threshold).astype(int)


# ==========================================
# STEP 6: ADD RESULTS TO DATAFRAME
# ==========================================

result_df = df.copy()

result_df["fraud_score"] = fraud_scores

result_df["prediction"] = predictions

result_df["status"] = result_df["prediction"].map({
    0: "NORMAL",
    1: "FRAUD"
})


# ==========================================
# STEP 7: SAVE OUTPUT CSV
# ==========================================

os.makedirs("data", exist_ok=True)

result_df.to_csv(OUTPUT_PATH, index=False)

print("\n===== BATCH PREDICTION COMPLETE =====")
print("Total Transactions:", len(result_df))
print("Predicted Fraud:", int(predictions.sum()))
print("Predicted Normal:", int((predictions == 0).sum()))
print("Output File:", OUTPUT_PATH)

print("\nPrediction Preview:")
print(result_df.head())
