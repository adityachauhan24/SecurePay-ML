
import json
import joblib
import pandas as pd
from pathlib import Path
import io

from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.preprocess import preprocess_transaction





# =====================================
# CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="SecurePay ML API",
    description="Machine Learning Based Fraud Detection",
    version="1.0.0"
)

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://secure-pay-ml-three.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =====================================
# LOAD SAVED MODEL
# =====================================

model = joblib.load("models/xgboost_model.pkl")

with open("models/model_metadata.json", "r") as file:
    metadata = json.load(file)

threshold = metadata["threshold"]
feature_names = metadata["feature_names"]


# =====================================
# DEFINE TRANSACTION SCHEMA
# =====================================

class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: str
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


# =====================================
# HOME ENDPOINT
# =====================================

@app.get("/")
def home():
    return {"message": "SecurePay ML API is running"}


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health_check():
    return {"status": "healthy"}


# =====================================
# SINGLE TRANSACTION PREDICTION
# =====================================


# =====================================
# SINGLE TRANSACTION PREDICTION
# =====================================

@app.post("/predict")
def predict_fraud(transaction: Transaction):

    # 1. Validate transaction type
    valid_types = {
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "PAYMENT",
        "DEBIT"
    }

    if transaction.type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail="Invalid transaction type."
        )

    # 2. Validate transaction amount
    if transaction.amount <= 0:
        raise HTTPException(
            status_code=422,
            detail="Transaction amount must be greater than zero."
        )

    # 3. Check sender balance for outgoing transactions
    outgoing_types = {
        "TRANSFER",
        "CASH_OUT",
        "PAYMENT",
        "DEBIT"
    }

    if (
        transaction.type in outgoing_types
        and transaction.amount > transaction.oldbalanceOrg
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient Balance! "
                f"Available: {transaction.oldbalanceOrg}, "
                f"Requested: {transaction.amount}"
            )
        )

    # 4. Convert request into DataFrame
    df = pd.DataFrame([transaction.model_dump()])

    # 5. Apply saved preprocessing
    X = preprocess_transaction(df)

    # 6. Maintain original feature order
    X = X[feature_names]

    # 7. Predict fraud score
    fraud_score = float(model.predict_proba(X)[0, 1])

    # 8. Apply selected threshold
    prediction = int(fraud_score >= threshold)

    return {
        "fraud_score": round(fraud_score, 6),
        "prediction": prediction,
        "status": "FRAUD" if prediction == 1 else "NORMAL",
        "threshold": threshold
    }


# =====================================
# BATCH CSV PREDICTION
# =====================================

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):

    # Step 1: Check file extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    # Step 2: Read uploaded file
    contents = await file.read()

    # Limit upload size to 10 MB
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="CSV file exceeds 10 MB limit."
        )

    # Step 3: Convert CSV into DataFrame
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except (pd.errors.ParserError, UnicodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable CSV file."
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty."
        )

    # Step 4: Preprocess transactions
    try:
        X = preprocess_transaction(df)
        X = X[feature_names]
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    # Step 5: Predict fraud scores
    fraud_scores = model.predict_proba(X)[:, 1]

    predictions = (fraud_scores >= threshold).astype(int)

    # Step 6: Add prediction columns
    result_df = df.copy()

    result_df["fraud_score"] = fraud_scores
    result_df["prediction"] = predictions
    result_df["status"] = result_df["prediction"].map({
        0: "NORMAL",
        1: "FRAUD"
    })

    # Step 7: Create output CSV
    output = io.StringIO()

    result_df.to_csv(output, index=False)
    output.seek(0)

    # Step 8: Return downloadable CSV
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                'attachment; filename="securepay_predictions.csv"'
        }
    )



# ==========================================
# GET MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def get_model_info():

    project_root = Path(__file__).resolve().parent.parent

    metrics_path = (
        project_root / "models" / "evaluation_metrics.json"
    )

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    return {
        "name": metadata.get(
            "model_name",
            "SecurePay ML XGBoost"
        ),

        "algorithm": "XGBoost Classifier",

        "approach": "Supervised Binary Classification",

        "dataset": metrics["dataset"],

        "training_samples": metadata.get(
            "training_sample_size"
        ),

        "feature_count": len(
            metadata.get("feature_names", [])
        ),

        "threshold": metadata["threshold"],

        "test_metrics": {
            "accuracy": metrics["test_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "f2_score": metrics["f2_score"],
            "average_precision": metrics["average_precision"]
        },

        "model_comparison": metrics["model_comparison"],

        "evaluation_note": (
            "Historical results on synthetic PaySim data. "
            "The random test split had been inspected during "
            "development, so these figures should not be "
            "treated as fully independent production estimates."
        )
    }
