
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"

MODEL_DIR = "models"

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amountToBalanceRatio"
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amountToBalanceRatio",
    "isAmountGreaterThanBalance",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(df):

    df = df.copy()

    df["amountToBalanceRatio"] = (
        df["amount"] / df["oldbalanceOrg"].clip(lower=1)
    )

    df["isAmountGreaterThanBalance"] = (
        df["amount"] > df["oldbalanceOrg"]
    ).astype(int)

    return df


# ==========================================
# ENCODING
# ==========================================

def encode_features(df):

    df = create_features(df)

    selected_features = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amountToBalanceRatio",
        "isAmountGreaterThanBalance"
    ]

    X = df[selected_features]

    X_encoded = pd.get_dummies(
        X,
        columns=["type"],
        dtype=int
    )

    # Ensure identical columns and order
    X_encoded = X_encoded.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    return X_encoded


# ==========================================
# TRAINING DATA PREPARATION
# ==========================================

def prepare_data():

    df = pd.read_csv(DATA_PATH)

    X_encoded = encode_features(df)

    y = df["isFraud"]

    # 80% train+validation, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 60% train, 20% validation, 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train
    )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    X_train[NUMERIC_FEATURES] = scaler.fit_transform(
        X_train[NUMERIC_FEATURES]
    )

    X_val[NUMERIC_FEATURES] = scaler.transform(
        X_val[NUMERIC_FEATURES]
    )

    X_test[NUMERIC_FEATURES] = scaler.transform(
        X_test[NUMERIC_FEATURES]
    )

    # Save fitted scaler
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        scaler,
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================
# PREPROCESS NEW TRANSACTIONS
# ==========================================

def preprocess_transaction(df):

    # Accept a DataFrame with one or more transactions
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if not df["type"].isin(TRANSACTION_TYPES).all():
        raise ValueError("Invalid transaction type found.")

    X_encoded = encode_features(df)

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    X_encoded[NUMERIC_FEATURES] = scaler.transform(
        X_encoded[NUMERIC_FEATURES]
    )

    return X_encoded


# ==========================================
# RUN DIRECTLY
# ==========================================

if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    print("\n===== DATASET SHAPES =====")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    print("\n===== FRAUD COUNTS =====")
    print("Training Fraud:", y_train.sum())
    print("Validation Fraud:", y_val.sum())
    print("Testing Fraud:", y_test.sum())

    print("\n===== FRAUD PERCENTAGES =====")
    print("Training:", round(y_train.mean() * 100, 4))
    print("Validation:", round(y_val.mean() * 100, 4))
    print("Testing:", round(y_test.mean() * 100, 4))

    print("\nScaler saved successfully!")
