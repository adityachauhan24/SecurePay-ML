
from pathlib import Path
from shutil import copy2
import json
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
EXP = MODELS / "experiments"
BACKUP = MODELS / "backup_before_1m"

# Existing application artifacts
live_model = MODELS / "xgboost_model.pkl"
live_scaler = MODELS / "scaler.pkl"
live_metadata = MODELS / "model_metadata.json"
live_metrics = MODELS / "evaluation_metrics.json"

# Trained 1M experiment
new_model = EXP / "xgboost_1m.pkl"
new_scaler = EXP / "scaler_1m.pkl"
new_metadata = EXP / "xgboost_1m_metadata.json"
new_results = EXP / "xgboost_1m_test_results.json"

# Verify all necessary files exist
for path in [
    live_model, live_scaler, live_metadata,
    new_model, new_scaler, new_metadata, new_results
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

with open(new_metadata, encoding="utf-8") as file:
    metadata = json.load(file)

with open(new_results, encoding="utf-8") as file:
    results = json.load(file)

# Verify saved model and metadata compatibility
model = joblib.load(new_model)
scaler = joblib.load(new_scaler)

if model.n_features_in_ != len(metadata["feature_names"]):
    raise ValueError("Model feature count mismatch")

if scaler.n_features_in_ != 5:
    raise ValueError("Unexpected numeric scaler features")

if results["threshold"] != metadata["threshold"]:
    raise ValueError("Threshold mismatch")

# Preserve original artifacts on the first activation
BACKUP.mkdir(parents=True, exist_ok=True)

for path in [
    live_model, live_scaler,
    live_metadata, live_metrics
]:
    if path.exists():
        destination = BACKUP / path.name
        if not destination.exists():
            copy2(path, destination)

# Prepare model metadata for FastAPI
production_metadata = {
    "model_name": "SecurePay ML XGBoost 1M",
    "threshold": metadata["threshold"],
    "feature_names": metadata["feature_names"],
    "training_sample_size": metadata["training_sample_size"],
    "scale_pos_weight": metadata["scale_pos_weight"],
    "best_validation_f1": metadata["validation_f1"]
}

# Preserve existing algorithm comparison records
comparison = [
    {"model": "Logistic Regression", "validation_f1": 0.1691},
    {"model": "Random Forest", "validation_f1": 0.6599},
    {"model": "XGBoost", "validation_f1": metadata["validation_f1"]}
]

evaluation = {
    "dataset": "PaySim synthetic mobile-money transactions",
    "test_accuracy": results["accuracy"],
    "precision": results["precision"],
    "recall": results["recall"],
    "f1_score": results["f1_score"],
    "f2_score": results["f2_score"],
    "average_precision": results["average_precision"],
    "confusion_matrix": results["confusion_matrix"],
    "model_comparison": comparison
}

# Activate saved model and its matching scaler
copy2(new_model, live_model)
copy2(new_scaler, live_scaler)

with open(live_metadata, "w", encoding="utf-8") as file:
    json.dump(production_metadata, file, indent=4)

with open(live_metrics, "w", encoding="utf-8") as file:
    json.dump(evaluation, file, indent=4)

print("\n1M XGBoost model activated successfully!")
print("Training samples:", metadata["training_sample_size"])
print("Threshold:", metadata["threshold"])
print("Test F1:", results["f1_score"])
print("Test Recall:", results["recall"])
print("Backup:", BACKUP)
print("\nRestart FastAPI to use the new model.")
