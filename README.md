
# SecurePay ML

### Machine Learning-Based Fraud Detection System

SecurePay ML is an end-to-end machine learning project designed to identify potentially fraudulent transactions using the PaySim synthetic mobile-money dataset.

The project combines machine learning with a FastAPI backend and a React frontend to support single-transaction predictions, batch CSV analysis, and model performance visualization.

## Features

- Single transaction fraud prediction
- Batch transaction prediction through CSV uploads
- Downloadable CSV prediction reports
- Fraud vs normal distribution chart
- Transaction type analysis
- Machine learning performance dashboard
- Comparison of multiple classification algorithms

## Application Screenshots

### Dashboard
![SecurePay Dashboard](screenshots/dashboard.png)

### Model Performance
![Model Performance](screenshots/model-performance.png)

### Single Transaction Prediction
![Single Prediction](screenshots/single-prediction.png)

### Batch CSV Analysis
![Batch Analysis](screenshots/batch-analysis.png)

## Tech Stack

**Machine Learning**
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

**Backend**
- FastAPI
- Uvicorn
- Pydantic

**Frontend**
- React
- Vite
- Recharts
- Papa Parse
- Lucide React

## Dataset

The project uses the PaySim synthetic mobile-money transaction dataset.

- Total transactions: 6,362,620
- Fraudulent transactions: 8,213
- Fraud rate: approximately 0.129%

The dataset is highly imbalanced, meaning fraudulent transactions are much less frequent than normal transactions.

## Machine Learning Workflow

1. Load and explore the PaySim dataset.
2. Select transaction features and prepare the data.
3. Perform feature engineering and one-hot encoding.
4. Split the data into training, validation, and testing partitions.
5. Apply numerical feature scaling.
6. Train Logistic Regression, Random Forest, and XGBoost.
7. Experiment with different XGBoost training sample sizes.
8. Select a model using validation metrics.
9. Evaluate the selected model and integrate it with FastAPI.
10. Display predictions and evaluation results in React.

## Algorithms Used

### Logistic Regression

Used as the initial linear classification baseline.

Best recorded validation F1-Score: **16.91%**

### Random Forest

Used as a tree-based ensemble baseline.

Best recorded validation F1-Score: **65.99%**

### XGBoost

Used as the main gradient-boosted tree classifier.

Three training-size experiments were conducted.

| Model | Training Samples | Validation F1 |
|---|---:|---:|
| XGBoost 300K | 300,000 | 86.66% |
| XGBoost 1M | 1,000,000 | 95.46% |
| XGBoost Full | 3,817,572 | 79.59% |

The 1M XGBoost version achieved the highest observed validation F1 among these three experiments and was selected for application integration.

These results come from the project's random-split experiments. Training configurations and sampling procedures should be considered when interpreting comparisons.

## Selected Model Performance

Selected model: **XGBoost — 1 Million Training Samples**

Historical random-split test results:

| Metric | Result |
|---|---:|
| Accuracy | 99.9877% |
| Precision | 91.46% |
| Recall | 99.76% |
| F1-Score | 95.43% |
| F2-Score | 97.98% |
| Average Precision | 99.10% |
| Decision Threshold | 0.99 |

### Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| Actual Normal | 1,270,728 | 153 |
| Actual Fraud | 4 | 1,639 |

### Why Accuracy Alone Can Be Misleading

The PaySim dataset contains approximately 99.87% normal transactions. A classifier that predicts every transaction as normal would therefore achieve a very high accuracy without detecting any fraud.

Precision, Recall, F1-Score, and Average Precision provide more meaningful information about fraud detection performance.

**Evaluation limitations:** These are historical results on synthetic data. The random test partition had been inspected during development, so the reported results should not be considered fully independent estimates or real-world production performance.

## Model Input Features

The deployed model uses 11 processed features:

- step
- amount
- oldbalanceOrg
- oldbalanceDest
- amountToBalanceRatio
- isAmountGreaterThanBalance
- type_CASH_IN
- type_CASH_OUT
- type_DEBIT
- type_PAYMENT
- type_TRANSFER

## System Architecture

React Frontend
    |
    v
FastAPI Backend
    |
    v
Transaction Preprocessing
    |
    v
Saved XGBoost Model
    |
    v
Fraud Score and Classification
    |
    v
React Dashboard / CSV Report

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API welcome information |
| `/health` | GET | API health information |
| `/predict` | POST | Single transaction prediction |
| `/predict-batch` | POST | Batch CSV predictions |
| `/model-info` | GET | Model details and evaluation metrics |

## Running the Project

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create a Python Virtual Environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start FastAPI

From the project root:

```bash
python -m uvicorn backend.main:app --reload
```

Backend API:
`http://127.0.0.1:8000`

API documentation:
`http://127.0.0.1:8000/docs`

### 5. Start React

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:
`http://localhost:5173`

**Important:** API inference requires the matching trained model, scaler, metadata, and evaluation files under `models/`. If trained artifacts are not included in the repository, they must be obtained or generated separately.

## Example Transaction Input

```json
{
  "step": 101,
  "type": "CASH_OUT",
  "amount": 250000,
  "oldbalanceOrg": 250000,
  "oldbalanceDest": 0
}
```

## Limitations

- Uses synthetic PaySim transactions, not real banking data.
- Offline results do not establish real-world fraud detection accuracy.
- Fraud scores are not calibrated real-world probabilities.
- The dashboard stores analyzed transaction history in browser memory only.
- Re-uploading the same CSV counts the transactions again.
- Real-world deployment would require security, monitoring, calibration, and additional validation.

## Future Improvements

- Persistent transaction history using a database
- User authentication
- More robust input validation and error handling
- Evaluation on additional independent datasets
- Model monitoring and drift detection

## Project Purpose

This project demonstrates supervised machine learning, class imbalance handling, feature engineering, model evaluation, REST API integration, and frontend development in an end-to-end fraud detection application.
