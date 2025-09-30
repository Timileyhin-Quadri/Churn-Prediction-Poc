# Churn-Prediction-Poc
End-to-end ML pipeline for predicting customer churn with data preprocessing, model training, REST API, and web dashboard.
## 🎯Overview
* Data Preprocessing: KNN imputation, feature engineering, SMOTE for class imbalance
* Model Training: Compares Logistic Regression, Decision Tree, Random Forest, and XGBoost
* REST API: FastAPI service for predictions
* Web Dashboard: Streamlit interface for easy interaction
## ✨Features
### Data Preprocessing (preproces.py)
* Missing Value Handling: KNN imputation for numerical features, mode imputation for categorical
* Feature Engineering: Creates 9+ derived features including:
** Balance change and volatility metrics
** Credit/debit ratios and changes
** Account activity indicators
** Balance consistency measures
** Outlier Treatment: IQR-based clipping
** Class Imbalance Handling: SMOTE, oversampling, or undersampling
** Scaling: StandardScaler for numerical features
### Model Training (train.py)
* Multiple Models: Trains and compares 4 classification algorithms
* Hyperparameter Tuning: GridSearchCV with 5-fold cross-validation
* Comprehensive Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
* Model Persistence: Saves best model and feature names automatically
* Results Tracking: Exports comparison CSV with all metrics
## 📁Project Structure
```
├── data/
│   └── churn_prediction.csv       # Raw dataset
├── models/                         # Trained models (auto-generated)
├── results/                        # Model comparisons (auto-generated)
├── preproces.py                    # Preprocessing pipeline
├── train.py                        # Model training
├── main.py                          # FastAPI service
└── streamlit_app.py                # Web interface
```
## 🔧Installation
### Prerequisites
* Python 3.8+
### Setup
1. Clone the repository
```
git clone <repository-url>
cd churn-prediction
```
2. Create virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
## 🚀Usage
* Preprocess Data: `python preproces.py`
* Train Model: `python model_training.py`
* Run API: `uvicorn main:app --reload`
* Launch Dashboard: `streamlit run streamlit_app.py`
## 📡API Documentation
Predict Single Customer
Endpoint: `POST /predict`
Request Body:
```
{
  "age": 35,
  "gender": "male",
  "vintage": 24,
  "current_balance": 5000.0,
  "previous_month_end_balance": 4500.0,
  "average_monthly_balance_prevq": 4800.0,
  "average_monthly_balance_prevq2": 4600.0,
  "current_month_balance": 5000.0,
  "previous_month_balance": 4500.0,
  "current_month_credit": 3000.0,
  "current_month_debit": 2500.0,
  "previous_month_credit": 2800.0,
  "previous_month_debit": 2200.0,
  "dependents": 2,
  "occupation": "salaried",
  "customer_nw_category": 2,
  "city": 101,
  "branch_code": 1001
}
```
Response:
```
{
  "churn_probability": 0.3245,
  "churn_prediction": "Will Not Churn",
  "risk_level": "Medium",
  "confidence": 0.6755,
  "key_factors": [
    "Standard risk assessment completed"
  ]
}
```
## 🌐Deployment
* Docker Deployment
```
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
* 🚀Deploy to Azure Container Instances
```
# Build & Tag
docker build -t churn-api .
docker tag churn-api <ACR_NAME>.azurecr.io/churn-api:v1

# Push to ACR
az login
az acr login --name <ACR_NAME>
docker push <ACR_NAME>.azurecr.io/churn-api:v1

# Deploy to ACI
az container create \
  --resource-group <RESOURCE_GROUP> \
  --name churn-api \
  --image <ACR_NAME>.azurecr.io/churn-api:v1 \
  --ports 8000

# Get Public IP
az container show \
  --resource-group <RESOURCE_GROUP> \
  --name churn-api \
  --query ipAddress.ip --output tsv
```
## License 
MIT
