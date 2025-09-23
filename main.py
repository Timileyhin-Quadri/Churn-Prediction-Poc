from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from typing import Optional
import glob

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Simple API to predict customer churn",
    version="1.0.0"
)

# Global variables to store loaded model and features
model = None
feature_names = None
original_categorical_features = None  # Store original categorical info

class CustomerData(BaseModel):
    """Input data model for customer information - matches preprocessing pipeline"""
    # Core demographic features
    age: int
    gender: str  # "male" or "female"
    vintage: int  # months with company
    
    # All balance features from original dataset
    current_balance: float
    previous_month_end_balance: float
    average_monthly_balance_prevq: float
    average_monthly_balance_prevq2: float  # Added missing feature
    current_month_balance: float  # Added missing feature
    previous_month_balance: float  # Added missing feature
    
    # Transaction features
    current_month_credit: float
    current_month_debit: float
    previous_month_credit: float
    previous_month_debit: float
    
    # Required features from original dataset
    dependents: Optional[float] = 0
    occupation: Optional[str] = "unknown"
    customer_nw_category: Optional[int] = 1
    city: Optional[float] = 1
    branch_code: Optional[int] = 1  # Added missing feature

class ChurnPrediction(BaseModel):
    """Output model for prediction results"""
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: str  # "Will Churn" or "Will Not Churn"
    risk_level: str  # "Low", "Medium", "High"
    confidence: float
    key_factors: list

def load_latest_model():
    """Load the most recent trained model and feature names - IMPROVED VERSION"""
    global model, feature_names, original_categorical_features
    
    try:
        # Check if models directory exists
        if not os.path.exists("models/"):
            raise FileNotFoundError("models/ directory does not exist")
        
        # Find the latest model file
        model_files = glob.glob("models/best_model_*.pkl")
        if not model_files:
            raise FileNotFoundError("No trained model found in models/ directory")
        
        latest_model_file = max(model_files, key=os.path.getctime)
        print(f"Loading model from: {latest_model_file}")
        
        # Load the model
        with open(latest_model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # Try multiple strategies to get feature names
        feature_names = None
        
        # Strategy 1: Try to find matching timestamp feature file
        try:
            # Extract timestamp more carefully
            filename = os.path.basename(latest_model_file)
            # Handle different possible filename formats
            if "_" in filename:
                parts = filename.split("_")
                # Look for timestamp in last part (remove .pkl extension)
                timestamp = parts[-1].replace('.pkl', '')
                expected_feature_file = f"models/feature_names_{timestamp}.pkl"
                
                if os.path.exists(expected_feature_file):
                    print(f"📁 Found matching feature file: {expected_feature_file}")
                    with open(expected_feature_file, 'rb') as f:
                        feature_names = pickle.load(f)
                else:
                    print(f"⚠️ Expected feature file not found: {expected_feature_file}")
        except Exception as e:
            print(f"⚠️ Error with timestamp extraction: {e}")
        
        # Strategy 2: Try any available feature file (use latest)
        if feature_names is None:
            feature_files = glob.glob("models/feature_names_*.pkl")
            if feature_files:
                latest_feature_file = max(feature_files, key=os.path.getctime)
                print(f"📁 Using latest available feature file: {latest_feature_file}")
                with open(latest_feature_file, 'rb') as f:
                    feature_names = pickle.load(f)
            else:
                print("⚠️ No feature_names_*.pkl files found")
        
        # Strategy 3: Try to get features from model itself
        if feature_names is None:
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
                print(f"📁 Got features from model.feature_names_in_")
            else:
                print("⚠️ Model doesn't have feature_names_in_ attribute")
        
        # Strategy 4: Try loading from pipeline if model is a pipeline
        if feature_names is None and hasattr(model, 'steps'):
            try:
                # Get the final estimator from pipeline
                final_estimator = model.steps[-1][1]
                if hasattr(final_estimator, 'feature_names_in_'):
                    feature_names = list(final_estimator.feature_names_in_)
                    print(f"📁 Got features from pipeline final estimator")
            except Exception as e:
                print(f"⚠️ Error extracting features from pipeline: {e}")
        
        # If all strategies failed, raise an error
        if feature_names is None:
            raise ValueError("Could not load feature names from any source!")
            
        print(f"✅ Loaded {len(feature_names)} features")
        
        # Parse categorical features from feature names
        original_categorical_features = {
            'gender_categories': [f.replace('gender_', '') for f in feature_names if f.startswith('gender_')],
            'occupation_categories': [f.replace('occupation_', '') for f in feature_names if f.startswith('occupation_')]
        }
        print(f"✅ Found categorical features: {original_categorical_features}")
        
        # Debug: Print first few feature names
        print(f"📋 First 10 features: {feature_names[:10]}")
        
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"💡 Debug information:")
        print(f"   - Available model files: {glob.glob('models/best_model_*.pkl')}")
        print(f"   - Available feature files: {glob.glob('models/feature_names_*.pkl')}")
        print(f"   - Models directory exists: {os.path.exists('models/')}")
        return False


def engineer_features(data: dict) -> dict:
    """Create engineered features exactly as in preprocessing pipeline"""
    # Balance-related features (matching preprocessing exactly)
    data['balance_change'] = data['current_balance'] - data['previous_month_end_balance']
    data['balance_volatility'] = abs(data['average_monthly_balance_prevq'] - data['average_monthly_balance_prevq2'])
    
    # Credit/Debit patterns (matching preprocessing exactly)
    if data['current_month_debit'] != 0:
        data['credit_debit_ratio'] = data['current_month_credit'] / data['current_month_debit']
    else:
        data['credit_debit_ratio'] = 0
        
    data['credit_change'] = data['current_month_credit'] - data['previous_month_credit']
    data['debit_change'] = data['current_month_debit'] - data['previous_month_debit']
    
    # Account activity indicators
    data['is_dormant_account'] = 1 if (data['current_month_credit'] == 0 and data['current_month_debit'] == 0) else 0
    data['avg_monthly_activity'] = (data['current_month_credit'] + data['current_month_debit']) / 2
    
    # Balance consistency (matching preprocessing exactly)
    if data['average_monthly_balance_prevq'] != 0:
        data['balance_consistency'] = 1 - (abs(data['current_balance'] - data['average_monthly_balance_prevq']) / 
                                         abs(data['average_monthly_balance_prevq']))
    else:
        data['balance_consistency'] = 0
    
    # Add days_since_last_transaction (default value since we don't have actual date)
    data['days_since_last_transaction'] = 30  # Default to 30 days
    
    return data

def prepare_features(customer_data: CustomerData) -> pd.DataFrame:
    """
    Convert customer data to model-ready features, correctly handling
    one-hot encoding with drop_first=True logic.
    """
    if not feature_names:
        raise ValueError("Feature names not loaded. Cannot prepare features.")

    # 1. Start with a dictionary of all expected features set to 0.
    feature_dict = {feature: 0 for feature in feature_names}

    # 2. Get input data and create engineered features.
    input_data = customer_data.dict()
    input_data = engineer_features(input_data)

    # 3. Fill the dictionary with numerical and engineered feature values.
    for key, value in input_data.items():
        if key in feature_dict:
            feature_dict[key] = value

    # 4. Handle one-hot encoding, respecting the `drop_first=True` logic.
    # We only try to set the column that would have been created by get_dummies.
    # If the input is the "dropped" category, no column will match, and all
    # related columns will correctly stay 0.
    
    # Gender
    gender = input_data.get('gender', '').lower()
    gender_col = f"gender_{gender}"
    if gender_col in feature_dict:
        feature_dict[gender_col] = 1

    # Occupation
    occupation = input_data.get('occupation', 'unknown').lower()
    occupation_col = f"occupation_{occupation}"
    if occupation_col in feature_dict:
        feature_dict[occupation_col] = 1

    # 5. Create the DataFrame.
    df = pd.DataFrame([feature_dict])

    # 6. CRITICAL: Ensure the final column order is exactly what the model expects.
    return df[feature_names]


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def get_key_factors(customer_data: CustomerData) -> list:
    """Identify key risk factors for this customer"""
    factors = []
    
    # Balance factors
    if customer_data.current_balance < customer_data.previous_month_end_balance:
        factors.append("Declining account balance")
    
    if customer_data.current_balance < 1000:
        factors.append("Low account balance")
    
    # Activity factors
    if customer_data.current_month_credit == 0 and customer_data.current_month_debit == 0:
        factors.append("No account activity this month")
    
    # Credit/Debit ratio
    if customer_data.current_month_debit > customer_data.current_month_credit:
        factors.append("Spending more than receiving")
    
    # Tenure factor
    if customer_data.vintage < 12:
        factors.append("New customer (less than 1 year)")
    
    # Default message if no specific factors
    if not factors:
        factors.append("Standard risk assessment completed")
    
    return factors


@app.on_event("startup")
async def startup_event():
    success = load_latest_model()
    if not success:
        print("⚠️ Warning: Could not load model. API will not work properly.")

@app.get("/")
async def root():
    return {
        "message": "Churn Prediction API is running!",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "api_status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": feature_names is not None,
        "total_features": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        features_df = prepare_features(customer)

        # --- COMMENTED OUT: Verbose Debugging for Production ---
        # print("\n--- API PREDICTION DEBUG ---")
        # if hasattr(model, 'feature_names_in_'):
        #     ...
        # print("----------------------------\n")

        prediction_proba = model.predict_proba(features_df)[0]
        churn_probability = prediction_proba[1]
        will_churn = "Will Churn" if churn_probability > 0.5 else "Will Not Churn"
        risk_level = get_risk_level(churn_probability)
        confidence = max(prediction_proba)
        key_factors = get_key_factors(customer)

        return ChurnPrediction(
            churn_probability=round(churn_probability, 4),
            churn_prediction=will_churn,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            key_factors=key_factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(customers: list[CustomerData]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        results = []
        for i, customer in enumerate(customers):
            features_df = prepare_features(customer)
            prediction_proba = model.predict_proba(features_df)[0]
            churn_probability = prediction_proba[1]

            results.append(
                ChurnPrediction(
                    customer_id=f"customer_{i+1}",
                    churn_probability=round(churn_probability, 4),
                    churn_prediction="Will Churn" if churn_probability > 0.5 else "Will Not Churn",
                    risk_level=get_risk_level(churn_probability),
                    confidence=round(max(prediction_proba), 4),
                    key_factors=get_key_factors(customer)
                )
            )

        return {"predictions": results, "total_customers": len(customers)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch prediction: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    model_info = {
        "model_type": type(model).__name__,
        "features_count": len(feature_names) if feature_names else "Unknown",
        "sample_features": feature_names[:10] if feature_names else []
    }

    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names and len(importances) == len(feature_names):
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                model_info["top_features"] = feature_importance[:10]
    except:
        pass

    return model_info

@app.get("/example")
async def get_example():
    return {
        "example_request": {
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
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)