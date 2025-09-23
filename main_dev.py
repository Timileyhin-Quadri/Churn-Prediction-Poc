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
    """Load model when the API starts"""
    success = load_latest_model()
    if not success:
        print("⚠️ Warning: Could not load model. API will not work properly.")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Churn Prediction API is running!",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api_status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": feature_names is not None,
        "total_features": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerData):
    """Predict churn probability for a customer"""
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Prepare features
        features_df = prepare_features(customer)
        
        # --------------------- START: DEBUGGING BLOCK ---------------------
        print("\n--- API PREDICTION DEBUG ---")
        
        # Check if the model has the 'feature_names_in_' attribute
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            passed_features = features_df.columns.tolist()

            print(f"Features passed to model ({len(passed_features)}):")
            print(passed_features)
            
            print(f"\nFeatures expected by model ({len(expected_features)}):")
            print(expected_features)
            
            # Compare the feature sets
            passed_set = set(passed_features)
            expected_set = set(expected_features)
            
            if passed_set != expected_set:
                print("\n❌ MISMATCH FOUND!")
                missing_from_passed = expected_set - passed_set
                extra_in_passed = passed_set - expected_set
                if missing_from_passed:
                    print(f"   - Missing from API features: {list(missing_from_passed)}")
                if extra_in_passed:
                    print(f"   - Extra features from API: {list(extra_in_passed)}")
            else:
                print("\n✅ Feature sets MATCH.")
                # If sets match, check the order
                if passed_features != list(expected_features):
                    print("   - ❌ But feature ORDER is INCORRECT.")
                else:
                    print("   - ✅ And feature ORDER is CORRECT.")

        else:
            # Fallback for models without the attribute (like some pipelines)
            print("Model object does not have 'feature_names_in_'. Comparing against loaded feature list.")
            if feature_names:
                 if list(features_df.columns) == feature_names:
                     print("✅ Features match the loaded feature_names list.")
                 else:
                     print("❌ Features DO NOT match the loaded feature_names list.")

        print("----------------------------\n")
        # ---------------------- END: DEBUGGING BLOCK ----------------------

        # Make prediction
        prediction_proba = model.predict_proba(features_df)[0]
        churn_probability = prediction_proba[1]  # Probability of churning
        
        # (The rest of your function continues here...)
        
        # Determine prediction and risk level
        will_churn = "Will Churn" if churn_probability > 0.5 else "Will Not Churn"
        risk_level = get_risk_level(churn_probability)
        confidence = max(prediction_proba)  # Confidence in prediction
        
        # Get key factors
        key_factors = get_key_factors(customer)
        
        return ChurnPrediction(
            churn_probability=round(churn_probability, 4),
            churn_prediction=will_churn,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            key_factors=key_factors
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/predict_batch")
async def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers"""
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        results = []
        
        for i, customer in enumerate(customers):
            # Prepare features
            features_df = prepare_features(customer)
            
            # Make prediction
            prediction_proba = model.predict_proba(features_df)[0]
            churn_probability = prediction_proba[1]
            
            # Create result
            result = ChurnPrediction(
                customer_id=f"customer_{i+1}",
                churn_probability=round(churn_probability, 4),
                churn_prediction="Will Churn" if churn_probability > 0.5 else "Will Not Churn",
                risk_level=get_risk_level(churn_probability),
                confidence=round(max(prediction_proba), 4),
                key_factors=get_key_factors(customer)
            )
            results.append(result)
        
        return {"predictions": results, "total_customers": len(customers)}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error making batch prediction: {str(e)}"
        )

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_info = {
        "model_type": type(model).__name__,
        "features_count": len(feature_names) if feature_names else "Unknown",
        "sample_features": feature_names[:10] if feature_names else []
    }
    
    # Try to get additional model info
    try:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            if feature_names and len(importances) == len(feature_names):
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                model_info["top_features"] = feature_importance[:10]
    except:
        pass
    
    return model_info

@app.get("/debug/files")
async def debug_files():
    """Debug endpoint to check what files are available"""
    return {
        "models_dir_exists": os.path.exists("models/"),
        "model_files": glob.glob("models/best_model_*.pkl"),
        "feature_files": glob.glob("models/feature_names_*.pkl"),
        "all_model_files": os.listdir("models/") if os.path.exists("models/") else [],
        "model_loaded": model is not None,
        "feature_names_loaded": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else 0
    }

@app.get("/debug/model_features")
async def get_model_features():
    """
    Returns the exact list of feature names and their order that the
    loaded model expects for predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    expected_features = []
    source = "Not available"

    # Scikit-learn models store feature names in 'feature_names_in_'
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_.tolist()
        source = "model.feature_names_in_"
    # Check pipeline steps for the final estimator
    elif hasattr(model, 'steps'):
        final_estimator = model.steps[-1][1]
        if hasattr(final_estimator, 'feature_names_in_'):
            expected_features = final_estimator.feature_names_in_.tolist()
            source = f"Final estimator in pipeline ({type(final_estimator).__name__})"
    # Fallback to the loaded feature_names list
    elif feature_names:
        expected_features = feature_names
        source = "Loaded from feature_names.pkl (fallback)"

    if not expected_features:
        return {"error": "Could not determine expected features from the model."}

    return {
        "model_type": type(model).__name__,
        "features_source": source,
        "features_count": len(expected_features),
        "expected_features": expected_features
    }
@app.get("/debug/expected_features")
async def get_expected_features():
    """Debug endpoint to see exactly what features the model expects"""
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Feature names not loaded")
    
    # Group features by type for easier understanding
    categorical_features = [f for f in feature_names if '_' in f and any(cat in f for cat in ['gender_', 'occupation_'])]
    engineered_features = [f for f in feature_names if f in [
        'balance_change', 'balance_volatility', 'credit_debit_ratio', 
        'credit_change', 'debit_change', 'is_dormant_account', 
        'avg_monthly_activity', 'balance_consistency', 'days_since_last_transaction'
    ]]
    original_features = [f for f in feature_names if f not in categorical_features + engineered_features]
    
    return {
        "total_features": len(feature_names),
        "categorical_features": sorted(categorical_features),
        "engineered_features": sorted(engineered_features), 
        "original_features": sorted(original_features),
        "all_features": feature_names
    }

@app.post("/debug/show_my_features")
async def debug_my_features(customer: CustomerData):
    """Debug endpoint to see what features are created for a customer"""
    try:
        # Prepare features
        features_df = prepare_features(customer)
        
        return {
            "created_features": list(features_df.columns),
            "features_count": len(features_df.columns),
            "sample_values": {col: float(features_df[col].iloc[0]) for col in features_df.columns[:15]},
            "active_categorical": {col: int(features_df[col].iloc[0]) for col in features_df.columns 
                                 if col.startswith(('gender_', 'occupation_')) and features_df[col].iloc[0] == 1},
            "feature_names_available": feature_names is not None,
            "expected_features_count": len(feature_names) if feature_names else 0,
            "features_match": list(features_df.columns) == feature_names if feature_names else False
        }
        
    except Exception as e:
        return {"error": str(e)}

# Example usage endpoint
@app.get("/example")
async def get_example():
    """Get an example of how to use the API"""
    return {
        "example_request": {
            "age": 35,
            "gender": "male",
            "vintage": 24,
            "current_balance": 5000.0,
            "previous_month_end_balance": 4500.0,
            "average_monthly_balance_prevq": 4800.0,
            "average_monthly_balance_prevq2": 4600.0,  # Added missing
            "current_month_balance": 5000.0,  # Added missing
            "previous_month_balance": 4500.0,  # Added missing
            "current_month_credit": 3000.0,
            "current_month_debit": 2500.0,
            "previous_month_credit": 2800.0,
            "previous_month_debit": 2200.0,
            "dependents": 2,
            "occupation": "salaried",  # Use valid occupation
            "customer_nw_category": 2,
            "city": 101,
            "branch_code": 1001  # Added missing
        },
        "usage": {
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /predict_batch",
            "health_check": "GET /health",
            "model_info": "GET /model_info",
            "debug_features": "GET /debug/expected_features",
            "test_features": "POST /debug/show_my_features"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Churn Prediction API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)