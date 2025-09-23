import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
import warnings
import os
import joblib

# Import modules for classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Define paths and constants
RAW_DATA_PATH = "data/churn_prediction.csv"
MODEL_DIR = "models/"
RESULTS_DIR = "results/"
TARGET_COL = "churn"

def get_models_config():
    """Define models and their hyperparameter grids"""
    return {
        "LogisticRegression": {
            "model": Pipeline([
                ('scaler', StandardScaler()), 
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            "params": {
                "classifier__penalty": ["l1", "l2"],
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"]
            }
        },
        
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"]
            }
        },
        
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        },
        
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42, 
                use_label_encoder=False,
                eval_metric="logloss"
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        }
    }

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # ROC AUC (if model supports probability prediction)
    roc_auc = None
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            print(f"   ⚠️ Could not calculate ROC AUC: {e}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def print_model_results(name, metrics, training_time, best_params):
    """Print formatted model results"""
    print(f"\n{'='*60}")
    print(f"🎯 {name} Results")
    print(f"{'='*60}")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    print(f"🔧 Best Parameters: {best_params}")
    print(f"\n📊 Performance Metrics:")
    print(f"   • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   • Precision: {metrics['precision']:.4f}")
    print(f"   • Recall:    {metrics['recall']:.4f}")
    print(f"   • F1-Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"   • ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n🔍 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives:  {cm[0,0]:4d}   False Positives: {cm[0,1]:4d}")
    print(f"   False Negatives: {cm[1,0]:4d}   True Positives:  {cm[1,1]:4d}")

def save_model_results(results, timestamp):
    """Save all model results to a CSV file"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, f"model_comparison_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"💾 Results saved to: {results_path}")
    return results_path

def save_best_model(model, model_name, feature_names, timestamp):
    """Save the best model and its features"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, f"best_model_{model_name}_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save feature names
    features_path = os.path.join(MODEL_DIR, f"feature_names_{timestamp}.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)
    
    print(f"💾 Best model ({model_name}) saved to: {model_path}")
    print(f"💾 Feature names saved to: {features_path}")
    
    return model_path, features_path

def train_churn_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models and return the best performing one
    """
    print("🚀 Starting Model Training Pipeline")
    print("=" * 60)
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get models configuration
    models_config = get_models_config()
    
    # Initialize tracking variables
    results = []
    best_model = None
    best_score = -1.0
    best_name = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Test data shape: {X_test.shape}")
    print(f"🎯 Class distribution in training set:")
    train_dist = pd.Series(y_train).value_counts(normalize=True) * 100
    for class_val, pct in train_dist.items():
        print(f"   Class {class_val}: {pct:.1f}%")
    
    # Training loop
    total_start_time = time.time()
    
    for name, config in models_config.items():
        print(f"\n🔄 Training {name}...")
        start_time = time.time()
        
        try:
            # Setup GridSearchCV
            grid = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0
            )
            
            # Fit the model
            grid.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Get best model
            best_estimator = grid.best_estimator_
            
            # Evaluate on test set
            metrics = evaluate_model(best_estimator, X_test, y_test)
            
            # Print results
            print_model_results(name, metrics, training_time, grid.best_params_)
            
            # Store results
            result_row = {
                'Model': name,
                'Training_Time': training_time,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'ROC_AUC': metrics['roc_auc'],
                'Best_Params': str(grid.best_params_)
            }
            results.append(result_row)
            
            # Check if this is the best model (based on ROC AUC)
            if metrics['roc_auc'] is not None and metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = best_estimator
                best_name = name
                
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Final results summary
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"⏱️  Total training time: {total_time:.2f} seconds")
    
    if best_model is not None:
        print(f"🥇 Best Model: {best_name}")
        print(f"🎯 Best ROC AUC Score: {best_score:.4f}")
        
        # Save results and best model
        results_path = save_model_results(results, timestamp)
        model_path, features_path = save_best_model(
            best_model, best_name, list(X_train.columns), timestamp
        )
        
        # Display final comparison table
        print(f"\n📋 Model Comparison Summary:")
        results_df = pd.DataFrame(results)
        print(results_df[['Model', 'ROC_AUC', 'F1_Score', 'Accuracy', 'Training_Time']].round(4))
        
        return best_model, best_name, best_score, results_path, model_path
    
    else:
        print("❌ No models were successfully trained!")
        return None, None, None, None, None

def load_and_train_from_preprocessing():
    """Load data from preprocessing script and train models"""
    try:
        # Import the preprocessing script
        import preproces as preprocessing_module
        
        # Load and preprocess data
        print("📥 Loading and preprocessing data...")
        df_raw = pd.read_csv(RAW_DATA_PATH)
        
        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessing_module.process_data_for_classification(
            df=df_raw,
            target_col=TARGET_COL,
            test_size=0.2,
            random_state=42,
            scaling_method='standard',
            imbalance_method='smote'
        )
        
        print("✅ Preprocessing completed successfully!")
        
        # Train models
        return train_churn_models(X_train, X_test, y_train, y_test)
        
    except Exception as e:
        print(f"❌ Error in preprocessing or training: {e}")
        return None, None, None, None, None

if __name__ == "__main__":
    # Option 1: Load data from preprocessing and train
    best_model, best_name, best_score, results_path, model_path = load_and_train_from_preprocessing()
    
    if best_model is not None:
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"🏆 Best model: {best_name} (ROC AUC: {best_score:.4f})")
        print(f"📁 Model saved at: {model_path}")
        print(f"📊 Results saved at: {results_path}")
    else:
        print("\n❌ Training pipeline failed!")

# Alternative usage: If you already have preprocessed data
def train_with_existing_data(X_train, X_test, y_train, y_test):
    """Train models with already preprocessed data"""
    return train_churn_models(X_train, X_test, y_train, y_test)