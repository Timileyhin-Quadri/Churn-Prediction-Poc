import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_PATH = 'data/churn_prediction.csv'

def clean_column_names(df):
    """Make column headers lowercase and remove whitespace for consistency."""
    df.columns = (
        df.columns.str.strip()  # remove leading/trailing spaces
        .str.lower()  # make lowercase
        .str.replace(" ", "_")  # replace spaces with underscore
    )
    return df

def check_missing_values(df):
    """Returns a DataFrame with counts and percentages of missing values per column."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    return pd.DataFrame({'missing_count': missing, 'missing_percent': percent}).sort_values(by='missing_count', ascending=False)

def handle_missing_values(df, numerical_features, categorical_features):
    """Handle missing values using appropriate strategies for different feature types."""
    print("   • Handling missing values...")
    
    # Use KNN imputation for numerical features
    numerical_in_df = [col for col in numerical_features if col in df.columns]
    if numerical_in_df and any(df[numerical_in_df].isnull().sum()):
        knn_imputer = KNNImputer(n_neighbors=5)
        df[numerical_in_df] = knn_imputer.fit_transform(df[numerical_in_df])
    
    # Use mode for categorical features
    categorical_in_df = [col for col in categorical_features if col in df.columns]
    for col in categorical_in_df:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_value, inplace=True)
    
    return df

def remove_duplicates(df):
    """Remove duplicate rows."""
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f"   • Removing {duplicates_count} duplicate rows")
        return df.drop_duplicates()
    return df

def handle_outliers(df, column):
    """Clip outliers in a column using IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    if outliers_count > 0:
        print(f"   • Capped {outliers_count} outliers in {column}")
    return df

def engineer_features(df):
    """Create new features from existing data."""
    print("   • Engineering new features...")
    
    # Balance-related features
    df['balance_change'] = df['current_balance'] - df['previous_month_end_balance']
    df['balance_volatility'] = abs(df['average_monthly_balance_prevq'] - df['average_monthly_balance_prevq2'])
    
    # Credit/Debit patterns - handle division by zero
    df['credit_debit_ratio'] = np.where(
        df['current_month_debit'] != 0,
        df['current_month_credit'] / df['current_month_debit'],
        0  # Set to 0 if debit is 0 to avoid inf values
    )
    df['credit_change'] = df['current_month_credit'] - df['previous_month_credit']
    df['debit_change'] = df['current_month_debit'] - df['previous_month_debit']
    
    # Account activity indicators
    df['is_dormant_account'] = ((df['current_month_credit'] == 0) & 
                               (df['current_month_debit'] == 0)).astype(int)
    df['avg_monthly_activity'] = (df['current_month_credit'] + df['current_month_debit']) / 2
    
    # Balance consistency - handle division by zero
    df['balance_consistency'] = np.where(
        df['average_monthly_balance_prevq'] != 0,
        1 - (abs(df['current_balance'] - df['average_monthly_balance_prevq']) / 
             abs(df['average_monthly_balance_prevq'])),
        0
    )
    
    # Process last_transaction for recency features
    if 'last_transaction' in df.columns:
        try:
            df['last_transaction'] = pd.to_datetime(df['last_transaction'])
            reference_date = df['last_transaction'].max()
            df['days_since_last_transaction'] = (reference_date - df['last_transaction']).dt.days
            df.drop('last_transaction', axis=1, inplace=True)
        except:
            print("   ⚠️ Could not process last_transaction datetime")
    
    # Replace any inf values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def check_class_balance(df, target):
    """Returns the distribution of classes in the target column."""
    return df[target].value_counts(normalize=True) * 100

def handle_class_imbalance(X, y, method="smote"):
    """Handle class imbalance using different resampling techniques."""
    if method == "smote":
        sampler = SMOTE(random_state=42)
    elif method == "oversample":
        sampler = RandomOverSampler(random_state=42)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("method must be one of ['smote', 'oversample', 'undersample']")

    X_res, y_res = sampler.fit_resample(X, y) # type: ignore
    return X_res, y_res

def process_data_for_classification(df, target_col='churn', test_size=0.2, random_state=42, 
                                  scaling_method='standard', imbalance_method='smote'):
    """
    Performs a full data preprocessing pipeline for churn prediction.
    """
    print("🚀 Starting Churn Prediction Preprocessing Pipeline")
    print("=" * 60)
    
    # Define feature lists based on the actual churn dataset
    numerical_features = [
        'vintage', 'age', 'dependents', 'city', 'current_balance', 
        'previous_month_end_balance', 'average_monthly_balance_prevq',
        'average_monthly_balance_prevq2', 'current_month_credit', 
        'previous_month_credit', 'current_month_debit', 'previous_month_debit',
        'current_month_balance', 'previous_month_balance'
    ]
    
    categorical_features = ['gender', 'occupation']
    ordinal_features = ['customer_nw_category', 'branch_code']
    
    print(f"📊 Initial dataset shape: {df.shape}")
    
    # Step 1: Clean column names
    df = clean_column_names(df)
    print("✅ Column names cleaned")
    
    # Step 2: Check missing values
    missing_info = check_missing_values(df)
    if missing_info['missing_count'].sum() > 0:
        print(f"📋 Missing values found:\n{missing_info[missing_info['missing_count'] > 0]}")
    
    # Step 3: Handle missing values and duplicates
    df = handle_missing_values(df, numerical_features, categorical_features)
    df = remove_duplicates(df)
    
    # Step 4: Feature engineering (before splitting to avoid data leakage in feature creation logic)
    df = engineer_features(df)
    
    # Update numerical features list with engineered features
    engineered_features = [
        'balance_change', 'balance_volatility', 'credit_debit_ratio',
        'credit_change', 'debit_change', 'is_dormant_account',
        'avg_monthly_activity', 'balance_consistency'
    ]
    
    if 'days_since_last_transaction' in df.columns:
        engineered_features.append('days_since_last_transaction')
    
    numerical_features.extend(engineered_features)
    
    # Step 5: Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")
    
    # Remove customer_id from features if present
    feature_cols = [col for col in df.columns if col not in [target_col, 'customer_id']]
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"🎯 Target distribution:\n{check_class_balance(df, target_col)}")
    
    # Step 6: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✅ Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 7: Handle outliers on training data only
    print("🎯 Handling outliers in training data...")
    outlier_features = [f for f in numerical_features if f in X_train.columns]
    for col in outlier_features:
        X_train = handle_outliers(X_train, column=col)
    
    # Step 8: Encode categorical features
    print("🔤 Encoding categorical features...")
    # One-hot encode categorical features
    categorical_in_data = [f for f in categorical_features if f in X_train.columns]
    if categorical_in_data:
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_in_data, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_in_data, drop_first=True)
        
        # Align columns between train and test
        train_cols = X_train_encoded.columns
        test_cols = X_test_encoded.columns
        
        missing_in_test = list(set(train_cols) - set(test_cols))
        for col in missing_in_test:
            X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[train_cols]
    else:
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
    
    # Step 9: Scale numerical features
    print("📏 Scaling numerical features...")
    numerical_in_encoded = [col for col in X_train_encoded.columns 
                           if col not in [f for sublist in [categorical_in_data] for f in sublist]]
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
        X_train_encoded[numerical_in_encoded] = scaler.fit_transform(X_train_encoded[numerical_in_encoded])
        X_test_encoded[numerical_in_encoded] = scaler.transform(X_test_encoded[numerical_in_encoded])
    
    # Step 9.5: Final check for any remaining NaN values
    print("🔍 Checking for any remaining NaN values...")
    if X_train_encoded.isnull().sum().sum() > 0:
        print("   ⚠️ Found remaining NaN values, filling with 0...")
        X_train_encoded = X_train_encoded.fillna(0)
    if X_test_encoded.isnull().sum().sum() > 0:
        X_test_encoded = X_test_encoded.fillna(0)
    
    # Step 10: Handle class imbalance on training data only
    print("⚖️ Handling class imbalance...")
    X_train_resampled, y_train_resampled = handle_class_imbalance(
        X_train_encoded, y_train, method=imbalance_method
    )
    
    resampled_distribution = pd.Series(y_train_resampled).value_counts(normalize=True) * 100 # type: ignore
    print(f"✅ Class distribution after {imbalance_method}:\n{resampled_distribution}")
    
    print(f"\n🎉 Preprocessing Complete!")
    print(f"📊 Final shapes - X_train: {X_train_resampled.shape}, X_test: {X_test_encoded.shape}")
    print(f"🎯 Features created: {X_train_encoded.shape[1]}")
    
    return X_train_resampled, X_test_encoded, y_train_resampled, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir="data/cleaned/"):
    """Saves the preprocessed data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert to DataFrame if needed
    if hasattr(X_train, 'toarray'):  # If sparse matrix from resampling
        X_train = pd.DataFrame(X_train.toarray(), columns=X_test.columns)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name='churn')
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name='churn')
    
    # Save files
    X_train.to_csv(os.path.join(output_dir, f"X_train_{timestamp}.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, f"X_test_{timestamp}.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, f"y_train_{timestamp}.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, f"y_test_{timestamp}.csv"), index=False)
    
    print(f"💾 Processed data saved to {output_dir}")

if __name__ == "__main__":
    try:
        # Load the raw churn prediction data
        df_raw = pd.read_csv(RAW_DATA_PATH)
        print(f"📥 Raw Data Shape: {df_raw.shape}")
        
        # Run the preprocessing pipeline
        X_train_resampled, X_test_encoded, y_train_resampled, y_test = process_data_for_classification(
            df=df_raw, 
            target_col='churn',
            test_size=0.2,
            random_state=42,
            scaling_method='standard',
            imbalance_method='smote'
        )
        
        # Save processed data
        save_processed_data(X_train_resampled, X_test_encoded, y_train_resampled, y_test)
        
        print("\n🚀 Pipeline executed successfully! Data ready for model training.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please ensure '{RAW_DATA_PATH}' exists.")
    except Exception as e:
        print(f"❌ An error occurred during pipeline execution: {e}")