import requests
import json

# API base URL
BASE_URL = "http://churnpoc-api.westeurope.azurecontainer.io:8000"


def test_health_check():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("🏥 Health Check:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://127.0.0.1:8000")
        return False

def test_single_prediction():
    """Test single customer prediction"""
    print("\n🔮 Testing Single Prediction:")
    
    # Example customer data - matching all required fields from preprocessing
    customer_data = {
    "age": 42,  # mid-age group, typically higher churn risk if dissatisfied
    "gender": "male",
    "vintage": 6,  # relatively NEW customer (low loyalty)
    "current_balance": 500.0,  # very low balance
    "previous_month_end_balance": 2000.0,  # sharp decline
    "average_monthly_balance_prevq": 2200.0,  # was higher in previous quarter
    "average_monthly_balance_prevq2": 2400.0,
    "current_month_balance": 500.0,  # still low
    "previous_month_balance": 2000.0,
    "current_month_credit": 600.0,  # low inflow
    "current_month_debit": 1200.0,  # spending more than credit inflow
    "previous_month_credit": 1500.0,  # inflow dropping
    "previous_month_debit": 1300.0,
    "dependents": 3,  # higher financial burden
    "occupation": "self_employed",  # often higher churn segment (unstable income)
    "customer_nw_category": 3,  # lowest NW category (high-risk)
    "city": 101,
    "branch_code": 1001
}

    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=customer_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   🎯 Churn Probability: {result['churn_probability']:.2%}")
            print(f"   📊 Prediction: {result['churn_prediction']}")
            print(f"   ⚠️  Risk Level: {result['risk_level']}")
            print(f"   🎯 Confidence: {result['confidence']:.2%}")
            print("   📋 Key Factors:")
            for factor in result['key_factors']:
                print(f"      • {factor}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")

def test_batch_prediction():
    """Test batch prediction with multiple customers"""
    print("\n👥 Testing Batch Prediction:")
    
    customers = [
        {
            "age": 28,
            "gender": "female",
            "vintage": 6,  # New customer
            "current_balance": 500.0,  # Low balance
            "previous_month_end_balance": 600.0,
            "average_monthly_balance_prevq": 550.0,
            "average_monthly_balance_prevq2": 580.0,
            "current_month_balance": 500.0,
            "previous_month_balance": 600.0,
            "current_month_credit": 200.0,
            "current_month_debit": 300.0,
            "previous_month_credit": 250.0,
            "previous_month_debit": 200.0,
            "dependents": 0,
            "occupation": "student",
            "customer_nw_category": 1,
            "city": 201,
            "branch_code": 2001
        },
        {
            "age": 45,
            "gender": "male",
            "vintage": 48,  # Long-term customer
            "current_balance": 15000.0,  # High balance
            "previous_month_end_balance": 14500.0,
            "average_monthly_balance_prevq": 14800.0,
            "average_monthly_balance_prevq2": 14600.0,
            "current_month_balance": 15000.0,
            "previous_month_balance": 14500.0,
            "current_month_credit": 5000.0,
            "current_month_debit": 3000.0,
            "previous_month_credit": 4800.0,
            "previous_month_debit": 2900.0,
            "dependents": 3,
            "occupation": "manager",
            "customer_nw_category": 3,
            "city": 301,
            "branch_code": 3001
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=customers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful! Processed {result['total_customers']} customers")
            
            for i, prediction in enumerate(result['predictions']):
                print(f"\n   Customer {i+1}:")
                print(f"     🎯 Churn Probability: {prediction['churn_probability']:.2%}")
                print(f"     📊 Prediction: {prediction['churn_prediction']}")
                print(f"     ⚠️  Risk Level: {prediction['risk_level']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error making batch prediction: {e}")

def test_model_info():
    """Test getting model information"""
    print("\n🤖 Testing Model Info:")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            result = response.json()
            print("✅ Model info retrieved!")
            print(f"   🧠 Model Type: {result['model_type']}")
            print(f"   📊 Features Count: {result['features_count']}")
            
            if 'top_features' in result:
                print("   🏆 Top Important Features:")
                for feature, importance in result['top_features']:
                    print(f"      • {feature}: {importance:.4f}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error getting model info: {e}")

def get_example():
    """Get API usage example"""
    print("\n📖 Getting API Example:")
    
    try:
        response = requests.get(f"{BASE_URL}/example")
        if response.status_code == 200:
            result = response.json()
            print("✅ Example retrieved!")
            print("\n📝 Example Request:")
            print(json.dumps(result['example_request'], indent=2))
            print("\n🛠️ Available Endpoints:")
            for endpoint, description in result['usage'].items():
                print(f"   • {description}: {endpoint}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting example: {e}")

def debug_features():
    """Debug what features the model expects vs what we create"""
    print("\n🔍 Debugging Feature Alignment:")
    
    try:
        # Get expected features
        response = requests.get(f"{BASE_URL}/debug/expected_features")
        if response.status_code == 200:
            expected = response.json()
            print("✅ Expected features retrieved!")
            print(f"   📊 Total expected: {expected['total_features']}")
            print(f"   🏷️  Categorical: {len(expected['categorical_features'])}")
            print(f"   🛠️  Engineered: {len(expected['engineered_features'])}")
            print(f"   📋 Original: {len(expected['original_features'])}")
            
            print("\n   🏷️ Categorical features expected:")
            for feat in expected['categorical_features']:
                print(f"      • {feat}")
                
        # Test what we create
        test_customer = {
            "age": 35,
            "gender": "Male",  # Try with capital M to match expected
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
            "occupation": "salaried",  # Try a different occupation
            "customer_nw_category": 2,
            "city": 101,
            "branch_code": 1001
        }
        
        response = requests.post(f"{BASE_URL}/debug/show_my_features", json=test_customer)
        if response.status_code == 200:
            created = response.json()
            print(f"\n✅ Created features analyzed!")
            print(f"   📊 Total created: {created['features_count']}")
            print(f"   📋 Sample values: {created['sample_values']}")
            
        print("\n💡 This will help identify the mismatch!")
            
    except Exception as e:
        print(f"❌ Error debugging features: {e}")

def main():
    """Run all tests"""
    print("🚀 Testing Churn Prediction API")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        return
    
    # Debug features first
    debug_features()
    
    # Run other tests
    test_single_prediction()
    test_batch_prediction()
    test_model_info()
    get_example()
    
    print("\n🎉 All tests completed!")
    print("\n💡 Try these commands:")
    print("   • curl http://localhost:8000/health")
    print("   • curl http://localhost:8000/debug/expected_features")
    print("   • Open http://localhost:8000/docs in your browser for interactive API docs")

if __name__ == "__main__":
    main()
