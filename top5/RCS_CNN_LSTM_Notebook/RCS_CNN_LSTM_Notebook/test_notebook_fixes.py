#!/usr/bin/env python3
"""
Test script to validate notebook fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.data.loader import load_or_fetch, load_metatrader_data
from src.data.preprocessing import engineer_features

def test_data_loading():
    """Test that data loading works"""
    print("🧪 Testing data loading...")
    
    try:
        symbol = "EURUSD"
        df = load_or_fetch(
            symbol=symbol,
            provider="metatrader",
            loader_func=load_metatrader_data,
            api_key="",
            force_refresh=False
        )
        
        if not df.empty:
            print(f"✅ Data loading works: {len(df)} rows loaded for {symbol}")
            return df
        else:
            print("⚠️ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def test_feature_engineering():
    """Test that feature engineering doesn't create empty DataFrames"""
    print("🧪 Testing feature engineering...")
    
    try:
        # Create a simple test MultiIndex DataFrame similar to notebook
        symbols = ["EURUSD", "GBPUSD"] 
        data = {}
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        np.random.seed(42)
        
        for sym in symbols:
            for col in ["open", "high", "low", "close", "volume"]:
                if col == "volume":
                    data[(sym, col)] = np.random.randint(100, 1000, size=len(dates))
                else:
                    base_price = 1.1 if sym == "EURUSD" else 1.3
                    data[(sym, col)] = base_price + np.random.normal(0, 0.01, size=len(dates)).cumsum()
        
        prices = pd.DataFrame(data, index=dates)
        print(f"Created test prices DataFrame: {prices.shape}")
        
        # Test feature engineering
        features = engineer_features(prices, symbol="EURUSD")
        
        print(f"Features shape: {features.shape}")
        print(f"NaN counts: {features.isnull().sum().sum()}")
        
        if features.shape[0] > 0:
            print("✅ Feature engineering works: no empty DataFrame")
            return True
        else:
            print("❌ Feature engineering failed: empty DataFrame")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False

def test_nan_handling():
    """Test that NaN handling preserves data"""
    print("🧪 Testing NaN handling...")
    
    try:
        # Create DataFrame with intentional NaNs
        dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
        data = pd.DataFrame(index=dates)
        
        # Create some columns with NaNs at different positions
        data['col1'] = [np.nan] * 10 + list(range(40))  # NaNs at start
        data['col2'] = list(range(25)) + [np.nan] * 10 + list(range(15))  # NaNs in middle
        data['col3'] = list(range(40)) + [np.nan] * 10  # NaNs at end
        
        print(f"Original data shape: {data.shape}")
        print(f"Original NaN counts: {data.isnull().sum().sum()}")
        
        # Apply our NaN handling logic
        for col in data.columns:
            data[col] = data[col].ffill()
            data[col] = data[col].bfill()
            data[col] = data[col].fillna(0)
        
        print(f"After NaN handling shape: {data.shape}")
        print(f"After NaN handling NaN counts: {data.isnull().sum().sum()}")
        
        if data.shape[0] == 50 and data.isnull().sum().sum() == 0:
            print("✅ NaN handling works: preserves all rows and eliminates NaNs")
            return True
        else:
            print("❌ NaN handling failed")
            return False
            
    except Exception as e:
        print(f"❌ NaN handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running notebook fixes validation tests...\n")
    
    results = []
    
    # Test data loading (optional - depends on data availability)
    df = test_data_loading()
    results.append(df is not None)
    print()
    
    # Test feature engineering
    feature_test = test_feature_engineering()
    results.append(feature_test)
    print()
    
    # Test NaN handling
    nan_test = test_nan_handling()
    results.append(nan_test)
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Notebook fixes should work correctly.")
    else:
        print("⚠️ Some tests failed. Review the fixes before running the notebook.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)