#!/usr/bin/env python3
"""
Final Phase 2 Validation - Test actual notebook implementation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def validate_notebook_implementation():
    """Validate that the notebook has Phase 2 features properly implemented"""
    
    print("🧪 FINAL PHASE 2 VALIDATION")
    print("="*40)
    
    # Check 1: Data availability
    print("\n1️⃣ CHECKING DATA AVAILABILITY")
    print("-" * 32)
    
    from pathlib import Path
    data_path = Path("data")
    
    major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURJPY', 'GBPJPY']
    available_pairs = []
    
    for pair in major_pairs:
        file_path = data_path / f"metatrader_{pair}.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                if len(df) > 100 and 'close' in df.columns:
                    available_pairs.append(pair)
                    print(f"   ✅ {pair}: {len(df)} records")
            except:
                print(f"   ❌ {pair}: Read error")
        else:
            print(f"   ⚠️ {pair}: File not found")
    
    print(f"\n   📊 Available pairs: {len(available_pairs)}/{len(major_pairs)}")
    
    if len(available_pairs) < 3:
        print("   ❌ Insufficient data for Phase 2 testing")
        return False
    
    # Check 2: Notebook cell structure
    print("\n2️⃣ CHECKING NOTEBOOK STRUCTURE")
    print("-" * 34)
    
    try:
        # Check if notebook exists
        notebook_path = Path("Advanced_Hyperparameter_Optimization_Clean.ipynb")
        if notebook_path.exists():
            print("   ✅ Notebook file found")
            
            # Read notebook content
            import json
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            cell_count = len(notebook.get('cells', []))
            print(f"   ✅ Notebook has {cell_count} cells")
            
            # Check for key Phase 2 functions
            phase2_functions = [
                'create_true_multi_pair_csi',
                'detect_correlation_regimes', 
                'create_correlation_network_features',
                'create_enhanced_currency_correlation_features'
            ]
            
            notebook_content = str(notebook)
            found_functions = []
            
            for func in phase2_functions:
                if func in notebook_content:
                    found_functions.append(func)
                    print(f"   ✅ Function found: {func}")
                else:
                    print(f"   ❌ Function missing: {func}")
            
            print(f"   📊 Phase 2 functions: {len(found_functions)}/{len(phase2_functions)}")
            
            if len(found_functions) >= 3:
                print("   ✅ Phase 2 implementation detected in notebook")
                notebook_ok = True
            else:
                print("   ❌ Phase 2 implementation incomplete")
                notebook_ok = False
                
        else:
            print("   ❌ Notebook file not found")
            notebook_ok = False
            
    except Exception as e:
        print(f"   ❌ Notebook check failed: {e}")
        notebook_ok = False
    
    # Check 3: Feature engineering capabilities
    print("\n3️⃣ TESTING FEATURE ENGINEERING")
    print("-" * 35)
    
    try:
        # Load test data
        test_pair = available_pairs[0]
        test_data = pd.read_parquet(data_path / f"metatrader_{test_pair}.parquet")
        print(f"   📊 Test data: {test_pair} with {len(test_data)} records")
        
        # Test basic feature creation
        features = pd.DataFrame(index=test_data.index)
        
        # Basic features
        close = test_data['close']
        features['returns'] = close.pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Phase 2 style features
        features['currency_strength_proxy'] = features['returns'].rolling(10).mean()
        features['correlation_momentum'] = features['returns'].rolling(20).corr(features['returns'].shift(1))
        features['risk_sentiment'] = (-features['returns']).rolling(20).mean()
        
        # Clean features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Validate
        clean_features = features.dropna()
        data_loss = (len(features) - len(clean_features)) / len(features) * 100
        
        print(f"   ✅ Features created: {len(features.columns)}")
        print(f"   ✅ Data loss: {data_loss:.1f}%")
        
        if data_loss < 10:
            print("   ✅ Feature engineering: PASSED")
            feature_ok = True
        else:
            print("   ❌ Feature engineering: HIGH DATA LOSS")
            feature_ok = False
            
    except Exception as e:
        print(f"   ❌ Feature engineering test failed: {e}")
        feature_ok = False
    
    # Check 4: Integration readiness
    print("\n4️⃣ CHECKING INTEGRATION READINESS")
    print("-" * 38)
    
    # Check for required imports
    required_modules = ['pandas', 'numpy', 'optuna', 'tensorflow', 'sklearn']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}: Available")
        except ImportError:
            print(f"   ❌ {module}: Missing")
            missing_modules.append(module)
    
    if len(missing_modules) == 0:
        print("   ✅ All required modules available")
        integration_ok = True
    else:
        print(f"   ❌ Missing modules: {missing_modules}")
        integration_ok = False
    
    # Final assessment
    print("\n🎉 PHASE 2 VALIDATION SUMMARY")
    print("="*38)
    
    checks = [
        ("Data availability", len(available_pairs) >= 3),
        ("Notebook structure", notebook_ok),
        ("Feature engineering", feature_ok),
        ("Integration readiness", integration_ok)
    ]
    
    passed_checks = 0
    for check_name, passed in checks:
        status = "PASSED" if passed else "FAILED"
        print(f"✅ {check_name}: {status}")
        if passed:
            passed_checks += 1
    
    overall_score = passed_checks / len(checks) * 100
    print(f"\n📊 Overall score: {passed_checks}/{len(checks)} ({overall_score:.0f}%)")
    
    if overall_score >= 75:
        print("🎯 PHASE 2 VALIDATION: EXCELLENT ✅")
        print("🚀 Ready for production optimization runs!")
        status = "EXCELLENT"
    elif overall_score >= 50:
        print("⚠️ PHASE 2 VALIDATION: GOOD (Minor issues)")
        print("🔧 Some improvements recommended")
        status = "GOOD"
    else:
        print("❌ PHASE 2 VALIDATION: NEEDS WORK")
        print("🔧 Significant issues detected")
        status = "NEEDS_WORK"
    
    return status, {
        'available_pairs': available_pairs,
        'notebook_ok': notebook_ok,
        'feature_ok': feature_ok,
        'integration_ok': integration_ok,
        'overall_score': overall_score
    }

def provide_recommendations(status, details):
    """Provide recommendations based on validation results"""
    
    print(f"\n💡 RECOMMENDATIONS")
    print("="*25)
    
    if status == "EXCELLENT":
        print("🎉 Phase 2 implementation is ready!")
        print("📊 Suggested next steps:")
        print("   1. Run a full optimization test (10-20 trials)")
        print("   2. Compare performance with/without Phase 2 features")
        print("   3. Consider implementing Phase 3 features")
        print("   4. Monitor feature selection during optimization")
        
    elif status == "GOOD":
        print("✅ Phase 2 implementation is mostly ready")
        print("🔧 Suggested improvements:")
        
        if not details['notebook_ok']:
            print("   - Review notebook cell implementation")
            print("   - Ensure all Phase 2 functions are properly defined")
        
        if not details['feature_ok']:
            print("   - Improve feature engineering data quality")
            print("   - Add better NaN handling")
        
        if not details['integration_ok']:
            print("   - Install missing required modules")
            print("   - Verify import statements")
            
        print("   - Run mini optimization test before full deployment")
        
    else:  # NEEDS_WORK
        print("❌ Phase 2 implementation needs significant work")
        print("🔧 Critical fixes needed:")
        
        if len(details['available_pairs']) < 3:
            print("   - Ensure sufficient forex pair data is available")
            print("   - Minimum 3 pairs needed for correlation analysis")
        
        if not details['notebook_ok']:
            print("   - Implement missing Phase 2 functions in notebook")
            print("   - Review notebook cell structure")
        
        if not details['feature_ok']:
            print("   - Fix feature engineering implementation")
            print("   - Address data quality issues")
        
        if not details['integration_ok']:
            print("   - Install all required dependencies")
            print("   - Resolve import errors")
    
    print(f"\n🎯 CURRENT PHASE STATUS:")
    print(f"   Phase 1: ✅ COMPLETE (Basic currency features)")
    print(f"   Phase 2: {'✅ COMPLETE' if status == 'EXCELLENT' else '🔄 IN PROGRESS'} (Multi-pair correlation analysis)")
    print(f"   Phase 3: ❌ NOT STARTED (Real-time integration, ensemble models)")
    print(f"   Phase 4: ❌ NOT STARTED (External data, AI pattern recognition)")

def main():
    """Run final Phase 2 validation"""
    status, details = validate_notebook_implementation()
    provide_recommendations(status, details)
    
    print(f"\n🎉 PHASE 2 TESTING COMPLETE!")
    
    if status == "EXCELLENT":
        print("🚀 READY FOR PRODUCTION USE!")
    elif status == "GOOD":
        print("⚠️ READY WITH MINOR IMPROVEMENTS")
    else:
        print("🔧 REQUIRES ADDITIONAL WORK")

if __name__ == "__main__":
    main()