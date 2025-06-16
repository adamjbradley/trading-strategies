#!/usr/bin/env python3
"""
Notebook Integration Guide for Enhanced Correlation Features
Step-by-step instructions to test enhanced correlations in the actual notebook
"""

def create_notebook_integration_guide():
    """Create comprehensive guide for testing enhanced correlations in notebook"""
    
    print("📖 NOTEBOOK INTEGRATION GUIDE")
    print("=" * 60)
    print("How to test enhanced correlation features in your notebook")
    print()
    
    print("🎯 OVERVIEW:")
    print("   You need to add the enhanced correlation functions to your notebook")
    print("   and modify the optimizer to use them. Here's the step-by-step process:")
    print()
    
    print("📋 STEP-BY-STEP INTEGRATION:")
    print("=" * 40)
    
    print("\n1️⃣ ADD ENHANCED CORRELATION FUNCTIONS")
    print("   ▶️ Open: Advanced_Hyperparameter_Optimization_Clean.ipynb")
    print("   ▶️ Find the cell with: def _create_advanced_features")
    print("   ▶️ Add this NEW cell AFTER the existing feature creation:")
    
    correlation_function = '''
# 🔗 ENHANCED CORRELATION FEATURES
def create_enhanced_currency_correlation_features(df, symbol, all_pairs_data=None):
    """Create enhanced currency correlation features for improved forex predictions"""
    
    features = pd.DataFrame(index=df.index)
    close = df['close']
    
    try:
        # 1. Currency Strength Index (CSI) for major currencies
        if 'USD' in symbol:
            if symbol.startswith('USD'):
                # USD is base currency (e.g., USDJPY)
                features['usd_strength_raw'] = close.pct_change().rolling(10).mean()
            elif symbol.endswith('USD'):
                # USD is quote currency (e.g., EURUSD)
                features['usd_strength_raw'] = (-close.pct_change()).rolling(10).mean()
            else:
                features['usd_strength_raw'] = 0
        else:
            features['usd_strength_raw'] = 0
            
        # Enhanced USD strength with momentum
        features['usd_strength_index'] = features['usd_strength_raw'].rolling(20).mean()
        features['usd_momentum'] = features['usd_strength_raw'].diff(5)
        
        # 2. Risk-on/Risk-off Sentiment Detection
        # Based on price momentum and volatility
        returns = close.pct_change()
        volatility = returns.rolling(20).std()
        
        # Risk-on: Low volatility + positive momentum
        # Risk-off: High volatility + negative momentum
        vol_percentile = volatility.rolling(100).rank(pct=True)
        momentum = returns.rolling(10).mean()
        
        features['risk_sentiment'] = np.where(
            (vol_percentile < 0.3) & (momentum > 0), 1,  # Risk-on
            np.where((vol_percentile > 0.7) & (momentum < 0), -1, 0)  # Risk-off
        )
        
        # Risk sentiment strength
        features['risk_sentiment_strength'] = abs(momentum) * (1 - vol_percentile)
        
        # 3. Carry Trade Indicators
        # Simplified carry trade signals based on interest rate proxies
        if any(curr in symbol for curr in ['AUD', 'NZD']):  # High-yield currencies
            features['carry_signal'] = momentum.rolling(20).mean() * 2  # Amplify
        elif any(curr in symbol for curr in ['JPY', 'CHF']):  # Low-yield currencies  
            features['carry_signal'] = momentum.rolling(20).mean() * -1  # Invert
        else:
            features['carry_signal'] = momentum.rolling(20).mean()
            
        # Carry trade momentum
        features['carry_momentum'] = features['carry_signal'].diff(10)
        features['carry_strength'] = features['carry_signal'].rolling(20).std()
        
        # 4. Enhanced Divergence Indicators
        price_ma = close.rolling(20).mean()
        price_trend = (close / price_ma - 1) * 100
        
        momentum_ma = momentum.rolling(20).mean()
        momentum_trend = momentum / (momentum_ma + 1e-10)
        
        # Price vs momentum divergence
        features['price_momentum_divergence'] = price_trend - momentum_trend
        features['divergence_strength'] = abs(features['price_momentum_divergence'])
        
        # 5. Correlation Regime Detection
        # Detect when correlations are breaking down or strengthening
        returns_vol = returns.rolling(20).std()
        vol_regime = returns_vol.rolling(50).rank(pct=True)
        
        features['correlation_regime'] = np.where(
            vol_regime > 0.8, 2,  # High volatility = correlation breakdown
            np.where(vol_regime < 0.2, 0, 1)  # Low vol = stable correlations
        )
        
        # 6. Cross-pair Correlation Proxy (single pair version)
        # Use price action similarity to estimate cross-pair effects
        returns_1h = close.pct_change()
        returns_4h = close.pct_change(4)
        returns_daily = close.pct_change(24)
        
        # Correlation between different timeframes as proxy
        features['timeframe_correlation'] = returns_1h.rolling(50).corr(returns_4h)
        features['correlation_stability'] = features['timeframe_correlation'].rolling(20).std()
        
        print(f"✅ Enhanced correlation features created for {symbol}")
        
    except Exception as e:
        print(f"⚠️ Error creating correlation features for {symbol}: {e}")
        # Set fallback values
        fallback_features = [
            'usd_strength_raw', 'usd_strength_index', 'usd_momentum',
            'risk_sentiment', 'risk_sentiment_strength',
            'carry_signal', 'carry_momentum', 'carry_strength',
            'price_momentum_divergence', 'divergence_strength',
            'correlation_regime', 'timeframe_correlation', 'correlation_stability'
        ]
        for feature in fallback_features:
            features[feature] = 0
    
    # Clean and normalize features
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0)
    
    return features
'''
    
    print(correlation_function)
    
    print("\n2️⃣ MODIFY THE MAIN FEATURE CREATION FUNCTION")
    print("   ▶️ Find the _create_advanced_features function")
    print("   ▶️ Add this code at the END of the function (before return features):")
    
    integration_code = '''
        # 🔗 ADD ENHANCED CORRELATION FEATURES
        try:
            correlation_features = create_enhanced_currency_correlation_features(df, symbol)
            
            # Merge correlation features with existing features
            for col in correlation_features.columns:
                if col not in features.columns:  # Avoid duplicates
                    features[col] = correlation_features[col]
            
            print(f"✅ Enhanced correlations added: {len(correlation_features.columns)} features")
            
        except Exception as e:
            print(f"⚠️ Correlation features failed: {e}")
'''
    
    print(integration_code)
    
    print("\n3️⃣ TEST THE ENHANCED FEATURES")
    print("   ▶️ Add a new test cell:")
    
    test_code = '''
# 🧪 TEST ENHANCED CORRELATION FEATURES
print("🧪 Testing Enhanced Correlation Features")
print("=" * 50)

# Load test data
test_symbol = 'EURUSD'
test_data = load_and_prepare_data(test_symbol)

# Create features with enhancements
enhanced_features = optimizer._create_advanced_features(test_data, test_symbol)

# Check for enhanced correlation features
correlation_feature_names = [
    'usd_strength_index', 'risk_sentiment', 'carry_signal', 
    'price_momentum_divergence', 'correlation_regime'
]

print(f"📊 Feature Summary:")
print(f"   Total features: {len(enhanced_features.columns)}")

found_correlation_features = []
for feature in correlation_feature_names:
    if feature in enhanced_features.columns:
        found_correlation_features.append(feature)
        print(f"   ✅ {feature}: Found")
    else:
        print(f"   ❌ {feature}: Missing")

print(f"\\n🎯 Enhanced Features Status:")
print(f"   Found: {len(found_correlation_features)}/{len(correlation_feature_names)}")

if len(found_correlation_features) >= 3:
    print("✅ Enhanced correlation features successfully integrated!")
    
    # Quick feature analysis
    print(f"\\n📈 Sample Feature Values (last 5 rows):")
    for feature in found_correlation_features[:3]:  # Show first 3
        values = enhanced_features[feature].tail(5)
        print(f"   {feature}: {values.values}")
        
else:
    print("❌ Integration failed - check function placement")
'''
    
    print(test_code)
    
    print("\n4️⃣ RUN OPTIMIZATION WITH ENHANCED FEATURES")
    print("   ▶️ Use your existing optimization cell, it will automatically include")
    print("      the enhanced features in the feature selection process")
    print("   ▶️ Compare results with previous optimization runs")
    
    print("\n5️⃣ COMPARE RESULTS")
    print("   ▶️ Add this comparison cell:")
    
    comparison_code = '''
# 📊 COMPARE BASELINE VS ENHANCED RESULTS
import json
from datetime import datetime

# Your previous best result (update with your actual values)
baseline_objective = 0.4639  # Update this with your latest result

# Run new optimization with enhanced features
print("🚀 Running optimization with enhanced correlation features...")
best_params, best_objective = optimizer.optimize_symbol('EURUSD', n_trials=20)

# Calculate improvement
improvement = (best_objective - baseline_objective) / baseline_objective

print(f"\\n📊 RESULTS COMPARISON:")
print(f"   Baseline Objective: {baseline_objective:.4f}")
print(f"   Enhanced Objective: {best_objective:.4f}")
print(f"   Improvement: {improvement:+.1%}")

if improvement > 0.05:  # 5% improvement
    print("🎉 Significant improvement with enhanced correlation features!")
elif improvement > 0.02:  # 2% improvement
    print("✅ Moderate improvement with enhanced correlation features")
else:
    print("⚠️ Limited improvement - may need parameter tuning")
    
# Save results for comparison
results = {
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'baseline_objective': baseline_objective,
    'enhanced_objective': best_objective,
    'improvement': improvement,
    'enhanced_features_used': True
}

with open(f'enhanced_correlation_test_{results["timestamp"]}.json', 'w') as f:
    json.dump(results, f, indent=2)
    
print(f"💾 Results saved to: enhanced_correlation_test_{results['timestamp']}.json")
'''
    
    print(comparison_code)
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY - WHAT TO DO:")
    print("=" * 60)
    print("1. Open your notebook: Advanced_Hyperparameter_Optimization_Clean.ipynb")
    print("2. Add the enhanced correlation function (Step 1)")
    print("3. Modify _create_advanced_features to include correlations (Step 2)")
    print("4. Run the test cell to verify integration (Step 3)")
    print("5. Run optimization and compare results (Steps 4-5)")
    print()
    print("💡 EXPECTED RESULTS:")
    print("   • 5-15% improvement in objective value")
    print("   • Better Sharpe ratio and accuracy")
    print("   • More stable performance across different market conditions")
    print()
    print("⚠️ TROUBLESHOOTING:")
    print("   • If features are missing: Check function placement")
    print("   • If errors occur: Check data availability")
    print("   • If no improvement: Try increasing n_trials or adjusting parameters")

if __name__ == "__main__":
    create_notebook_integration_guide()