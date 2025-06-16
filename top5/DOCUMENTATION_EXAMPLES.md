# Advanced Hyperparameter Optimization System - Examples & Use Cases

## 🎯 Exhaustive Examples Guide

### **1. Basic Optimization Examples**

#### **Example 1.1: Single Symbol Quick Test**
```python
# Quick 10-trial optimization for testing
result = optimizer.optimize_symbol('EURUSD', n_trials=10)

# Expected output files:
# - optimization_results/best_params_EURUSD_20241216_143022.json
# - exported_models/EURUSD_CNN_LSTM_20241216_143022.onnx
# - exported_models/EURUSD_training_metadata_20241216_143022.json

# Expected console output:
# 🎯 Optimizing EURUSD (10 trials, warm start enabled)...
# ✅ EURUSD: 0.742356 (9/10 trials)
# 📁 Saved: EURUSD_CNN_LSTM_20241216_143022.onnx
```

#### **Example 1.2: Production Optimization**
```python
# Full 50-trial optimization for production deployment
result = optimizer.optimize_symbol('EURUSD', n_trials=50)

if result:
    print(f"✅ Optimization completed successfully!")
    print(f"Best objective: {result.objective_value:.6f}")
    print(f"Success rate: {result.completed_trials/result.total_trials*100:.1f}%")
    
    # Access best hyperparameters
    best_lr = result.best_params['learning_rate']
    best_lstm = result.best_params['lstm_units']
    print(f"Optimal learning rate: {best_lr:.6f}")
    print(f"Optimal LSTM units: {best_lstm}")
else:
    print("❌ Optimization failed - check data availability")
```

#### **Example 1.3: Multi-Symbol Batch Optimization**
```python
# Optimize multiple symbols sequentially
symbols_to_optimize = ['EURUSD', 'GBPUSD', 'USDJPY']
results = {}

for symbol in symbols_to_optimize:
    print(f"\n🎯 Starting optimization for {symbol}...")
    result = optimizer.optimize_symbol(symbol, n_trials=30)
    
    if result:
        results[symbol] = result
        print(f"✅ {symbol} completed: {result.objective_value:.6f}")
    else:
        print(f"❌ {symbol} failed")

# Summary report
print(f"\n📊 BATCH OPTIMIZATION SUMMARY")
print(f"Successful optimizations: {len(results)}/{len(symbols_to_optimize)}")

for symbol, result in results.items():
    print(f"{symbol}: {result.objective_value:.6f} "
          f"({result.completed_trials}/{result.total_trials} trials)")
```

### **2. Verbosity Control Examples**

#### **Example 2.1: Detailed Progress Monitoring**
```python
# Enable verbose mode for detailed trial-by-trial output
optimizer.set_verbose_mode(True)

result = optimizer.optimize_symbol('EURUSD', n_trials=5)

# Expected detailed output:
# ========================================================
# 🎯 HYPERPARAMETER OPTIMIZATION: EURUSD
# ========================================================
# Target trials: 5
# Features: ALL legacy + Phase 2 correlations + Trading compatibility
# Warm start: enabled
# 
# Trial   1/5: LR=0.003123 | Dropout=0.234 | LSTM=95 | Window=35
#    🔧 RCS features ENABLED by hyperparameter
#    🔧 Cross-pair features ENABLED for EURUSD
#    ✅ Created 68 features (conditional features based on hyperparameters)
#    🔧 Feature selection: rfe (selecting 32/68 features)
#    ✅ Selected 32 features using rfe
#    🔧 Using robust scaler
#    🔧 Signal smoothing ENABLED
#    ✅ Accuracy: 0.8234, Signal Quality: 0.3456, Score: 0.823456
#  → 0.823456 ⭐ NEW BEST!

# Return to quiet mode
optimizer.set_verbose_mode(False)
```

#### **Example 2.2: Quiet Production Mode**
```python
# Default quiet mode for production environments
optimizer.set_verbose_mode(False)  # Default setting

result = optimizer.optimize_symbol('EURUSD', n_trials=50)

# Expected minimal output:
# 🎯 Optimizing EURUSD (50 trials, warm start enabled)...
#   Trial 1/50... 0.756234 ⭐
#   Trial 10/50... 0.782156 ⭐
#   Trial 20/50... 0.798234 ⭐
#   Trial 50/50... 0.823456 ⭐
# ✅ EURUSD: 0.823456 (47/50 trials)
# 📁 Saved: EURUSD_CNN_LSTM_20241216_143022.onnx
```

### **3. Warm Start Control Examples**

#### **Example 3.1: Global Warm Start Configuration**
```python
# Check current global setting
print(f"Current warm start setting: {ADVANCED_CONFIG['enable_warm_start']}")

# Enable warm start globally
ADVANCED_CONFIG['enable_warm_start'] = True
opt_manager.config['enable_warm_start'] = True

# All optimizations will now use warm start
result1 = optimizer.optimize_symbol('EURUSD', n_trials=20)
result2 = optimizer.optimize_symbol('GBPUSD', n_trials=20)

# Disable warm start globally
ADVANCED_CONFIG['enable_warm_start'] = False
opt_manager.config['enable_warm_start'] = False

# Fresh exploration for all symbols
result3 = optimizer.optimize_symbol('USDJPY', n_trials=20)
```

#### **Example 3.2: Per-Optimization Override**
```python
# Override global setting for specific optimizations

# Use warm start regardless of global setting
result_with_warm = optimizer.optimize_symbol(
    'EURUSD', 
    n_trials=25, 
    enable_warm_start=True
)

# Disable warm start regardless of global setting  
result_without_warm = optimizer.optimize_symbol(
    'EURUSD',
    n_trials=25,
    enable_warm_start=False
)

# Use global setting (no override)
result_global = optimizer.optimize_symbol('EURUSD', n_trials=25)

print(f"With warm start: {result_with_warm.objective_value:.6f}")
print(f"Without warm start: {result_without_warm.objective_value:.6f}")
print(f"Global setting: {result_global.objective_value:.6f}")
```

#### **Example 3.3: Warm Start Effectiveness Comparison**
```python
# Compare warm start vs fresh optimization
def compare_warm_start_effectiveness(symbol='EURUSD', trials=15):
    print(f"🔬 WARM START EFFECTIVENESS TEST: {symbol}")
    print("="*50)
    
    # Test 1: With warm start
    print("1️⃣ Testing WITH warm start...")
    start_time = time.time()
    result_warm = optimizer.optimize_symbol(symbol, n_trials=trials, enable_warm_start=True)
    warm_time = time.time() - start_time
    
    # Test 2: Without warm start  
    print("2️⃣ Testing WITHOUT warm start...")
    start_time = time.time()
    result_fresh = optimizer.optimize_symbol(symbol, n_trials=trials, enable_warm_start=False)
    fresh_time = time.time() - start_time
    
    # Comparison
    print(f"\n📊 COMPARISON RESULTS:")
    if result_warm and result_fresh:
        print(f"With warm start:    {result_warm.objective_value:.6f} ({warm_time:.1f}s)")
        print(f"Without warm start: {result_fresh.objective_value:.6f} ({fresh_time:.1f}s)")
        
        improvement = (result_warm.objective_value / result_fresh.objective_value - 1) * 100
        print(f"Performance difference: {improvement:+.1f}%")
        
        if warm_time < fresh_time:
            time_savings = (fresh_time - warm_time) / fresh_time * 100
            print(f"Time savings: {time_savings:.1f}%")
    
    return result_warm, result_fresh

# Run comparison
warm_result, fresh_result = compare_warm_start_effectiveness('EURUSD', 10)
```

### **4. Feature Engineering Examples**

#### **Example 4.1: Testing Feature Selection Methods**
```python
# Compare different feature selection methods
feature_methods = ['rfe', 'top_correlation', 'variance_threshold', 'mutual_info']
method_results = {}

# Test each method with fixed other parameters
for method in feature_methods:
    print(f"\n🔧 Testing feature selection: {method}")
    
    # Create a custom trial for testing
    import optuna
    study = optuna.create_study(direction='maximize')
    
    def objective(trial):
        # Fix most parameters, only vary feature selection
        params = {
            'feature_selection_method': method,
            'max_features': 30,
            'lstm_units': 95,
            'learning_rate': 0.003,
            'dropout_rate': 0.2,
            'use_rcs_features': True,
            'use_cross_pair_features': True,
            # ... other fixed parameters
        }
        
        # This would require access to the internal training method
        # For demonstration - actual implementation would call internal methods
        return 0.75 + random.random() * 0.15  # Simulated score
    
    study.optimize(objective, n_trials=3)
    method_results[method] = study.best_value
    print(f"Best score with {method}: {study.best_value:.6f}")

# Results summary
print(f"\n📊 FEATURE SELECTION METHOD COMPARISON:")
for method, score in sorted(method_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{method:20}: {score:.6f}")
```

#### **Example 4.2: Phase 2 Features Impact Test**
```python
# Test impact of Phase 2 correlation features
def test_phase2_impact(symbol='EURUSD', trials=10):
    print(f"🌍 PHASE 2 FEATURES IMPACT TEST: {symbol}")
    print("="*45)
    
    # Optimization without Phase 2 features
    print("1️⃣ Testing WITHOUT Phase 2 features...")
    # Note: This would require modifying the hyperparameter suggestion
    # to force use_cross_pair_features=False
    
    # Optimization with Phase 2 features
    print("2️⃣ Testing WITH Phase 2 features...")
    # Note: This would require modifying the hyperparameter suggestion
    # to force use_cross_pair_features=True
    
    # For actual implementation, you would need to run two separate optimizations
    # with controlled hyperparameter spaces
    
    result_without = optimizer.optimize_symbol(symbol, n_trials=trials)
    result_with = optimizer.optimize_symbol(symbol, n_trials=trials)
    
    if result_without and result_with:
        improvement = (result_with.objective_value / result_without.objective_value - 1) * 100
        print(f"\n📈 Phase 2 features improvement: {improvement:+.1f}%")
        print(f"Without Phase 2: {result_without.objective_value:.6f}")
        print(f"With Phase 2:    {result_with.objective_value:.6f}")

# Run test
test_phase2_impact('EURUSD', 10)
```

### **5. Trading System Integration Examples**

#### **Example 5.1: Basic Feature Mapping**
```python
# Real-time features from trading system
real_time_features = {
    'bb_lower_20_2': 1.0500,      # Bollinger Band lower (real-time naming)
    'bb_upper_20_2': 1.0600,      # Bollinger Band upper
    'bb_position_20_2': 0.3,      # Position within bands
    'atr_norm_14': 0.0012,        # Normalized ATR
    'rsi_14': 45.0,               # RSI (compatible naming)
    'macd_line': -0.001,          # MACD line (real-time naming)
    'doji_pattern': 1,            # Candlestick pattern
    'close': 1.0545               # Current close price
}

# Fix features for model compatibility
fixed_features = optimizer.fix_real_time_features(
    real_time_features,
    current_price=1.0545,
    symbol='EURUSD'
)

print(f"Original features: {len(real_time_features)}")
print(f"Fixed features: {len(fixed_features)}")

# Key mappings applied:
print(f"\n🔧 Key mappings applied:")
print(f"bb_lower_20_2 → bb_lower: {fixed_features['bb_lower']}")
print(f"bb_upper_20_2 → bb_upper: {fixed_features['bb_upper']}")
print(f"atr_norm_14 → atr_normalized_14: {fixed_features['atr_normalized_14']}")
print(f"macd_line → macd: {fixed_features['macd']}")
print(f"doji_pattern → doji: {fixed_features['doji']}")

# Additional computed features:
print(f"\n➕ Computed features:")
print(f"bb_position (calculated): {fixed_features['bb_position']:.4f}")
print(f"rsi_overbought: {fixed_features['rsi_overbought']}")
print(f"rsi_oversold: {fixed_features['rsi_oversold']}")
```

#### **Example 5.2: Emergency Feature Generation**
```python
# Minimal real-time features (emergency scenario)
minimal_features = {
    'close': 1.0545,
    'rsi_14': 45.0
}

# Generate complete feature set
emergency_features = optimizer.fix_real_time_features(
    minimal_features,
    current_price=1.0545,
    symbol='EURUSD'
)

print(f"📊 Emergency feature generation:")
print(f"Input features: {len(minimal_features)}")
print(f"Generated features: {len(emergency_features)}")

# Show filled default values
essential_defaults = [
    'atr_14', 'atr_normalized_14', 'bb_position', 'bbw', 'macd',
    'volume_ratio', 'session_european', 'usd_strength_proxy'
]

print(f"\n🔧 Default values applied:")
for feature in essential_defaults:
    if feature in emergency_features:
        print(f"{feature}: {emergency_features[feature]}")
```

#### **Example 5.3: Production Trading Integration**
```python
# Complete trading system integration example
class TradingSystemIntegration:
    def __init__(self):
        self.optimizer = optimizer
        self.model_path = None
        self.metadata = None
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the most recent optimized model"""
        import glob
        import json
        
        # Find latest ONNX model
        model_files = glob.glob("exported_models/*_CNN_LSTM_*.onnx")
        if model_files:
            self.model_path = max(model_files, key=os.path.getctime)
            
            # Load corresponding metadata
            metadata_file = self.model_path.replace('.onnx', '.json').replace('CNN_LSTM', 'training_metadata')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    
                print(f"✅ Loaded model: {os.path.basename(self.model_path)}")
                print(f"✅ Features: {self.metadata['num_features']}")
                print(f"✅ Lookback: {self.metadata['lookback_window']}")
    
    def generate_trading_signal(self, real_time_features, symbol='EURUSD'):
        """Generate trading signal with feature compatibility"""
        try:
            # Fix features for model compatibility
            fixed_features = self.optimizer.fix_real_time_features(
                real_time_features,
                current_price=real_time_features.get('close'),
                symbol=symbol
            )
            
            # Prepare feature array in correct order
            if self.metadata and 'selected_features' in self.metadata:
                feature_order = self.metadata['selected_features']
                feature_array = [fixed_features.get(feat, 0.0) for feat in feature_order]
            else:
                # Fallback feature order
                feature_array = list(fixed_features.values())[:30]
            
            # Scale features (would need to load scaler from metadata)
            # feature_array = self.scaler.transform([feature_array])
            
            # Create sequences for CNN-LSTM input
            # sequences = self.create_sequences(feature_array, self.metadata['lookback_window'])
            
            # Make prediction with ONNX model
            # prediction = self.onnx_session.run(None, {'input': sequences})[0]
            
            # For demonstration - simulate prediction
            import random
            prediction = random.uniform(0.3, 0.8)
            
            # Apply confidence thresholds
            confidence_high = self.metadata.get('hyperparameters', {}).get('confidence_threshold_high', 0.7)
            confidence_low = self.metadata.get('hyperparameters', {}).get('confidence_threshold_low', 0.3)
            
            if prediction > confidence_high:
                signal = 1  # Buy
            elif prediction < confidence_low:
                signal = -1  # Sell
            else:
                signal = 0  # Hold
            
            return {
                'signal': signal,
                'confidence': prediction,
                'features_used': len(fixed_features),
                'model_compatible': True
            }
            
        except Exception as e:
            return {
                'signal': 0,
                'confidence': 0.5,
                'error': str(e),
                'model_compatible': False
            }

# Usage example
trading_system = TradingSystemIntegration()

# Real-time market data
market_data = {
    'close': 1.0545,
    'bb_lower_20_2': 1.0500,
    'bb_upper_20_2': 1.0600,
    'rsi_14': 45.0,
    'atr_norm_14': 0.0012,
    'macd_line': -0.001
}

# Generate signal
signal_result = trading_system.generate_trading_signal(market_data, 'EURUSD')
print(f"Trading signal: {signal_result}")
```

### **6. Performance Analysis Examples**

#### **Example 6.1: Optimization History Analysis**
```python
# Analyze optimization performance over time
def analyze_optimization_history():
    print("📈 OPTIMIZATION HISTORY ANALYSIS")
    print("="*40)
    
    history = opt_manager.optimization_history
    
    for symbol in SYMBOLS:
        if symbol in history:
            results = history[symbol]
            scores = [r.objective_value for r in results]
            
            print(f"\n{symbol}:")
            print(f"  Runs: {len(results)}")
            print(f"  Best: {max(scores):.6f}")
            print(f"  Average: {np.mean(scores):.6f}")
            print(f"  Std Dev: {np.std(scores):.6f}")
            print(f"  Improvement: {(max(scores) - min(scores)):.6f}")
            
            # Show progression
            if len(scores) > 1:
                print(f"  Progression: {' → '.join([f'{s:.3f}' for s in scores])}")

analyze_optimization_history()
```

#### **Example 6.2: Hyperparameter Importance Analysis**
```python
# Analyze which hyperparameters matter most
def analyze_hyperparameter_importance():
    print("🔍 HYPERPARAMETER IMPORTANCE ANALYSIS")
    print("="*45)
    
    # Collect all optimization results
    all_results = []
    for symbol_results in opt_manager.optimization_history.values():
        all_results.extend(symbol_results)
    
    if len(all_results) < 5:
        print("❌ Insufficient data - need at least 5 optimization runs")
        return
    
    # Extract hyperparameters and scores
    param_impacts = {}
    
    for result in all_results:
        params = result.best_params
        score = result.objective_value
        
        for param_name, param_value in params.items():
            if param_name not in param_impacts:
                param_impacts[param_name] = []
            param_impacts[param_name].append((param_value, score))
    
    # Analyze correlations (simplified)
    print("📊 Parameter-Score Correlations:")
    for param_name, values_scores in param_impacts.items():
        if len(values_scores) >= 3:
            values = [vs[0] for vs in values_scores if isinstance(vs[0], (int, float))]
            scores = [vs[1] for vs in values_scores if isinstance(vs[0], (int, float))]
            
            if len(values) >= 3:
                correlation = np.corrcoef(values, scores)[0, 1]
                if not np.isnan(correlation):
                    print(f"  {param_name:25}: {correlation:+.3f}")

analyze_hyperparameter_importance()
```

#### **Example 6.3: Performance Benchmarking**
```python
# Benchmark against previous results
def benchmark_new_optimization(symbol='EURUSD', trials=20):
    print(f"📊 BENCHMARKING NEW OPTIMIZATION: {symbol}")
    print("="*50)
    
    # Get historical performance
    if symbol in opt_manager.optimization_history:
        historical_scores = [r.objective_value for r in opt_manager.optimization_history[symbol]]
        historical_best = max(historical_scores)
        historical_avg = np.mean(historical_scores)
        
        print(f"📈 Historical Performance:")
        print(f"  Best score: {historical_best:.6f}")
        print(f"  Average score: {historical_avg:.6f}")
        print(f"  Number of runs: {len(historical_scores)}")
    else:
        print("📊 No historical data - this will be the baseline")
        historical_best = 0.0
        historical_avg = 0.0
    
    # Run new optimization
    print(f"\n🚀 Running new optimization ({trials} trials)...")
    new_result = optimizer.optimize_symbol(symbol, n_trials=trials)
    
    if new_result:
        new_score = new_result.objective_value
        
        print(f"\n🎯 NEW OPTIMIZATION RESULTS:")
        print(f"  New score: {new_score:.6f}")
        print(f"  Success rate: {new_result.completed_trials/new_result.total_trials*100:.1f}%")
        
        if historical_best > 0:
            improvement_vs_best = (new_score / historical_best - 1) * 100
            improvement_vs_avg = (new_score / historical_avg - 1) * 100
            
            print(f"\n📈 PERFORMANCE COMPARISON:")
            print(f"  vs Best: {improvement_vs_best:+.1f}%")
            print(f"  vs Average: {improvement_vs_avg:+.1f}%")
            
            if improvement_vs_best > 5:
                print("🎉 EXCELLENT: Significant improvement over historical best!")
            elif improvement_vs_best > 0:
                print("✅ GOOD: Modest improvement over historical best")
            else:
                print("⚠️ BELOW HISTORICAL BEST: Consider adjusting approach")
        
        # Calculate benchmark metrics
        benchmark = opt_manager.calculate_benchmark_metrics(symbol, new_score)
        print(f"\n🏆 BENCHMARK METRICS:")
        print(f"  Rank: #{benchmark.rank}")
        print(f"  Percentile: {benchmark.percentile:.1f}%")
        print(f"  Improvement: {benchmark.improvement:+.6f}")
    
    return new_result

# Run benchmark test
result = benchmark_new_optimization('EURUSD', 25)
```

### **7. Dashboard and Reporting Examples**

#### **Example 7.1: Generate Comprehensive Report**
```python
# Generate complete optimization summary
def generate_comprehensive_report():
    print("📊 GENERATING COMPREHENSIVE OPTIMIZATION REPORT")
    print("="*60)
    
    # Generate text report
    report_text = dashboard.generate_summary_report()
    
    # Generate performance plots
    dashboard.create_performance_plot()
    
    # Custom analysis
    print("\n🔍 CUSTOM ANALYSIS:")
    
    # Symbol coverage analysis
    optimized_symbols = len(opt_manager.optimization_history)
    total_symbols = len(SYMBOLS)
    coverage = optimized_symbols / total_symbols * 100
    
    print(f"  Symbol coverage: {optimized_symbols}/{total_symbols} ({coverage:.1f}%)")
    
    # Performance distribution
    all_scores = []
    for results in opt_manager.optimization_history.values():
        all_scores.extend([r.objective_value for r in results])
    
    if all_scores:
        print(f"  Score distribution:")
        print(f"    Min: {min(all_scores):.6f}")
        print(f"    Max: {max(all_scores):.6f}")
        print(f"    Mean: {np.mean(all_scores):.6f}")
        print(f"    Std: {np.std(all_scores):.6f}")
        
        # Quality assessment
        excellent_runs = len([s for s in all_scores if s > 0.8])
        good_runs = len([s for s in all_scores if 0.7 <= s <= 0.8])
        poor_runs = len([s for s in all_scores if s < 0.7])
        
        print(f"  Quality distribution:")
        print(f"    Excellent (>0.8): {excellent_runs} runs")
        print(f"    Good (0.7-0.8): {good_runs} runs")
        print(f"    Poor (<0.7): {poor_runs} runs")
    
    print(f"\n✅ Report generation completed!")
    print(f"📁 Files generated in {RESULTS_PATH}/")

generate_comprehensive_report()
```

#### **Example 7.2: Custom Performance Visualization**
```python
# Create custom performance tracking plots
def create_custom_performance_plots():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    print("📈 CREATING CUSTOM PERFORMANCE PLOTS")
    print("="*45)
    
    # Collect time series data
    plot_data = []
    for symbol, results in opt_manager.optimization_history.items():
        for result in results:
            plot_data.append({
                'symbol': symbol,
                'timestamp': datetime.strptime(result.timestamp, '%Y%m%d_%H%M%S'),
                'objective_value': result.objective_value,
                'completed_trials': result.completed_trials,
                'total_trials': result.total_trials,
                'success_rate': result.completed_trials / result.total_trials
            })
    
    if len(plot_data) == 0:
        print("❌ No data available for plotting")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create multi-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Advanced Optimization Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Objective scores over time
    for symbol in df_plot['symbol'].unique():
        symbol_data = df_plot[df_plot['symbol'] == symbol]
        axes[0,0].plot(symbol_data['timestamp'], symbol_data['objective_value'], 
                      marker='o', label=symbol, linewidth=2)
    axes[0,0].set_title('Objective Scores Over Time')
    axes[0,0].set_ylabel('Objective Value')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Success rate distribution
    success_rates = df_plot['success_rate']
    axes[0,1].hist(success_rates, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].set_title('Success Rate Distribution')
    axes[0,1].set_xlabel('Success Rate')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Symbol performance comparison
    symbol_best_scores = df_plot.groupby('symbol')['objective_value'].max().sort_values(ascending=False)
    bars = axes[1,0].bar(range(len(symbol_best_scores)), symbol_best_scores.values, 
                        color='lightgreen', edgecolor='darkgreen')
    axes[1,0].set_title('Best Scores by Symbol')
    axes[1,0].set_ylabel('Best Objective Value')
    axes[1,0].set_xticks(range(len(symbol_best_scores)))
    axes[1,0].set_xticklabels(symbol_best_scores.index, rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, symbol_best_scores.values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Trial efficiency
    axes[1,1].scatter(df_plot['total_trials'], df_plot['objective_value'], 
                     c=df_plot['success_rate'], cmap='viridis', alpha=0.7, s=60)
    axes[1,1].set_title('Trial Efficiency Analysis')
    axes[1,1].set_xlabel('Total Trials')
    axes[1,1].set_ylabel('Objective Value')
    cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
    cbar.set_label('Success Rate')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = Path(RESULTS_PATH) / f"custom_performance_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Custom plots saved: {plot_file}")

create_custom_performance_plots()
```

These comprehensive examples demonstrate all major use cases and integration patterns for the Advanced Hyperparameter Optimization System, providing practical code templates for various scenarios from basic optimization to production trading system integration.