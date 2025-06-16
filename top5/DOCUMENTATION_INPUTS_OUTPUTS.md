# Advanced Hyperparameter Optimization System - Inputs & Outputs Reference

## 📥 Complete Inputs Reference

### **1. Data Input Files**

#### **Supported File Formats**
```python
# Primary format (recommended)
"data/metatrader_{SYMBOL}.parquet"    # Fast, compressed pandas format

# Alternative formats
"data/metatrader_{SYMBOL}.h5"         # HDF5 format
"data/metatrader_{SYMBOL}.csv"        # CSV format
"data/{SYMBOL}.parquet"               # Direct symbol naming
"data/{SYMBOL}.h5"                    # Direct symbol naming  
"data/{SYMBOL}.csv"                   # Direct symbol naming
```

#### **Required Data Schema**
```python
# Minimum required columns
{
    'close': float,        # REQUIRED - Closing price
    'high': float,         # REQUIRED - High price (will use close if missing)
    'low': float,          # REQUIRED - Low price (will use close if missing)
    'volume': float,       # OPTIONAL - Trading volume
    'tick_volume': float,  # ALTERNATIVE to volume
    'open': float,         # OPTIONAL - Opening price (will use close if missing)
}

# Index requirements
- DatetimeIndex with timezone-aware timestamps (preferred)
- String timestamps that can be parsed to datetime
- Minimum 100 records required
- Chronologically sorted data
```

#### **Example Data Formats**

**Parquet Format (Recommended):**
```python
import pandas as pd

# Load example
df = pd.read_parquet("data/metatrader_EURUSD.parquet")
print(df.head())

#                           close    high     low   volume
# timestamp                                              
# 2023-01-01 00:00:00+00:00  1.0701  1.0705  1.0698   1250
# 2023-01-01 01:00:00+00:00  1.0699  1.0703  1.0695   1180
# 2023-01-01 02:00:00+00:00  1.0702  1.0707  1.0697   1340
```

**CSV Format:**
```csv
timestamp,close,high,low,volume,open
2023-01-01 00:00:00,1.0701,1.0705,1.0698,1250,1.0700
2023-01-01 01:00:00,1.0699,1.0703,1.0695,1180,1.0701
2023-01-01 02:00:00,1.0702,1.0707,1.0697,1340,1.0699
```

### **2. Configuration Inputs**

#### **Global Configuration (ADVANCED_CONFIG)**
```python
ADVANCED_CONFIG = {
    'n_trials_per_symbol': 50,        # Default trials per optimization
    'cv_splits': 5,                   # Cross-validation splits (not currently used)
    'timeout_per_symbol': 1800,       # Timeout in seconds (30 minutes)
    'n_jobs': 1,                      # Parallel jobs (sequential for stability)
    'enable_pruning': True,           # Enable Optuna pruning
    'enable_warm_start': True,        # Enable warm start from historical results
    'enable_transfer_learning': True  # Enable parameter transfer between symbols
}
```

#### **Symbol List**
```python
SYMBOLS = [
    'EURUSD',  # Euro/US Dollar
    'GBPUSD',  # British Pound/US Dollar  
    'USDJPY',  # US Dollar/Japanese Yen
    'AUDUSD',  # Australian Dollar/US Dollar
    'USDCAD',  # US Dollar/Canadian Dollar
    'EURJPY',  # Euro/Japanese Yen
    'GBPJPY'   # British Pound/Japanese Yen
]
```

#### **Directory Paths**
```python
DATA_PATH = "data"                    # Input data directory
RESULTS_PATH = "optimization_results" # Output results directory
MODELS_PATH = "exported_models"       # Model export directory
```

### **3. Hyperparameter Search Spaces**

#### **Data Parameters**
```python
{
    'lookback_window': [20, 24, 28, 31, 35, 55, 59, 60],  # Historical context length
    'max_features': (25, 40),                              # Number of features to select
    'feature_selection_method': [                          # Feature selection strategy
        'rfe',                    # Recursive Feature Elimination
        'top_correlation',        # Highest correlation to target
        'variance_threshold',     # Highest variance features
        'mutual_info'            # Mutual information criterion
    ],
    'scaler_type': [                                       # Feature scaling method
        'robust',                # RobustScaler (median/IQR)
        'standard',              # StandardScaler (mean/std)  
        'minmax'                 # MinMaxScaler (0-1 range)
    ]
}
```

#### **Model Architecture Parameters**
```python
{
    'conv1d_filters_1': [24, 32, 40, 48],                # First conv layer filters
    'conv1d_filters_2': [40, 48, 56, 64],                # Second conv layer filters
    'conv1d_kernel_size': [2, 3],                        # Convolution kernel size
    'lstm_units': (85, 110, step=5),                     # LSTM memory units
    'lstm_return_sequences': [False, True],              # LSTM output format
    'dense_units': (30, 60, step=5),                     # Dense layer size
    'num_dense_layers': [1, 2],                          # Number of dense layers
    'batch_normalization': [True, False]                 # Use batch normalization
}
```

#### **Regularization Parameters**
```python
{
    'dropout_rate': (0.15, 0.28),                        # Dropout probability
    'l1_reg': (1e-6, 2e-5, log=True),                   # L1 regularization strength
    'l2_reg': (5e-5, 3e-4, log=True),                   # L2 regularization strength
}
```

#### **Training Parameters**
```python
{
    'optimizer': ['adam', 'rmsprop'],                     # Optimization algorithm
    'learning_rate': (0.002, 0.004),                     # Learning rate
    'batch_size': [64, 96, 128],                         # Training batch size
    'epochs': (80, 180),                                 # Maximum training epochs
    'patience': (5, 15),                                 # Early stopping patience
    'reduce_lr_patience': (3, 8)                         # Learning rate reduction patience
}
```

#### **Trading Parameters**
```python
{
    'confidence_threshold_high': (0.60, 0.80),           # High confidence threshold
    'confidence_threshold_low': (0.20, 0.40),            # Low confidence threshold
    'signal_smoothing': [True, False]                    # Apply signal smoothing
}
```

#### **Advanced Feature Parameters**
```python
{
    'use_rcs_features': [True, False],                   # Rate of Change Scaled features
    'use_cross_pair_features': [True, False]             # Cross-pair correlation features
}
```

### **4. Method Invocation Inputs**

#### **Basic Optimization**
```python
# Single symbol optimization
result = optimizer.optimize_symbol(
    symbol='EURUSD',              # Required: Currency pair symbol
    n_trials=50,                  # Optional: Number of optimization trials
    enable_warm_start=None        # Optional: Override global warm start setting
)
```

#### **Verbose Mode Control**
```python
# Enable detailed output
optimizer.set_verbose_mode(True)

# Disable detailed output (default)
optimizer.set_verbose_mode(False)
```

#### **Feature Mapping for Trading Systems**
```python
# Fix real-time features for trading compatibility
fixed_features = optimizer.fix_real_time_features(
    real_time_features={           # Required: Dict of real-time features
        'bb_lower_20_2': 1.0500,
        'bb_upper_20_2': 1.0600,
        'atr_norm_14': 0.0012,
        'rsi_14': 45.0
    },
    current_price=1.0545,          # Optional: Current price for calculations
    symbol='EURUSD'                # Optional: Symbol for symbol-specific features
)
```

#### **Dashboard and Reporting**
```python
# Generate summary report
report = dashboard.generate_summary_report()

# Create performance plots
dashboard.create_performance_plot()

# Pre-defined convenience functions
run_quick_test()              # 10 trials on EURUSD
run_multi_symbol_test()       # 15 trials each on 3 symbols
run_benchmark_report()        # Generate reports and plots
run_verbose_test()            # Demo verbose mode
run_warm_start_demo()         # Demo warm start control
```

---

## 📤 Complete Outputs Reference

### **1. Optimization Results Files**

#### **Best Parameters JSON**
```python
# File: optimization_results/best_params_{SYMBOL}_{TIMESTAMP}.json
{
    "symbol": "EURUSD",
    "timestamp": "20241216_143022",
    "objective_value": 0.823456,
    "best_params": {
        # Data parameters
        "lookback_window": 35,
        "max_features": 32,
        "feature_selection_method": "rfe",
        "scaler_type": "robust",
        
        # Architecture parameters  
        "conv1d_filters_1": 40,
        "conv1d_filters_2": 56,
        "conv1d_kernel_size": 3,
        "lstm_units": 95,
        "lstm_return_sequences": false,
        "dense_units": 45,
        "num_dense_layers": 1,
        "batch_normalization": true,
        
        # Regularization
        "dropout_rate": 0.234,
        "l1_reg": 8.45e-6,
        "l2_reg": 1.23e-4,
        
        # Training
        "optimizer": "adam",
        "learning_rate": 0.003123,
        "batch_size": 96,
        "epochs": 145,
        "patience": 8,
        "reduce_lr_patience": 5,
        
        # Trading
        "confidence_threshold_high": 0.742,
        "confidence_threshold_low": 0.287,
        "signal_smoothing": true,
        
        # Advanced features
        "use_rcs_features": true,
        "use_cross_pair_features": true
    },
    "mean_accuracy": 0.82,
    "mean_sharpe": 1.45,
    "std_accuracy": 0.04,
    "std_sharpe": 0.32,
    "num_features": 32,
    "total_trials": 50,
    "completed_trials": 47,
    "study_name": "advanced_cnn_lstm_EURUSD_20241216_143022",
    "trading_system_compatible": true,
    "all_fixes_applied": true
}
```

#### **Optimization Summary Report**
```markdown
# File: optimization_results/optimization_summary_{TIMESTAMP}.md

# Optimization Summary Report
Generated: 2024-12-16 14:30:22

## Overall Statistics
- Total symbols: 7
- Optimized symbols: 3
- Total optimization runs: 8
- Coverage: 42.9%

## Symbol Performance
1. **EURUSD**: 0.823456 (3 runs, latest: 20241216_143022)
2. **GBPUSD**: 0.798234 (2 runs, latest: 20241215_091234)
3. **USDJPY**: 0.776543 (3 runs, latest: 20241214_160945)

## Unoptimized Symbols
- AUDUSD: No optimization runs
- USDCAD: No optimization runs
- EURJPY: No optimization runs
- GBPJPY: No optimization runs

## Best Parameters Available
- **EURUSD**: 0.823456 (20241216_143022)
- **GBPUSD**: 0.798234 (20241215_091234)
- **USDJPY**: 0.776543 (20241214_160945)
```

### **2. Model Export Files**

#### **ONNX Model Files**
```python
# File: exported_models/{SYMBOL}_CNN_LSTM_{TIMESTAMP}.onnx
# Binary ONNX model file for production deployment
# Compatible with ONNX Runtime, TensorRT, and other inference engines
# Input shape: (batch_size, lookback_window, num_features)
# Output shape: (batch_size, 1) - sigmoid probability
```

#### **Training Metadata JSON**
```python
# File: exported_models/{SYMBOL}_training_metadata_{TIMESTAMP}.json
{
    "symbol": "EURUSD",
    "timestamp": "20241216_143022",
    "hyperparameters": {
        # Complete hyperparameter set from optimization
        "lookback_window": 35,
        "max_features": 32,
        # ... (full parameter set)
    },
    "selected_features": [
        # Actual features selected by feature selection method
        "close", "returns", "rsi_14", "macd", "bb_position",
        "atr_normalized_14", "volatility_20", "momentum_5",
        "eur_strength_proxy", "session_european", "rcs_momentum",
        # ... (up to max_features)
    ],
    "num_features": 32,
    "lookback_window": 35,
    "input_shape": [35, 32],
    "model_architecture": "CNN-LSTM",
    "framework": "tensorflow/keras",
    "export_format": "ONNX_ONLY",
    "scaler_type": "RobustScaler",
    "onnx_compatible": true,
    "trading_system_compatible": true,
    "feature_mapping": {
        # Complete feature mapping for trading system compatibility
        "bb_lower_20_2": "bb_lower",
        "bb_upper_20_2": "bb_upper",
        # ... (complete mapping)
    },
    "legacy_features_included": true,
    "phase_2_correlations_included": true,
    "session_logic_fixed": true,
    "threshold_validation_fixed": true,
    "gradient_clipping_enabled": true
}
```

### **3. Return Objects**

#### **OptimizationResult Dataclass**
```python
@dataclass
class OptimizationResult:
    symbol: str                    # Currency pair optimized
    timestamp: str                 # Optimization completion time
    objective_value: float         # Best objective score achieved
    best_params: Dict[str, Any]    # Best hyperparameter combination
    mean_accuracy: float           # Average validation accuracy
    mean_sharpe: float             # Average Sharpe ratio (estimated)
    std_accuracy: float            # Accuracy standard deviation
    std_sharpe: float              # Sharpe standard deviation
    num_features: int              # Number of features used
    total_trials: int              # Total trials attempted
    completed_trials: int          # Successfully completed trials
    study_name: str                # Optuna study identifier
```

#### **Feature Mapping Output**
```python
# Output from fix_real_time_features()
{
    'bb_lower': 1.0500,           # Mapped from bb_lower_20_2
    'bb_upper': 1.0600,           # Mapped from bb_upper_20_2
    'bb_position': 0.3,           # Calculated from current_price and bands
    'atr_normalized_14': 0.0012,  # Mapped from atr_norm_14
    'rsi_14': 45.0,               # Direct pass-through
    'rsi_overbought': 0,          # Calculated from rsi_14 < 70
    'rsi_oversold': 0,            # Calculated from rsi_14 > 30
    'macd': 0,                    # Default value
    'volume_ratio': 1.0,          # Default value
    'session_asian': 0,           # Default value
    'session_european': 1,        # Default value based on time
    'usd_strength_proxy': 0,      # Default value
    'risk_sentiment': 0,          # Default value
    # ... (complete feature set with 70+ features)
}
```

### **4. Console Output Formats**

#### **Quiet Mode Output**
```
🎯 Optimizing EURUSD (50 trials, warm start enabled)...
  Trial 1/50... 0.756234 ⭐
  Trial 5/50... 0.768345
  Trial 10/50... 0.782156 ⭐
  Trial 20/50... 0.798234 ⭐
  Trial 30/50... 0.803456
  Trial 40/50... 0.812345
  Trial 50/50... 0.823456 ⭐
✅ EURUSD: 0.823456 (47/50 trials)
📁 Saved: EURUSD_CNN_LSTM_20241216_143022.onnx
```

#### **Verbose Mode Output**
```
========================================================
🎯 HYPERPARAMETER OPTIMIZATION: EURUSD
========================================================
Target trials: 50
Features: ALL legacy + Phase 2 correlations + Trading compatibility
Warm start: enabled

Trial   1/50: LR=0.003123 | Dropout=0.234 | LSTM=95 | Window=35
   🔧 RCS features ENABLED by hyperparameter
   🔧 Cross-pair features ENABLED for EURUSD
   ✅ Created 68 features (conditional features based on hyperparameters)
   🔧 Feature selection: rfe (selecting 32/68 features)
   ✅ Selected 32 features using rfe
   🔧 Using robust scaler
   🔧 Signal smoothing ENABLED
   ✅ Accuracy: 0.8234, Signal Quality: 0.3456, Score: 0.823456
 → 0.823456 ⭐ NEW BEST!

Trial   2/50: LR=0.002987 | Dropout=0.198 | LSTM=100 | Window=28
   ❌ RCS features DISABLED by hyperparameter
   ❌ Cross-pair features DISABLED by hyperparameter
   ✅ Created 52 features (conditional features based on hyperparameters)
   🔧 Feature selection: variance_threshold (selecting 28/52 features)
   ✅ Selected 28 features using variance_threshold
   🔧 Using standard scaler
   ❌ Signal smoothing DISABLED
   ✅ Accuracy: 0.7987, Signal Quality: 0.2876, Score: 0.796834
 → 0.796834
```

### **5. Error Outputs**

#### **Data Loading Errors**
```python
# No data available
❌ No data available for EURJPY

# Insufficient data
❌ Insufficient data: only 45 records (minimum 100 required)

# Invalid data format
❌ Data loading error: Invalid column 'close' - contains non-numeric values
```

#### **Optimization Errors**
```python
# Training failures
❌ EURUSD: Failed (Model training error: Insufficient memory)

# Feature errors
⚠️ Feature creation failed: Cross-pair correlation calculation error
❌ Feature selection failed (RFE error), using variance fallback
```

#### **Export Errors**
```python
# ONNX export failures
❌ ONNX export failed: tf2onnx not available
❌ ONNX export failed: Model contains unsupported operations
```

### **6. File Locations Summary**

```
top5/
├── data/                                             # INPUT DATA
│   ├── metatrader_EURUSD.parquet                    # Required format
│   ├── metatrader_GBPUSD.parquet
│   ├── metatrader_USDJPY.parquet
│   ├── metatrader_AUDUSD.parquet
│   ├── metatrader_USDCAD.parquet
│   ├── metatrader_EURJPY.parquet
│   └── metatrader_GBPJPY.parquet
│
├── optimization_results/                             # OPTIMIZATION OUTPUTS
│   ├── best_params_EURUSD_20241216_143022.json     # Best parameters
│   ├── best_params_GBPUSD_20241215_091234.json
│   ├── optimization_summary_20241216_143500.md      # Summary reports
│   └── optimization_performance_20241216_143500.png # Performance plots
│
├── exported_models/                                  # MODEL OUTPUTS
│   ├── EURUSD_CNN_LSTM_20241216_143022.onnx        # ONNX models
│   ├── EURUSD_training_metadata_20241216_143022.json # Model metadata
│   ├── GBPUSD_CNN_LSTM_20241215_091234.onnx
│   └── GBPUSD_training_metadata_20241215_091234.json
│
└── Advanced_Hyperparameter_Optimization_Clean.ipynb # MAIN SYSTEM
```

This comprehensive reference covers all possible inputs and outputs for the Advanced Hyperparameter Optimization System.