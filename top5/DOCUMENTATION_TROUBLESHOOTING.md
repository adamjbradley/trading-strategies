# Advanced Hyperparameter Optimization System - Troubleshooting Guide

## 🔧 Common Issues & Solutions

### **1. Data Loading Issues**

#### **Issue 1.1: No Data Files Found**
```
❌ No data available for EURUSD
```

**Causes:**
- Missing data files in the `data/` directory
- Incorrect file naming convention
- Unsupported file format

**Solutions:**
```python
# Check data directory structure
import os
from pathlib import Path

data_path = Path("data")
print(f"Data directory exists: {data_path.exists()}")
print(f"Data directory contents: {list(data_path.glob('*'))}")

# Check for expected file patterns
symbol = "EURUSD"
patterns = [
    f"metatrader_{symbol}.parquet",
    f"metatrader_{symbol}.h5", 
    f"metatrader_{symbol}.csv",
    f"{symbol}.parquet",
    f"{symbol}.h5",
    f"{symbol}.csv"
]

for pattern in patterns:
    file_path = data_path / pattern
    print(f"{pattern}: {'EXISTS' if file_path.exists() else 'MISSING'}")
```

**Fix:**
1. Ensure files are named correctly: `metatrader_EURUSD.parquet`
2. Use supported formats: `.parquet` (preferred), `.h5`, `.csv`
3. Place files in the `data/` directory relative to notebook

#### **Issue 1.2: Insufficient Data**
```
❌ Insufficient data: only 45 records (minimum 100 required)
```

**Causes:**
- Data file too small for meaningful optimization
- Data filtering removing too many records

**Solutions:**
```python
# Check data size and quality
import pandas as pd

data_file = "data/metatrader_EURUSD.parquet"
df = pd.read_parquet(data_file)

print(f"Total records: {len(df)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Columns: {df.columns.tolist()}")

# Check for required columns
required_cols = ['close', 'high', 'low']
for col in required_cols:
    if col in df.columns:
        valid_count = df[col].notna().sum()
        print(f"{col}: {valid_count}/{len(df)} valid values")
    else:
        print(f"❌ Missing required column: {col}")

# Check for invalid prices
if 'close' in df.columns:
    positive_prices = (df['close'] > 0).sum()
    print(f"Valid positive prices: {positive_prices}/{len(df)}")
```

**Fix:**
1. Use datasets with at least 1000 records (preferably 5000+)
2. Ensure data quality: no NaN values in critical columns
3. Use hourly or daily data, not tick data

#### **Issue 1.3: Data Format Problems**
```
❌ Data loading error: Invalid column 'close' - contains non-numeric values
```

**Causes:**
- CSV files with string values in numeric columns
- Incorrect data types after loading
- Missing or malformed timestamps

**Solutions:**
```python
# Diagnose data format issues
import pandas as pd
import numpy as np

def diagnose_data_format(file_path):
    print(f"🔍 DIAGNOSING DATA FORMAT: {file_path}")
    print("="*50)
    
    try:
        # Try different loading methods
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            df = pd.read_hdf(file_path, key='data')
        
        print(f"✅ File loaded successfully")
        print(f"Shape: {df.shape}")
        print(f"Index type: {type(df.index)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check data types
        print(f"\n📊 DATA TYPES:")
        print(df.dtypes)
        
        # Check for non-numeric values in numeric columns
        numeric_cols = ['close', 'high', 'low', 'volume', 'open']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col])
                    print(f"✅ {col}: All numeric")
                except:
                    non_numeric = df[~pd.to_numeric(df[col], errors='coerce').notna()]
                    print(f"❌ {col}: {len(non_numeric)} non-numeric values")
                    if len(non_numeric) > 0:
                        print(f"   Examples: {non_numeric[col].head().tolist()}")
        
        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"⚠️ Index is not DatetimeIndex: {type(df.index)}")
            if df.index.dtype == 'object':
                try:
                    pd.to_datetime(df.index)
                    print(f"✅ Index can be converted to datetime")
                except:
                    print(f"❌ Index cannot be converted to datetime")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return None

# Run diagnosis
df = diagnose_data_format("data/metatrader_EURUSD.parquet")
```

**Fix:**
1. Ensure numeric columns contain only numeric values
2. Convert index to DatetimeIndex: `df.index = pd.to_datetime(df.index)`
3. Clean data before saving: `df = df.dropna().select_dtypes(include=[np.number])`

### **2. Optimization Failures**

#### **Issue 2.1: All Trials Failing**
```
❌ EURUSD: Failed (0/50 trials)
```

**Causes:**
- TensorFlow/CUDA memory issues
- Feature engineering errors
- Model architecture incompatibility

**Solutions:**
```python
# Memory diagnostics
import tensorflow as tf
import psutil
import os

def diagnose_optimization_failure():
    print("🔍 OPTIMIZATION FAILURE DIAGNOSIS")
    print("="*45)
    
    # Check system memory
    memory = psutil.virtual_memory()
    print(f"System Memory:")
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Used: {memory.percent:.1f}%")
    
    # Check TensorFlow
    print(f"\nTensorFlow:")
    print(f"  Version: {tf.__version__}")
    print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Check GPU memory if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    
    # Test basic optimization components
    print(f"\n🧪 COMPONENT TESTING:")
    
    # Test data loading
    try:
        test_data = optimizer._load_symbol_data('EURUSD')
        if test_data is not None:
            print(f"✅ Data loading: SUCCESS ({len(test_data)} records)")
        else:
            print(f"❌ Data loading: FAILED")
            return
    except Exception as e:
        print(f"❌ Data loading: ERROR - {e}")
        return
    
    # Test feature creation
    try:
        test_features = optimizer._create_advanced_features(test_data, symbol='EURUSD')
        print(f"✅ Feature creation: SUCCESS ({len(test_features.columns)} features)")
    except Exception as e:
        print(f"❌ Feature creation: ERROR - {e}")
        return
    
    # Test model creation
    try:
        test_params = {
            'lstm_units': 50, 'conv1d_filters_1': 32, 'conv1d_filters_2': 48,
            'dropout_rate': 0.2, 'learning_rate': 0.001, 'l1_reg': 1e-5, 'l2_reg': 1e-4
        }
        test_model = optimizer._create_onnx_compatible_model((30, 25), test_params)
        print(f"✅ Model creation: SUCCESS")
        del test_model  # Free memory
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"❌ Model creation: ERROR - {e}")
        return
    
    print(f"\n✅ All components working - issue may be parameter-specific")

diagnose_optimization_failure()
```

**Fix:**
1. **Memory Issues**: Reduce batch size, enable memory growth for GPU
2. **CUDA Issues**: Set `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` to force CPU
3. **Feature Issues**: Check verbose mode to see where failures occur

#### **Issue 2.2: Low Success Rate**
```
✅ EURUSD: 0.742356 (12/50 trials)
```

**Causes:**
- Aggressive hyperparameter ranges
- Data quality issues
- Model architecture mismatches

**Solutions:**
```python
# Analyze trial failure patterns
def analyze_trial_failures():
    print("📊 TRIAL FAILURE PATTERN ANALYSIS")
    print("="*45)
    
    # Enable verbose mode for detailed diagnosis
    optimizer.set_verbose_mode(True)
    
    # Run small test with monitoring
    print("Running diagnostic optimization (5 trials)...")
    result = optimizer.optimize_symbol('EURUSD', n_trials=5)
    
    optimizer.set_verbose_mode(False)
    
    if result:
        success_rate = result.completed_trials / result.total_trials
        print(f"\nSuccess rate: {success_rate*100:.1f}%")
        
        if success_rate < 0.8:
            print("⚠️ Low success rate detected")
            print("Common causes:")
            print("1. Memory constraints - try reducing batch_size range")
            print("2. Learning rate too high - try narrower LR range")
            print("3. Model too complex - try reducing LSTM units")
            
            # Suggest parameter adjustments
            print("\n🔧 Suggested fixes:")
            print("- Reduce batch_size range: [32, 64] instead of [64, 128]")
            print("- Narrow learning_rate: [0.001, 0.003] instead of [0.002, 0.004]")
            print("- Reduce lstm_units: [50, 80] instead of [85, 110]")

analyze_trial_failures()
```

**Fix:**
1. **Conservative Parameters**: Start with narrower hyperparameter ranges
2. **Memory Management**: Use smaller batch sizes and models
3. **Gradual Scaling**: Test with small trials first, then scale up

#### **Issue 2.3: ONNX Export Failures**
```
❌ ONNX export failed: tf2onnx not available
```

**Causes:**
- Missing tf2onnx library
- TensorFlow version incompatibility
- Model contains unsupported operations

**Solutions:**
```python
# ONNX export diagnostics
def diagnose_onnx_export():
    print("🔍 ONNX EXPORT DIAGNOSIS")
    print("="*35)
    
    # Check tf2onnx availability
    try:
        import tf2onnx
        print(f"✅ tf2onnx available: {tf2onnx.__version__}")
    except ImportError:
        print("❌ tf2onnx not available")
        print("Fix: pip install tf2onnx")
        return
    
    # Check ONNX availability
    try:
        import onnx
        print(f"✅ onnx available: {onnx.__version__}")
    except ImportError:
        print("❌ onnx not available")
        print("Fix: pip install onnx")
        return
    
    # Check TensorFlow compatibility
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Test basic ONNX conversion
    try:
        # Create simple test model
        test_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        @tf.function
        def model_func(x):
            return test_model(x)
        
        input_signature = [tf.TensorSpec((None, 5), tf.float32)]
        onnx_model, _ = tf2onnx.convert.from_function(model_func, input_signature, opset=13)
        
        print("✅ Basic ONNX conversion: SUCCESS")
        
    except Exception as e:
        print(f"❌ Basic ONNX conversion: ERROR - {e}")

diagnose_onnx_export()
```

**Fix:**
1. **Install Dependencies**: `pip install tf2onnx onnx`
2. **TensorFlow Compatibility**: Use TensorFlow 2.8+ with tf2onnx 1.13+
3. **Model Architecture**: Avoid custom layers, use standard Keras layers only

### **3. Feature Engineering Issues**

#### **Issue 3.1: Feature Mismatch in Trading System**
```
Feature names should match those that were passed during fit.
Feature names unseen at fit time: ['bb_lower_20_2', 'atr_norm_14']
```

**Causes:**
- Real-time feature names differ from training names
- Missing feature mapping
- Incomplete feature fix implementation

**Solutions:**
```python
# Test feature mapping system
def test_feature_mapping():
    print("🔧 TESTING FEATURE MAPPING SYSTEM")
    print("="*45)
    
    # Test common real-time features
    real_time_features = {
        'bb_lower_20_2': 1.0500,
        'bb_upper_20_2': 1.0600,
        'bb_position_20_2': 0.3,
        'atr_norm_14': 0.0012,
        'atr_norm_21': 0.0015,
        'rsi_14': 45.0,
        'macd_line': -0.001,
        'macd_signal_line': 0.0005,
        'doji_pattern': 1,
        'hammer_pattern': 0,
        'close': 1.0545
    }
    
    print(f"Input features: {len(real_time_features)}")
    print(f"Input names: {list(real_time_features.keys())}")
    
    # Apply feature fix
    fixed_features = optimizer.fix_real_time_features(
        real_time_features,
        current_price=1.0545,
        symbol='EURUSD'
    )
    
    print(f"\nOutput features: {len(fixed_features)}")
    
    # Check specific mappings
    mapping_tests = [
        ('bb_lower_20_2', 'bb_lower'),
        ('bb_upper_20_2', 'bb_upper'),
        ('bb_position_20_2', 'bb_position'),
        ('atr_norm_14', 'atr_normalized_14'),
        ('atr_norm_21', 'atr_normalized_21'),
        ('macd_line', 'macd'),
        ('macd_signal_line', 'macd_signal'),
        ('doji_pattern', 'doji'),
        ('hammer_pattern', 'hammer')
    ]
    
    print(f"\n🔧 MAPPING VERIFICATION:")
    for rt_name, expected_name in mapping_tests:
        if expected_name in fixed_features:
            original_value = real_time_features.get(rt_name, 'N/A')
            mapped_value = fixed_features[expected_name]
            print(f"✅ {rt_name} → {expected_name}: {original_value} → {mapped_value}")
        else:
            print(f"❌ {rt_name} → {expected_name}: MISSING")
    
    # Check for essential trading features
    essential_features = [
        'bb_position', 'atr_14', 'atr_normalized_14', 'rsi_14', 
        'macd', 'volume_ratio', 'session_european'
    ]
    
    print(f"\n✅ ESSENTIAL FEATURES CHECK:")
    missing_essential = []
    for feature in essential_features:
        if feature in fixed_features:
            print(f"✅ {feature}: {fixed_features[feature]}")
        else:
            print(f"❌ {feature}: MISSING")
            missing_essential.append(feature)
    
    if missing_essential:
        print(f"\n⚠️ Missing essential features: {missing_essential}")
        print("This will cause trading system errors")
    else:
        print(f"\n🎉 All essential features present - trading system ready!")

test_feature_mapping()
```

**Fix:**
1. **Use Feature Mapping**: Always call `optimizer.fix_real_time_features()` before prediction
2. **Update Mapping**: Add new real-time feature names to the mapping dictionary
3. **Test Integration**: Verify mapping with your actual real-time feature names

#### **Issue 3.2: NaN/Infinite Values in Features**
```
⚠️ Feature creation error: infinite values detected
```

**Causes:**
- Division by zero in technical indicators
- Missing price data causing calculation errors
- Extreme price movements

**Solutions:**
```python
# Feature quality diagnostics
def diagnose_feature_quality():
    print("🔍 FEATURE QUALITY DIAGNOSIS")
    print("="*40)
    
    # Load test data
    test_data = optimizer._load_symbol_data('EURUSD')
    if test_data is None:
        print("❌ No test data available")
        return
    
    # Create features
    features = optimizer._create_advanced_features(test_data, symbol='EURUSD')
    
    print(f"Feature matrix shape: {features.shape}")
    
    # Check for data quality issues
    nan_counts = features.isna().sum()
    inf_counts = np.isinf(features.select_dtypes(include=[np.number])).sum()
    
    print(f"\n📊 DATA QUALITY SUMMARY:")
    print(f"Total NaN values: {nan_counts.sum()}")
    print(f"Total infinite values: {inf_counts.sum()}")
    
    # Show problematic features
    if nan_counts.sum() > 0:
        print(f"\n❌ FEATURES WITH NaN VALUES:")
        problematic_nan = nan_counts[nan_counts > 0]
        for feature, count in problematic_nan.items():
            percentage = count / len(features) * 100
            print(f"  {feature}: {count} ({percentage:.1f}%)")
    
    if inf_counts.sum() > 0:
        print(f"\n❌ FEATURES WITH INFINITE VALUES:")
        problematic_inf = inf_counts[inf_counts > 0]
        for feature, count in problematic_inf.items():
            percentage = count / len(features) * 100
            print(f"  {feature}: {count} ({percentage:.1f}%)")
    
    # Check feature ranges
    print(f"\n📈 FEATURE RANGE ANALYSIS:")
    numeric_features = features.select_dtypes(include=[np.number])
    for col in numeric_features.columns[:10]:  # First 10 features
        col_data = numeric_features[col]
        print(f"  {col}: [{col_data.min():.6f}, {col_data.max():.6f}] "
              f"(mean: {col_data.mean():.6f})")
    
    # Data loss after cleaning
    clean_features = features.replace([np.inf, -np.inf], np.nan).dropna()
    data_loss = (len(features) - len(clean_features)) / len(features) * 100
    
    print(f"\n🧹 AFTER CLEANING:")
    print(f"Original rows: {len(features)}")
    print(f"Clean rows: {len(clean_features)}")
    print(f"Data loss: {data_loss:.1f}%")
    
    if data_loss > 10:
        print("⚠️ High data loss - investigate feature calculation errors")
    else:
        print("✅ Acceptable data loss")

diagnose_feature_quality()
```

**Fix:**
1. **Add Safety Checks**: Use `+ 1e-10` in denominators to prevent division by zero
2. **Clip Extreme Values**: Use `features.clip(lower=q01, upper=q99)` for outlier handling
3. **Robust Calculations**: Use rolling windows with `min_periods` parameter

### **4. Performance Issues**

#### **Issue 4.1: Slow Optimization**
```
Trial 10/50... (5 minutes per trial)
```

**Causes:**
- Large datasets
- Complex feature engineering
- Inefficient hyperparameter ranges

**Solutions:**
```python
# Performance optimization
def optimize_performance():
    print("⚡ PERFORMANCE OPTIMIZATION GUIDE")
    print("="*45)
    
    # Data size recommendations
    print("📊 DATA SIZE OPTIMIZATION:")
    print("- Recommended data size: 2000-5000 records")
    print("- Maximum efficient size: 10000 records")
    print("- Use hourly data instead of minute data")
    print("- Consider data sampling for initial testing")
    
    # Feature optimization
    print("\n🔧 FEATURE OPTIMIZATION:")
    print("- Start with max_features=20-25")
    print("- Disable RCS features initially: use_rcs_features=False")
    print("- Use variance_threshold feature selection for speed")
    print("- Consider reducing Phase 2 features for testing")
    
    # Model optimization
    print("\n🏗️ MODEL OPTIMIZATION:")
    print("- Start with smaller LSTM units: 50-70")
    print("- Use smaller batch sizes: 32-64")
    print("- Reduce epochs: 50-100 for testing")
    print("- Use aggressive early stopping: patience=5")
    
    # Hyperparameter optimization
    print("\n🎯 HYPERPARAMETER OPTIMIZATION:")
    print("- Start with 10-20 trials for testing")
    print("- Use narrow ranges for critical parameters")
    print("- Enable pruning for faster convergence")
    print("- Use warm start for subsequent runs")
    
    # System optimization
    print("\n💻 SYSTEM OPTIMIZATION:")
    print("- Close unnecessary applications")
    print("- Use SSD storage for data files")
    print("- Enable GPU if available (but may use more memory)")
    print("- Monitor system memory usage")

optimize_performance()
```

**Fix:**
1. **Reduce Data Size**: Use 2000-5000 records for initial testing
2. **Simplify Features**: Start with essential features only
3. **Smaller Models**: Use LSTM units 50-70, batch size 32-64
4. **Aggressive Pruning**: Enable early stopping with patience=5

#### **Issue 4.2: Memory Errors**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Causes:**
- GPU memory exhaustion
- Large batch sizes
- Memory leaks in TensorFlow

**Solutions:**
```python
# Memory management
def setup_memory_management():
    print("💾 MEMORY MANAGEMENT SETUP")
    print("="*40)
    
    import tensorflow as tf
    import os
    
    # Disable GPU if causing memory issues
    print("🔧 GPU Memory Configuration:")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except:
            # If memory growth fails, disable GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("⚠️ GPU disabled due to memory constraints")
    else:
        print("ℹ️ No GPU detected - using CPU")
    
    # Set memory limit if needed
    try:
        # Limit GPU memory to 2GB if available
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
            print("✅ GPU memory limited to 2GB")
    except:
        print("⚠️ Could not set GPU memory limit")
    
    # Memory monitoring
    import psutil
    memory = psutil.virtual_memory()
    print(f"\n📊 System Memory Status:")
    print(f"Total: {memory.total / (1024**3):.1f} GB")
    print(f"Available: {memory.available / (1024**3):.1f} GB")
    print(f"Usage: {memory.percent:.1f}%")
    
    if memory.percent > 80:
        print("⚠️ High memory usage - consider:")
        print("- Closing other applications")
        print("- Using smaller batch sizes")
        print("- Reducing data size")
        print("- Forcing CPU usage")

setup_memory_management()
```

**Fix:**
1. **Force CPU**: Set `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`
2. **Reduce Batch Size**: Use 16-32 instead of 64-128
3. **Clear Memory**: Call `tf.keras.backend.clear_session()` between trials
4. **Monitor Usage**: Check system memory before starting optimization

### **5. Integration Issues**

#### **Issue 5.1: Import Errors**
```
ModuleNotFoundError: No module named 'optuna'
```

**Causes:**
- Missing required libraries
- Virtual environment issues
- Version incompatibilities

**Solutions:**
```python
# Check and install dependencies
def check_dependencies():
    print("📦 DEPENDENCY CHECK")
    print("="*25)
    
    required_packages = {
        'optuna': '3.0+',
        'tensorflow': '2.8+',
        'scikit-learn': '1.0+',
        'pandas': '1.3+',
        'numpy': '1.19+',
        'tf2onnx': '1.13+',
        'onnx': '1.12+',
        'matplotlib': '3.3+',
        'seaborn': '0.11+'
    }
    
    missing_packages = []
    
    for package, version in required_packages.items():
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                print(f"✅ {package}: {module.__version__}")
            else:
                print(f"✅ {package}: installed")
        except ImportError:
            print(f"❌ {package}: MISSING (required: {version})")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print(f"\n🎉 All dependencies satisfied!")

check_dependencies()
```

**Fix:**
1. **Install Packages**: `pip install optuna tensorflow scikit-learn pandas numpy tf2onnx onnx`
2. **Update Packages**: `pip install --upgrade package_name`
3. **Virtual Environment**: Ensure correct environment is activated

#### **Issue 5.2: File Path Issues**
```
FileNotFoundError: data/metatrader_EURUSD.parquet not found
```

**Causes:**
- Incorrect working directory
- Relative path issues
- Missing data directory

**Solutions:**
```python
# File path diagnostics
def diagnose_file_paths():
    print("📁 FILE PATH DIAGNOSIS")
    print("="*30)
    
    import os
    from pathlib import Path
    
    # Check current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Check for expected directories
    expected_dirs = ['data', 'optimization_results', 'exported_models']
    for dir_name in expected_dirs:
        dir_path = cwd / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/: EXISTS")
            # List contents
            contents = list(dir_path.glob('*'))
            print(f"   Contents: {len(contents)} files")
            for item in contents[:5]:  # Show first 5 items
                print(f"     - {item.name}")
            if len(contents) > 5:
                print(f"     ... and {len(contents)-5} more")
        else:
            print(f"❌ {dir_name}/: MISSING")
    
    # Check for specific data files
    print(f"\n📊 DATA FILE CHECK:")
    data_dir = cwd / 'data'
    if data_dir.exists():
        for symbol in SYMBOLS:
            file_path = data_dir / f"metatrader_{symbol}.parquet"
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024*1024)  # MB
                print(f"✅ {symbol}: {file_size:.1f} MB")
            else:
                print(f"❌ {symbol}: MISSING")
    
    # Suggest fixes
    print(f"\n🔧 RECOMMENDED FIXES:")
    if not (cwd / 'data').exists():
        print("1. Create data directory: mkdir data")
    print("2. Ensure notebook is in the correct directory")
    print("3. Use absolute paths if relative paths fail")
    print("4. Check file naming convention: metatrader_SYMBOL.parquet")

diagnose_file_paths()
```

**Fix:**
1. **Create Directories**: `mkdir data optimization_results exported_models`
2. **Check Working Directory**: Ensure notebook is in the project root
3. **Use Absolute Paths**: Modify paths if working directory is incorrect

### **6. Quick Diagnostic Commands**

```python
# Quick system health check
def quick_health_check():
    print("🏥 QUICK SYSTEM HEALTH CHECK")
    print("="*40)
    
    # 1. Data availability
    try:
        test_data = optimizer._load_symbol_data('EURUSD')
        print(f"✅ Data loading: {len(test_data) if test_data else 0} records")
    except:
        print(f"❌ Data loading: FAILED")
    
    # 2. Feature creation
    try:
        if test_data is not None:
            features = optimizer._create_advanced_features(test_data, symbol='EURUSD')
            print(f"✅ Feature creation: {len(features.columns)} features")
        else:
            print(f"❌ Feature creation: No data")
    except Exception as e:
        print(f"❌ Feature creation: ERROR - {str(e)[:50]}")
    
    # 3. Memory status
    import psutil
    memory = psutil.virtual_memory()
    print(f"💾 Memory: {memory.available / (1024**3):.1f}GB available ({100-memory.percent:.1f}% free)")
    
    # 4. Dependencies
    critical_deps = ['optuna', 'tensorflow', 'sklearn', 'pandas']
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}: OK")
        except:
            print(f"❌ {dep}: MISSING")
    
    print(f"\n🎯 System ready for optimization!")

# Run quick check
quick_health_check()
```

This troubleshooting guide covers the most common issues and provides practical solutions for the Advanced Hyperparameter Optimization System.