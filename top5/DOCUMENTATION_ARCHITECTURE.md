# Advanced Hyperparameter Optimization System - Architecture Documentation

## 🏗️ Technical Architecture & Methodology

### **System Design Philosophy**

The system follows a **modular, extensible architecture** designed for production forex trading environments:

1. **Separation of Concerns**: Data loading, feature engineering, optimization, and model export are isolated
2. **Dependency Injection**: Components accept managers/configurations to enable testing and flexibility  
3. **State Management**: Optimization history and warm start parameters preserved across sessions
4. **Error Resilience**: Comprehensive exception handling with graceful degradation
5. **Production Integration**: ONNX export and feature mapping for real-time trading systems

---

## 🧩 Core Component Architecture

### **1. AdvancedOptimizationManager**

**Purpose**: Central coordinator for optimization workflow and state management

**Key Responsibilities:**
- Load and manage optimization history from JSON files
- Provide warm start parameters from best previous results  
- Calculate benchmark metrics and performance comparisons
- Coordinate study resumption and parameter transfer

**Critical Methods:**
```python
def load_existing_results(self):
    """Scans optimization_results/ for best_params_*.json files"""
    # Loads historical optimization data
    # Builds benchmark database for performance comparison
    # Enables warm start with proven parameter combinations

def get_warm_start_params(self, symbol: str) -> Optional[Dict]:
    """Returns best known parameters for symbol or EURUSD fallback"""
    # Symbol-specific parameters preferred
    # EURUSD parameters used as universal fallback
    # None returned if no historical data exists

def calculate_benchmark_metrics(self, symbol: str, current_score: float):
    """Compares current optimization against historical performance"""
    # Calculates improvement over previous best
    # Determines percentile ranking among all runs
    # Provides context for optimization quality assessment
```

**Data Storage Format:**
```python
# optimization_history structure
{
    'EURUSD': [
        OptimizationResult(
            symbol='EURUSD',
            timestamp='20241216_143022', 
            objective_value=0.8234,
            best_params={'lstm_units': 95, 'learning_rate': 0.003},
            # ... additional metrics
        )
    ]
}
```

### **2. StudyManager**

**Purpose**: Optuna study lifecycle management with intelligent warm starting

**Key Features:**
- **Smart Warm Start**: Enqueues best parameters + variations for faster convergence
- **Study Configuration**: Manages TPE sampler and median pruner settings
- **Parameter Variation**: Creates slight modifications of proven parameters

**Critical Methods:**
```python
def create_study(self, symbol: str, enable_warm_start: Optional[bool] = None):
    """Creates new Optuna study with optional warm start"""
    # Configures TPE sampler (n_startup_trials=10)
    # Sets up MedianPruner (n_startup_trials=5, n_warmup_steps=10)
    # Applies warm start if enabled globally or explicitly

def add_warm_start_trials(self, study: optuna.Study, symbol: str):
    """Enqueues proven parameters to accelerate optimization"""
    # Enqueues exact best parameters from history
    # Creates 2-3 variations with ±10-15% parameter adjustments
    # Focuses initial trials on high-probability success regions

def create_parameter_variation(self, base_params: dict, variation_factor: float):
    """Generates slight variations of successful parameters"""
    # Integer params: ±20% variation (lstm_units, filters, etc.)
    # Float params: ±variation_factor adjustment (learning_rate, dropout)
    # Categorical params: Unchanged to maintain architecture integrity
```

**Warm Start Strategy:**
1. **Exact Replication**: Best known parameters tried first
2. **Conservative Variation**: ±10% adjustments to numerical parameters  
3. **Aggressive Variation**: ±15% adjustments for broader exploration
4. **Fallback**: Random exploration if no historical data available

### **3. FixedAdvancedHyperparameterOptimizer**

**Purpose**: Main optimization engine with ALL hyperparameter fixes implemented

**Architecture Layers:**

#### **Layer 1: Hyperparameter Suggestion**
```python
def suggest_advanced_hyperparameters(self, trial: optuna.Trial, symbol: str):
    """Comprehensive hyperparameter space with validation"""
    # DATA PARAMETERS: lookback_window, max_features, feature_selection_method, scaler_type
    # ARCHITECTURE: conv1d_filters, lstm_units, dense_units, layers
    # REGULARIZATION: dropout_rate, l1_reg, l2_reg, batch_normalization  
    # TRAINING: optimizer, learning_rate, batch_size, epochs, patience
    # TRADING: confidence_thresholds with proper separation validation
    # ADVANCED: use_rcs_features, use_cross_pair_features, signal_smoothing
```

**Threshold Validation Logic:**
```python
# FIXED: Proper threshold validation with safety margin
confidence_high = params['confidence_threshold_high']
confidence_low = params['confidence_threshold_low'] 
min_separation = 0.15

if confidence_low >= confidence_high - min_separation:
    confidence_low = max(0.1, confidence_high - min_separation)
    params['confidence_threshold_low'] = confidence_low
```

#### **Layer 2: Feature Engineering (Fixed Implementation)**
```python
def _create_advanced_features(self, df: pd.DataFrame, symbol: str, params: dict):
    """FIXED: Features now controlled by hyperparameters"""
    
    # Get hyperparameter controls - THIS IS THE CRITICAL FIX
    use_cross_pair = params.get('use_cross_pair_features', True) if params else True
    use_rcs = params.get('use_rcs_features', True) if params else True
    
    # ALWAYS INCLUDED: Core technical indicators
    # - ATR-based volatility (trading compatible names)
    # - Multi-timeframe RSI (7, 14, 21 periods)
    # - Bollinger Bands (bb_upper, bb_lower, bb_middle, bbw, bb_position)
    # - MACD (macd, macd_signal, macd_histogram)
    # - Moving averages (SMA 5,10,20,50 + price ratios)
    # - Session features (Asian, European, US sessions with weekend filtering)
    # - Legacy indicators (CCI, ADX, Stochastic, candlestick patterns)
    
    # CONDITIONALLY INCLUDED: Advanced features based on hyperparameters
    if use_rcs:
        # Rate of Change Scaled features
        # - rcs_5, rcs_10 (volatility-normalized momentum)
        # - rcs_momentum, rcs_acceleration, rcs_divergence
        
    if use_cross_pair:
        # Phase 2 correlation features  
        # - USD/EUR strength proxies
        # - JPY safe-haven detection
        # - Risk sentiment analysis
        # - Correlation momentum tracking
```

**Feature Categorization:**
- **Core Features (Always)**: 45-60 features including all essential technical indicators
- **RCS Features (Optional)**: +5 volatility-normalized momentum features
- **Cross-Pair Features (Optional)**: +6 correlation and sentiment features
- **Total Range**: 45-71 features depending on hyperparameter settings

#### **Layer 3: Feature Selection (Fixed Implementation)**
```python
def _apply_feature_selection(self, X: pd.DataFrame, y: pd.Series, params: dict):
    """FIXED: Actually implements all feature selection methods"""
    
    method = params.get('feature_selection_method', 'variance_threshold')
    max_features = params.get('max_features', 30)
    
    if method == 'variance_threshold':
        # Original: Select features with highest variance
        feature_vars = X.var()
        selected_features = feature_vars.nlargest(max_features).index
        
    elif method == 'top_correlation':
        # NEW: Select features with highest correlation to target
        correlations = {col: abs(X[col].corr(y)) for col in X.columns}
        selected_features = pd.Series(correlations).nlargest(max_features).index
        
    elif method == 'mutual_info':
        # NEW: Select features using mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        selector.fit_transform(X.fillna(0), y)
        selected_features = X.columns[selector.get_support()]
        
    elif method == 'rfe':
        # NEW: Recursive feature elimination with RandomForest
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFE(estimator, n_features_to_select=max_features)
        selector.fit_transform(X.fillna(0), y)
        selected_features = X.columns[selector.support_]
```

#### **Layer 4: Model Training (Fixed Implementation)**
```python
def _train_and_evaluate_model(self, symbol: str, params: dict, price_data: pd.DataFrame):
    """FIXED: All hyperparameters now actually implemented"""
    
    # FIXED: Create features with hyperparameter controls
    features = self._create_advanced_features(price_data, symbol=symbol, params=params)
    
    # FIXED: Apply selected feature selection method
    X_selected = self._apply_feature_selection(X, y, params)
    
    # FIXED: Apply selected scaler type
    scaler_type = params.get('scaler_type', 'robust')
    scaler = {
        'robust': RobustScaler(),
        'standard': StandardScaler(), 
        'minmax': MinMaxScaler()
    }[scaler_type]
    
    # FIXED: Apply signal smoothing to predictions
    val_pred = model.predict(X_val, verbose=0).flatten()
    val_pred_smoothed = self._apply_signal_smoothing(val_pred, params)
    
    # FIXED: Use confidence thresholds in evaluation
    confidence_high = params.get('confidence_threshold_high', 0.7)
    confidence_low = params.get('confidence_threshold_low', 0.3)
    
    signals = np.where(val_pred_smoothed > confidence_high, 1,
                      np.where(val_pred_smoothed < confidence_low, -1, 0))
    
    # Calculate objective with signal quality bonus
    signal_quality = np.mean(np.abs(signals))  # Reward decisive signals
    score = accuracy * 0.8 + signal_quality * 0.2
```

#### **Layer 5: Signal Processing (Fixed Implementation)**
```python
def _apply_signal_smoothing(self, predictions: np.ndarray, params: dict):
    """FIXED: Actually implement signal smoothing hyperparameter"""
    
    use_smoothing = params.get('signal_smoothing', False)
    
    if use_smoothing and len(predictions) > 3:
        # Apply 3-point moving average smoothing
        smoothed = np.copy(predictions)
        for i in range(2, len(predictions)):
            smoothed[i] = np.mean(predictions[max(0, i-2):i+1])
        return smoothed
    else:
        return predictions
```

### **4. Trading System Integration**

**Purpose**: Bridge training and production environments with feature compatibility

**Key Components:**

#### **Feature Mapping System**
```python
def _create_trading_feature_mapping(self):
    """Maps real-time feature names to training-compatible names"""
    return {
        # Bollinger Band mappings (real-time -> training)
        'bb_lower_20_2': 'bb_lower',           # Standard BB lower band
        'bb_upper_20_2': 'bb_upper',           # Standard BB upper band  
        'bb_middle_20_2': 'bb_middle',         # Standard BB middle line
        'bb_position_20_2': 'bb_position',     # Price position in BB
        'bb_width_20_2': 'bbw',                # Bollinger Band Width
        
        # ATR mappings
        'atr_norm_14': 'atr_normalized_14',    # Normalized ATR 14-period
        'atr_norm_21': 'atr_normalized_21',    # Normalized ATR 21-period
        
        # MACD mappings  
        'macd_line': 'macd',                   # MACD main line
        'macd_signal_line': 'macd_signal',     # MACD signal line
        
        # Candlestick patterns
        'doji_pattern': 'doji',                # Doji candlestick
        'hammer_pattern': 'hammer',            # Hammer candlestick
        'engulfing_pattern': 'engulfing',      # Engulfing pattern
        
        # RSI variations
        'rsi_14_overbought': 'rsi_overbought', # RSI > 70 flag
        'rsi_14_oversold': 'rsi_oversold',     # RSI < 30 flag
    }
```

#### **Emergency Feature Generation**
```python
def fix_real_time_features(self, real_time_features, current_price=None, symbol=None):
    """Complete feature compatibility solution"""
    
    # Step 1: Apply direct mappings
    for rt_feature, value in real_time_features.items():
        if rt_feature in self.feature_mapping:
            mapped_name = self.feature_mapping[rt_feature]
            fixed_features[mapped_name] = value
            
    # Step 2: Add missing features with safe defaults
    for feature_name, default_value in self.trading_defaults.items():
        if feature_name not in fixed_features:
            fixed_features[feature_name] = default_value
            
    # Step 3: Compute derived features if possible
    if current_price and 'bb_upper' in fixed_features and 'bb_lower' in fixed_features:
        bb_range = fixed_features['bb_upper'] - fixed_features['bb_lower']
        if bb_range > 0:
            fixed_features['bb_position'] = (current_price - fixed_features['bb_lower']) / bb_range
            fixed_features['bb_position'] = max(0, min(1, fixed_features['bb_position']))
```

---

## 🔧 External Library Integration

### **Optuna (Hyperparameter Optimization)**
- **TPESampler**: Tree-structured Parzen Estimator for intelligent parameter suggestion
- **MedianPruner**: Early stopping for unpromising trials to save computation
- **Study Management**: Persistent optimization with resumption capabilities

**Configuration:**
```python
sampler = TPESampler(seed=42, n_startup_trials=10)  # 10 random trials before TPE
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Prune after 10 steps
```

### **TensorFlow/Keras (Deep Learning)**
- **Sequential Model**: Linear stack of CNN and LSTM layers
- **Conv1D**: 1D convolution for feature extraction from time series
- **LSTM**: Long Short-Term Memory for temporal pattern recognition
- **Regularization**: Dropout, L1/L2 regularization, BatchNormalization
- **Callbacks**: EarlyStopping, ReduceLROnPlateau for training optimization

**Model Architecture:**
```python
# Conv1D layers for feature extraction
Conv1D(filters=32, kernel_size=3, activation='relu') -> BatchNormalization -> Dropout
Conv1D(filters=48, kernel_size=3, activation='relu') -> BatchNormalization -> Dropout

# LSTM layer for temporal modeling  
LSTM(units=95, implementation=1, unroll=False) -> Dropout

# Dense layers for classification
Dense(units=45, activation='relu') -> Dropout -> Dense(1, activation='sigmoid')
```

### **Scikit-learn (Feature Processing)**
- **Scalers**: RobustScaler, StandardScaler, MinMaxScaler for feature normalization
- **Feature Selection**: SelectKBest, RFE, VarianceThreshold for dimensionality reduction
- **RandomForestClassifier**: Used in RFE for feature importance ranking
- **Metrics**: Various classification metrics for model evaluation

### **tf2onnx (Model Export)**
- **ONNX Conversion**: Converts TensorFlow models to ONNX format for production deployment
- **Opset 13**: Uses modern ONNX operators for broad compatibility
- **Optimization**: Model graph optimization for inference performance

**Export Process:**
```python
@tf.function
def model_func(x):
    return model(x)

input_signature = [tf.TensorSpec((None, lookback_window, num_features), tf.float32)]
onnx_model, _ = tf2onnx.convert.from_function(model_func, input_signature, opset=13)
```

### **Pandas/NumPy (Data Processing)**
- **DataFrame Operations**: Time series data manipulation and feature engineering
- **Rolling Windows**: Moving averages, correlations, and statistical calculations
- **Index Management**: DatetimeIndex handling for time-based operations
- **Mathematical Operations**: Vectorized calculations for technical indicators

---

## 📊 Methodology Details

### **Cross-Validation Strategy**
- **Time Series Split**: Chronological splitting to prevent data leakage
- **80/20 Split**: 80% training, 20% validation within each optimization trial
- **Forward Walk**: Validation always uses future data relative to training

### **Objective Function Design**
```python
# Multi-component objective balancing accuracy and signal quality
score = accuracy * 0.8 + signal_quality * 0.2

# Where:
# accuracy = model classification accuracy on validation set
# signal_quality = percentage of decisive signals (non-neutral predictions)
```

### **Early Stopping Strategy**
- **Validation Loss Monitoring**: Stop when validation loss stops improving
- **Patience**: Allow 5-15 epochs without improvement before stopping
- **Best Weight Restoration**: Revert to weights from best validation epoch

### **Gradient Clipping**
```python
# Prevent exploding gradients in LSTM training
optimizer = Adam(learning_rate=0.003, clipvalue=1.0)
```

### **Session Logic Implementation**
```python
# FIXED: Proper weekend handling for forex markets
session_asian_raw = ((hours >= 21) | (hours <= 6)).astype(int)
session_european_raw = ((hours >= 7) & (hours <= 16)).astype(int) 
session_us_raw = ((hours >= 13) & (hours <= 22)).astype(int)

# Weekend filtering (Saturday=5, Sunday=6)
is_weekend = (weekday >= 5).astype(int)
market_open = (1 - is_weekend)

features['session_asian'] = session_asian_raw * market_open
features['session_european'] = session_european_raw * market_open
features['session_us'] = session_us_raw * market_open
```

---

## 🔄 Data Flow Architecture

### **Input Processing Pipeline**
1. **Data Loading**: Multi-format support (.parquet, .h5, .csv)
2. **Validation**: Column name standardization, index conversion
3. **Cleaning**: NaN removal, positive price validation
4. **Feature Engineering**: Technical indicator calculation with hyperparameter controls
5. **Feature Selection**: Method-based dimensionality reduction
6. **Scaling**: Configurable normalization strategies
7. **Sequence Creation**: Sliding window approach for CNN-LSTM input

### **Training Pipeline**
1. **Data Splitting**: Chronological train/validation split
2. **Model Creation**: Architecture configuration from hyperparameters
3. **Training Loop**: Batch-based training with callbacks
4. **Evaluation**: Validation metrics calculation
5. **Signal Processing**: Optional smoothing and threshold application
6. **Objective Calculation**: Multi-component score computation

### **Output Generation Pipeline**
1. **Model Export**: ONNX format with metadata
2. **Parameter Storage**: JSON format with complete configuration
3. **Result Logging**: Optimization history preservation
4. **Report Generation**: Performance summaries and visualizations

This architecture ensures robust, scalable, and production-ready hyperparameter optimization for forex trading models.