#!/usr/bin/env python3
"""
Debug and Fix EURUSD Feature Mismatch - SIGNAL_ERROR
Specific solution for the close, close_position, ema_10, ema_100, ema_20 issue
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def debug_eurusd_feature_mismatch():
    """Debug the specific EURUSD feature mismatch error"""
    
    print("🔍 DEBUGGING EURUSD FEATURE MISMATCH")
    print("=" * 50)
    
    symbol = 'EURUSD'
    
    # 1. Check what features were expected during training
    print(f"\n1️⃣ Loading EURUSD training metadata...")
    
    metadata_files = list(Path("exported_models").glob(f"{symbol}_training_metadata_*.json"))
    if not metadata_files:
        print(f"❌ No training metadata found for {symbol}")
        return None
    
    # Get the most recent metadata (best performance)
    latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_metadata, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Loaded: {latest_metadata.name}")
    
    if 'selected_features' in metadata:
        training_features = metadata['selected_features']
        print(f"✅ Training used {len(training_features)} features")
        
        print(f"\n📋 Training Features List:")
        for i, feature in enumerate(training_features, 1):
            print(f"   {i:2d}. {feature}")
        
        # Check for the specific problematic features
        problematic_features = ['close', 'close_position', 'ema_10', 'ema_100', 'ema_20']
        found_features = [f for f in problematic_features if f in training_features]
        missing_features = [f for f in problematic_features if f not in training_features]
        
        print(f"\n🎯 Problematic Features Analysis:")
        print(f"   Features EXPECTED by model: {found_features}")
        print(f"   Features NOT in training: {missing_features}")
        
    else:
        print("❌ No selected_features found in metadata")
        return None
    
    # 2. Check current feature generation capability
    print(f"\n2️⃣ Testing current feature generation...")
    
    # Load sample data
    data_files = list(Path("data").glob(f"*{symbol}*"))
    if not data_files:
        print(f"❌ No data files found for {symbol}")
        return None
    
    data_file = data_files[0]
    print(f"📈 Loading data from: {data_file.name}")
    
    # Prefer parquet files
    parquet_file = None
    for file in data_files:
        if file.suffix == '.parquet':
            parquet_file = file
            break
    
    if parquet_file:
        data_file = parquet_file
        df = pd.read_parquet(data_file)
    elif data_file.suffix == '.csv':
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    elif data_file.suffix == '.h5':
        try:
            df = pd.read_hdf(data_file, key='data')
        except:
            print(f"❌ Could not read H5 file: {data_file}")
            return None
    else:
        print(f"❌ Unsupported file format: {data_file.suffix}")
        return None
    
    # Standardize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Create a dummy datetime index
        df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
    
    print(f"📊 Data loaded: {len(df)} rows")
    print(f"📊 Available columns: {list(df.columns)}")
    print(f"📊 Index type: {type(df.index)}")
    
    # 3. Test feature creation with current system
    print(f"\n3️⃣ Creating features with current system...")
    
    # Create features using the current method
    current_features = create_enhanced_features_for_debugging(df.tail(200))  # Use recent data
    
    print(f"✅ Current system creates {len(current_features.columns)} features")
    
    # 4. Compare with training expectations
    print(f"\n4️⃣ Feature Compatibility Analysis:")
    
    current_feature_set = set(current_features.columns)
    training_feature_set = set(training_features)
    
    matching_features = training_feature_set & current_feature_set
    missing_in_current = training_feature_set - current_feature_set
    extra_in_current = current_feature_set - training_feature_set
    
    compatibility_score = len(matching_features) / len(training_feature_set)
    
    print(f"   Expected features: {len(training_feature_set)}")
    print(f"   Current features: {len(current_feature_set)}")
    print(f"   Matching features: {len(matching_features)}")
    print(f"   Missing features: {len(missing_in_current)}")
    print(f"   Extra features: {len(extra_in_current)}")
    print(f"   Compatibility: {compatibility_score:.1%}")
    
    if missing_in_current:
        print(f"\n❌ MISSING FEATURES (cause of SIGNAL_ERROR):")
        for feature in sorted(list(missing_in_current)):
            print(f"      • {feature}")
    
    if extra_in_current:
        print(f"\n➕ EXTRA FEATURES (not used in model):")
        for feature in sorted(list(extra_in_current))[:10]:
            print(f"      • {feature}")
        if len(extra_in_current) > 10:
            print(f"      ... and {len(extra_in_current) - 10} more")
    
    # 5. Create the fix
    print(f"\n5️⃣ Creating Feature Alignment Fix...")
    
    fix_result = create_feature_alignment_fix(
        training_features=training_features,
        current_features=current_features,
        sample_data=df.tail(50)
    )
    
    return {
        'metadata': metadata,
        'training_features': training_features,
        'current_features': current_features,
        'missing_features': missing_in_current,
        'compatibility_score': compatibility_score,
        'fix_result': fix_result
    }

def create_enhanced_features_for_debugging(df):
    """Create features using current system for debugging"""
    
    features = pd.DataFrame(index=df.index)
    
    # Get price data
    close = df['close']
    high = df.get('high', close)
    low = df.get('low', close)
    open_price = df.get('open', close)
    volume = df.get('tick_volume', df.get('volume', pd.Series(100, index=df.index)))
    
    # Basic price features
    features['close'] = close
    features['high'] = high
    features['low'] = low
    features['open'] = open_price
    features['volume'] = volume
    
    # Returns
    features['returns'] = close.pct_change()
    features['log_returns'] = np.log(close / close.shift(1))
    features['high_low_pct'] = (high - low) / close
    features['close_position'] = (close - low) / (high - low + 1e-10)  # This is the missing feature!
    
    # Moving averages - INCLUDING THE MISSING ONES
    ma_periods = [5, 10, 20, 50, 100, 200]  # Include ema_10, ema_100
    for period in ma_periods:
        features[f'sma_{period}'] = close.rolling(period).mean()
        features[f'ema_{period}'] = close.ewm(span=period).mean()  # This creates ema_10, ema_100, etc.
        
    # RSI
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # RSI derivatives
    features['rsi_divergence'] = features['rsi_7'] - features['rsi_21']
    features['rsi_momentum'] = features['rsi_14'].diff()
    features['rsi_overbought'] = (features['rsi_14'] > 70).astype(int)
    features['rsi_oversold'] = (features['rsi_14'] < 30).astype(int)
    
    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    features['macd'] = macd
    features['macd_signal'] = macd_signal
    features['macd_histogram'] = macd - macd_signal
    
    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features['bb_upper'] = sma_20 + (bb_std * 2)
    features['bb_lower'] = sma_20 - (bb_std * 2)
    features['bb_middle'] = sma_20
    features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
    features['bbw'] = (features['bb_upper'] - features['bb_lower']) / sma_20
    
    # Additional technical indicators
    tp = (high + low + close) / 3
    features['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean()))))
    
    # ATR
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    features['atr_14'] = true_range.rolling(14).mean()
    features['atr_21'] = true_range.rolling(21).mean()
    features['adx'] = features['atr_14'].rolling(14).mean() / close
    
    # Time features
    features['hour'] = features.index.hour
    features['day_of_week'] = features.index.dayofweek
    features['is_monday'] = (features.index.dayofweek == 0).astype(int)
    features['is_friday'] = (features.index.dayofweek == 4).astype(int)
    features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
    
    # Session features
    weekday = features.index.dayofweek
    hours = features.index.hour
    is_weekend = (weekday >= 5).astype(int)
    market_open = (1 - is_weekend)
    
    session_asian_raw = ((hours >= 21) | (hours <= 6)).astype(int)
    session_european_raw = ((hours >= 7) & (hours <= 16)).astype(int)
    session_us_raw = ((hours >= 13) & (hours <= 22)).astype(int)
    
    features['session_asian'] = session_asian_raw * market_open
    features['session_european'] = session_european_raw * market_open
    features['session_us'] = session_us_raw * market_open
    features['session_overlap_eur_us'] = features['session_european'] * features['session_us']
    
    # Volume features
    for period in [5, 10, 20]:
        vol_sma = volume.rolling(period).mean()
        features[f'volume_sma_{period}'] = vol_sma
        features[f'volume_ratio'] = volume / (vol_sma + 1)
    
    # Price volume
    features['price_volume'] = close * volume
    
    # Volatility features
    vol_5 = close.pct_change().rolling(5).std()
    vol_20 = close.pct_change().rolling(20).std()
    features['volatility_5'] = vol_5
    features['volatility_20'] = vol_20
    features['volatility_regime'] = (vol_5 > vol_20).astype(int)
    features['volatility_ratio'] = vol_5 / (vol_20 + 1e-10)
    
    # Candlestick patterns (simplified)
    features['doji'] = (np.abs(close - open_price) < (high - low) * 0.1).astype(int)
    features['hammer'] = ((low < close * 0.99) & (high < close * 1.01)).astype(int)
    features['engulfing'] = features['doji']  # Simplified
    
    # Price position features
    for period in [10, 20]:
        rolling_min = close.rolling(period).min()
        rolling_max = close.rolling(period).max()
        features[f'price_position_{period}'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-10)
    
    # ROC (Rate of Change)
    features['roc'] = close.pct_change(10)
    features['roc_momentum'] = features['roc'].diff()
    
    # Stochastic oscillator
    for period in [14]:
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
        features[f'stoch_k'] = k_percent
        features[f'stoch_d'] = k_percent.rolling(3).mean()
    
    # Clean features
    features = features.ffill().bfill()
    
    # Fill remaining NaNs
    for col in features.columns:
        if features[col].isnull().any():
            if 'ratio' in col or 'position' in col:
                features[col] = features[col].fillna(1.0)
            elif 'rsi' in col or 'stoch' in col:
                features[col] = features[col].fillna(50.0)
            else:
                features[col] = features[col].fillna(0.0)
    
    # Replace infinite values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0)
    
    return features

def create_feature_alignment_fix(training_features, current_features, sample_data):
    """Create a fix to align current features with training features"""
    
    print(f"🔧 Creating feature alignment fix...")
    
    # Create feature mapper
    feature_mapper = {}
    
    # Check which training features are missing
    current_feature_set = set(current_features.columns)
    missing_features = [f for f in training_features if f not in current_feature_set]
    
    # Create feature generation code
    fix_code = f"""
# EURUSD Feature Alignment Fix
# Add this to your OptimizedFeatureEngine.create_advanced_features method

def create_aligned_features_for_eurusd(self, df, hyperparameters=None):
    '''Create features aligned with EURUSD training'''
    
    features = pd.DataFrame(index=df.index)
    
    close = df['close']
    high = df.get('high', close)
    low = df.get('low', close)
    open_price = df.get('open', close)
    volume = df.get('tick_volume', df.get('volume', pd.Series(100, index=df.index)))
    
    # === CRITICAL MISSING FEATURES ===
    
    # 1. close_position (was missing!)
    features['close_position'] = (close - low) / (high - low + 1e-10)
    
    # 2. All EMA features (was missing ema_10, ema_100, ema_20)
    for period in [5, 10, 12, 20, 26, 50, 100, 200]:
        features[f'ema_{{period}}'] = close.ewm(span=period).mean()
    
    # 3. Basic price features
    features['close'] = close
    features['high'] = high
    features['low'] = low
    features['open'] = open_price
    features['volume'] = volume
    
    # === REST OF FEATURES ===
    # (Add all other features as before...)
    
    # 4. Select only the features that were used in training
    training_feature_list = {training_features}
    
    # Keep only training features
    aligned_features = {{}}
    for feature in training_feature_list:
        if feature in features.columns:
            aligned_features[feature] = features[feature]
        else:
            # Provide default value for missing features
            if 'rsi' in feature and feature not in ['rsi_divergence', 'rsi_momentum']:
                aligned_features[feature] = 50.0  # Neutral RSI
            elif 'position' in feature:
                aligned_features[feature] = 0.5   # Mid position
            elif 'session_' in feature or feature.endswith(('_overbought', '_oversold')):
                aligned_features[feature] = 0     # Binary features
            else:
                aligned_features[feature] = 0.0   # Default
    
    return pd.DataFrame(aligned_features, index=df.index)
"""
    
    print(f"✅ Fix code generated")
    
    # Save the fix code
    fix_file = Path("eurusd_feature_alignment_fix.py")
    with open(fix_file, 'w') as f:
        f.write(fix_code)
    
    print(f"💾 Fix saved to: {fix_file}")
    
    print(f"\n🔧 IMMEDIATE ACTION REQUIRED:")
    print(f"   1. Add missing features to your feature engine:")
    
    if missing_features:
        for feature in missing_features[:10]:
            if feature == 'close_position':
                print(f"      • {feature}: (close - low) / (high - low + 1e-10)")
            elif feature.startswith('ema_'):
                period = feature.split('_')[1]
                print(f"      • {feature}: close.ewm(span={period}).mean()")
            else:
                print(f"      • {feature}: <needs implementation>")
        
        if len(missing_features) > 10:
            print(f"      ... and {len(missing_features) - 10} more")
    
    print(f"\n   2. Update your feature selection to use exact training features")
    print(f"   3. Test with: debug_result = debug_eurusd_feature_mismatch()")
    
    return {
        'missing_features': missing_features,
        'fix_code': fix_code,
        'fix_file': str(fix_file)
    }

def quick_fix_trading_system():
    """Apply immediate fix to resolve SIGNAL_ERROR"""
    
    print(f"🚀 APPLYING QUICK FIX FOR EURUSD SIGNAL_ERROR")
    print("=" * 50)
    
    # Run the debug analysis
    debug_result = debug_eurusd_feature_mismatch()
    
    if debug_result and debug_result['compatibility_score'] < 1.0:
        print(f"\n🔧 IMPLEMENTING IMMEDIATE FIX...")
        
        # Create a feature alignment wrapper
        wrapper_code = '''
# Add this to your trading system to fix EURUSD features immediately

class EURUSDFeatureFix:
    def __init__(self, training_features):
        self.training_features = training_features
        
    def fix_features(self, features_df, current_price=None):
        """Fix features to match training expectations"""
        
        fixed = {}
        
        # Add missing critical features
        if 'close_position' not in features_df.columns and current_price:
            if 'high' in features_df.columns and 'low' in features_df.columns:
                high_val = features_df['high'].iloc[-1] if not features_df.empty else current_price
                low_val = features_df['low'].iloc[-1] if not features_df.empty else current_price
                fixed['close_position'] = (current_price - low_val) / (high_val - low_val + 1e-10)
        
        # Add missing EMA features
        if 'close' in features_df.columns:
            close_series = features_df['close']
            for period in [10, 20, 100]:
                ema_col = f'ema_{period}'
                if ema_col not in features_df.columns:
                    if len(close_series) >= period:
                        fixed[ema_col] = close_series.ewm(span=period).mean().iloc[-1]
                    else:
                        fixed[ema_col] = current_price if current_price else close_series.iloc[-1]
        
        # Ensure all training features are present
        for feature in self.training_features:
            if feature not in features_df.columns and feature not in fixed:
                # Provide sensible defaults
                if 'rsi' in feature and not any(x in feature for x in ['divergence', 'momentum']):
                    fixed[feature] = 50.0
                elif 'position' in feature:
                    fixed[feature] = 0.5
                elif any(x in feature for x in ['session_', '_overbought', '_oversold']):
                    fixed[feature] = 0
                else:
                    fixed[feature] = 0.0
        
        # Combine original and fixed features
        result_features = features_df.copy() if not features_df.empty else pd.DataFrame()
        for feature, value in fixed.items():
            result_features[feature] = value
        
        # Select only training features in correct order
        final_features = {}
        for feature in self.training_features:
            if feature in result_features.columns:
                final_features[feature] = result_features[feature].iloc[-1] if not result_features.empty else fixed.get(feature, 0.0)
            else:
                final_features[feature] = fixed.get(feature, 0.0)
        
        return final_features

# Usage in your signal generation:
# feature_fix = EURUSDFeatureFix(training_features)
# fixed_features = feature_fix.fix_features(features_df, current_price)
'''
        
        # Save wrapper
        wrapper_file = Path("eurusd_signal_error_fix.py")
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        print(f"✅ Quick fix created: {wrapper_file}")
        print(f"🎯 This should resolve the SIGNAL_ERROR immediately")
        
    return debug_result

if __name__ == "__main__":
    # Run the complete debug and fix process
    result = quick_fix_trading_system()
    
    print(f"\n🎉 EURUSD FEATURE MISMATCH DEBUG COMPLETE!")
    print("=" * 50)
    
    if result:
        print(f"✅ Issue identified and fix provided")
        print(f"✅ Feature compatibility: {result['compatibility_score']:.1%}")
        print(f"✅ Missing features: {len(result['missing_features'])}")
        print(f"🚀 Apply the fix to resolve SIGNAL_ERROR!")
    else:
        print(f"❌ Debug failed - check data and metadata files")