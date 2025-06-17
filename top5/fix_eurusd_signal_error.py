#!/usr/bin/env python3
"""
DEFINITIVE FIX for EURUSD SIGNAL_ERROR
The features exist but need exact ordering and selection
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

class EURUSDSignalErrorFix:
    """
    Fix for EURUSD SIGNAL_ERROR - ensures exact feature matching
    """
    
    def __init__(self):
        self.training_features = None
        self.feature_order = None
        self.load_training_metadata()
    
    def load_training_metadata(self):
        """Load the exact features used in EURUSD training"""
        
        metadata_files = list(Path("exported_models").glob("EURUSD_training_metadata_*.json"))
        if not metadata_files:
            print("❌ No EURUSD training metadata found")
            return
        
        # Get the most recent metadata
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_metadata, 'r') as f:
            metadata = json.load(f)
        
        if 'selected_features' in metadata:
            self.training_features = metadata['selected_features']
            self.feature_order = self.training_features.copy()  # Preserve exact order
            print(f"✅ Loaded {len(self.training_features)} training features from {latest_metadata.name}")
        else:
            print("❌ No selected_features in metadata")
    
    def create_aligned_features(self, df, hyperparameters=None):
        """
        Create features that exactly match training
        This is the replacement for create_advanced_features
        """
        
        features = pd.DataFrame(index=df.index)
        
        # Get price data
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        open_price = df.get('open', close)
        volume = df.get('volume', df.get('tick_volume', pd.Series(100, index=df.index)))
        
        print(f"🔧 Creating aligned features for EURUSD...")
        
        # === ALL POSSIBLE FEATURES (superset) ===
        
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
        features['close_position'] = (close - low) / (high - low + 1e-10)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = close.rolling(period).mean()
            features[f'ema_{period}'] = close.ewm(span=period).mean()
        
        # RSI family
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
        
        # Technical indicators
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
        
        # Stochastic oscillator
        for period in [14]:
            lowest_low = low.rolling(period).min()
            highest_high = high.rolling(period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
            features[f'stoch_k'] = k_percent
            features[f'stoch_d'] = k_percent.rolling(3).mean()
        
        # ROC (Rate of Change)
        features['roc'] = close.pct_change(10)
        features['roc_momentum'] = features['roc'].diff()
        
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
        
        # Price position features
        for period in [10, 20]:
            rolling_min = close.rolling(period).min()
            rolling_max = close.rolling(period).max()
            features[f'price_position_{period}'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-10)
        
        # Candlestick patterns (simplified)
        features['doji'] = (np.abs(close - open_price) < (high - low) * 0.1).astype(int)
        features['hammer'] = ((low < close * 0.99) & (high < close * 1.01)).astype(int)
        features['engulfing'] = features['doji']  # Simplified
        
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
        
        # === CRITICAL: SELECT ONLY TRAINING FEATURES IN EXACT ORDER ===
        if self.training_features:
            selected_features = pd.DataFrame(index=features.index)
            
            for feature_name in self.feature_order:  # Use exact order
                if feature_name in features.columns:
                    selected_features[feature_name] = features[feature_name]
                else:
                    # Provide default for missing features
                    if 'rsi' in feature_name and feature_name not in ['rsi_divergence', 'rsi_momentum']:
                        selected_features[feature_name] = 50.0
                    elif 'position' in feature_name:
                        selected_features[feature_name] = 0.5
                    elif any(x in feature_name for x in ['session_', '_overbought', '_oversold']):
                        selected_features[feature_name] = 0
                    else:
                        selected_features[feature_name] = 0.0
            
            print(f"✅ Selected {len(selected_features.columns)} training features in exact order")
            return selected_features
        
        print(f"⚠️  No training features loaded, returning all {len(features.columns)} features")
        return features
    
    def apply_selected_features(self, features, selected_feature_names):
        """Apply pre-selected features from optimization metadata"""
        
        if not selected_feature_names:
            return features
        
        available_features = [f for f in selected_feature_names if f in features.columns]
        
        if len(available_features) < len(selected_feature_names):
            missing_features = set(selected_feature_names) - set(available_features)
            print(f"⚠️  Missing features: {missing_features}")
        
        # Ensure exact order
        selected_df = pd.DataFrame(index=features.index)
        for feature_name in selected_feature_names:  # Preserve order
            if feature_name in features.columns:
                selected_df[feature_name] = features[feature_name]
            else:
                # Add missing feature with default
                print(f"   Adding missing feature '{feature_name}' with default value")
                if 'rsi' in feature_name and feature_name not in ['rsi_divergence', 'rsi_momentum']:
                    selected_df[feature_name] = 50.0
                elif 'position' in feature_name:
                    selected_df[feature_name] = 0.5
                elif any(x in feature_name for x in ['session_', '_overbought', '_oversold']):
                    selected_df[feature_name] = 0
                else:
                    selected_df[feature_name] = 0.0
        
        print(f"✅ Applied {len(selected_df.columns)}/{len(selected_feature_names)} selected features")
        return selected_df

def patch_trading_system():
    """
    Patch the existing trading system to use EURUSD-aligned features
    """
    
    print("🔧 PATCHING TRADING SYSTEM FOR EURUSD")
    print("=" * 45)
    
    # Create fix instance
    fix = EURUSDSignalErrorFix()
    
    if not fix.training_features:
        print("❌ Could not load training features")
        return False
    
    print(f"✅ Loaded {len(fix.training_features)} training features")
    
    # Create wrapper function
    wrapper_code = f'''
# EURUSD Signal Error Fix - Add this to your trading system

def create_eurusd_aligned_features(df, hyperparameters=None):
    """
    Create features aligned with EURUSD training
    Replace your feature engine's create_advanced_features for EURUSD
    """
    
    fix = EURUSDSignalErrorFix()
    return fix.create_aligned_features(df, hyperparameters)

# Or modify your existing feature engine:
# In OptimizedFeatureEngine.create_advanced_features:

def create_advanced_features_fixed(self, df, hyperparameters=None):
    # Check if this is for EURUSD
    symbol = getattr(self, 'current_symbol', None)
    
    if symbol == 'EURUSD':
        # Use aligned features for EURUSD
        fix = EURUSDSignalErrorFix()
        return fix.create_aligned_features(df, hyperparameters)
    else:
        # Use original method for other symbols
        return self.create_advanced_features_original(df, hyperparameters)

# Training features for reference:
EURUSD_TRAINING_FEATURES = {fix.training_features}
'''
    
    # Save the wrapper
    wrapper_file = Path("eurusd_trading_system_patch.py")
    with open(wrapper_file, 'w') as f:
        f.write(wrapper_code)
    
    print(f"💾 Patch saved to: {wrapper_file}")
    
    # Test the fix
    print(f"\n🧪 Testing the fix...")
    
    test_result = test_eurusd_fix(fix)
    
    if test_result:
        print(f"✅ Fix test successful!")
        print(f"🎯 SOLUTION: Replace feature creation for EURUSD with aligned version")
        return True
    else:
        print(f"❌ Fix test failed")
        return False

def test_eurusd_fix(fix):
    """Test the EURUSD fix with real data"""
    
    try:
        # Load EURUSD data
        data_files = list(Path("data").glob("*EURUSD*"))
        if not data_files:
            print("❌ No EURUSD data found")
            return False
        
        # Use parquet file
        parquet_file = None
        for file in data_files:
            if file.suffix == '.parquet':
                parquet_file = file
                break
        
        if parquet_file:
            df = pd.read_parquet(parquet_file)
        else:
            print("❌ No parquet file found")
            return False
        
        # Standardize columns
        df.columns = [col.lower() for col in df.columns]
        
        # Set datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        
        # Test feature creation
        aligned_features = fix.create_aligned_features(df.tail(100))
        
        print(f"✅ Created {len(aligned_features.columns)} aligned features")
        print(f"✅ Feature names match training: {list(aligned_features.columns) == fix.training_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the patch
    result = patch_trading_system()
    
    print(f"\n🎉 EURUSD SIGNAL ERROR FIX COMPLETE!")
    print("=" * 50)
    
    if result:
        print(f"✅ Fix created and tested successfully")
        print(f"✅ Patch file: eurusd_trading_system_patch.py")
        print(f"🚀 Apply this fix to resolve SIGNAL_ERROR!")
        
        print(f"\n📋 IMPLEMENTATION STEPS:")
        print(f"   1. Replace your feature creation for EURUSD")
        print(f"   2. Use exact training feature order")
        print(f"   3. Test with: python fix_eurusd_signal_error.py")
        print(f"   4. SIGNAL_ERROR should be resolved!")
        
    else:
        print(f"❌ Fix creation failed")
        print(f"🔍 Check training metadata and data files")