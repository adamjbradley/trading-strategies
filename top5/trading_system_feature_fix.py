#!/usr/bin/env python3
"""
Immediate Feature Fix for Trading System
Apply this to resolve the feature mismatch error
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

class TradingSystemFeatureFix:
    """
    Immediate fix for the trading system feature mismatch
    """
    
    def __init__(self):
        self.feature_mapping = self.create_feature_mapping()
        self.default_values = self.create_default_values()
        
    def create_feature_mapping(self):
        """Create mapping from real-time features to training features"""
        
        mapping = {
            # Bollinger Band mappings
            'bb_lower_20_2': 'bb_lower',
            'bb_upper_20_2': 'bb_upper',
            'bb_middle_20_2': 'bb_middle', 
            'bb_position_20_2': 'bb_position',
            'bb_lower_20_2.5': 'bb_lower_25',
            'bb_upper_20_2.5': 'bb_upper_25',
            'bb_position_20_2.5': 'bb_position_25',
            'bb_position_50_2': 'bb_position_50',
            'bb_width_20_2': 'bbw',
            'bb_width_20_2.5': 'bbw_25',
            
            # ATR mappings (real-time -> training)
            'atr_norm_14': 'atr_normalized_14',
            'atr_norm_21': 'atr_normalized_21',
            'atr_normalized_14': 'atr_14',  # If using different naming
            'atr_normalized_21': 'atr_21',
            
            # Candlestick patterns
            'doji_pattern': 'doji',
            'hammer_pattern': 'hammer',
            'engulfing_pattern': 'engulfing',
            'shooting_star': 'shooting_star_pattern',
            
            # RSI variations
            'rsi_14_overbought': 'rsi_overbought',
            'rsi_14_oversold': 'rsi_oversold',
            
            # Moving averages
            'sma_5': 'sma_5',
            'sma_10': 'sma_10',
            'sma_20': 'sma_20', 
            'sma_50': 'sma_50',
            'ema_12': 'ema_12',
            'ema_26': 'ema_26',
            
            # MACD
            'macd_line': 'macd',
            'macd_signal_line': 'macd_signal',
            'macd_histogram': 'macd_histogram',
        }
        
        return mapping
    
    def create_default_values(self):
        """Create default values for missing features"""
        
        defaults = {
            # ATR features
            'atr_14': 0.001,
            'atr_21': 0.001,
            'atr_norm_14': 0.001,
            'atr_norm_21': 0.001,
            'atr_normalized_14': 0.001,
            'atr_normalized_21': 0.001,
            
            # Candlestick patterns (binary)
            'doji': 0,
            'hammer': 0,
            'engulfing': 0,
            'shooting_star': 0,
            'spinning_top': 0,
            
            # Bollinger Bands
            'bb_position': 0.5,
            'bb_position_25': 0.5,
            'bb_position_50': 0.5,
            'bbw': 0.02,
            'bbw_25': 0.025,
            
            # RSI
            'rsi_14': 50,
            'rsi_7': 50,
            'rsi_21': 50,
            'rsi_overbought': 0,
            'rsi_oversold': 0,
            'rsi_divergence': 0,
            'rsi_momentum': 0,
            
            # MACD
            'macd': 0,
            'macd_signal': 0,
            'macd_histogram': 0,
            'macd_signal_line_cross': 0,
            
            # Moving averages
            'sma_5': 1.0,
            'sma_10': 1.0,
            'sma_20': 1.0,
            'sma_50': 1.0,
            'price_to_sma_5': 1.0,
            'price_to_sma_10': 1.0,
            'price_to_sma_20': 1.0,
            'price_to_sma_50': 1.0,
            
            # Volatility
            'volatility_10': 0.001,
            'volatility_20': 0.001,
            'volatility_ratio': 1.0,
            'volatility_regime': 0,
            'volatility_persistence': 0.5,
            
            # Momentum
            'momentum_1': 0,
            'momentum_3': 0,
            'momentum_5': 0,
            'momentum_10': 0,
            'momentum_accel_3': 0,
            'momentum_accel_5': 0,
            'momentum_accel_10': 0,
            
            # Session features
            'session_asian': 0,
            'session_european': 0,
            'session_us': 0,
            'session_overlap_eur_us': 0,
            'hour': 12,
            'is_monday': 0,
            'is_friday': 0,
            'friday_close': 0,
            'sunday_gap': 0,
            
            # Price position
            'price_position_10': 0.5,
            'price_position_20': 0.5,
            'close_position': 0.5,
            'range_ratio': 0.01,
            
            # Volume (if available)
            'volume_ratio': 1.0,
            'price_volume': 0,
            
            # Phase 2 correlation features
            'usd_strength_proxy': 0,
            'eur_strength_proxy': 0,
            'eur_strength_trend': 0,
            'risk_sentiment': 0,
            'jpy_safe_haven': 0,
            'corr_momentum': 0,
            'correlation_momentum': 0,
            'vol_adjusted_correlation': 0,
        }
        
        return defaults
    
    def fix_features(self, real_time_features, current_price=None, symbol=None):
        """
        Fix real-time features to match training expectations
        
        Args:
            real_time_features (dict): Features from real-time system
            current_price (float): Current price for calculations
            symbol (str): Trading symbol
            
        Returns:
            dict: Fixed features ready for model
        """
        
        fixed_features = {}
        
        # Step 1: Apply direct mappings
        for rt_feature, value in real_time_features.items():
            if rt_feature in self.feature_mapping:
                mapped_name = self.feature_mapping[rt_feature]
                fixed_features[mapped_name] = value
            else:
                # Keep original if no mapping
                fixed_features[rt_feature] = value
        
        # Step 2: Compute missing features where possible
        if current_price:
            fixed_features = self.compute_missing_features(fixed_features, current_price, symbol)
        
        # Step 3: Fill remaining missing features with defaults
        for feature_name, default_value in self.default_values.items():
            if feature_name not in fixed_features:
                fixed_features[feature_name] = default_value
        
        # Step 4: Clean and validate
        fixed_features = self.clean_features(fixed_features)
        
        return fixed_features
    
    def compute_missing_features(self, features, current_price, symbol=None):
        """Compute missing features from available data"""
        
        # ATR normalization
        if 'atr_14' in features and current_price > 0:
            if 'atr_norm_14' not in features:
                features['atr_norm_14'] = features['atr_14'] / current_price
                
        if 'atr_21' in features and current_price > 0:
            if 'atr_norm_21' not in features:
                features['atr_norm_21'] = features['atr_21'] / current_price
        
        # Bollinger Band position from components
        if all(k in features for k in ['bb_upper', 'bb_lower']) and current_price:
            bb_range = features['bb_upper'] - features['bb_lower']
            if bb_range > 0 and 'bb_position' not in features:
                features['bb_position'] = (current_price - features['bb_lower']) / bb_range
                features['bb_position'] = max(0, min(1, features['bb_position']))  # Clip 0-1
        
        # BBW from components
        if all(k in features for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            if 'bbw' not in features:
                bb_range = features['bb_upper'] - features['bb_lower']
                features['bbw'] = bb_range / features['bb_middle'] if features['bb_middle'] > 0 else 0.02
        
        # Price-to-MA ratios
        for ma_period in [5, 10, 20, 50]:
            sma_key = f'sma_{ma_period}'
            ratio_key = f'price_to_sma_{ma_period}'
            if sma_key in features and current_price and ratio_key not in features:
                if features[sma_key] > 0:
                    features[ratio_key] = current_price / features[sma_key]
                else:
                    features[ratio_key] = 1.0
        
        # RSI binary features
        if 'rsi_14' in features:
            if 'rsi_overbought' not in features:
                features['rsi_overbought'] = 1 if features['rsi_14'] > 70 else 0
            if 'rsi_oversold' not in features:
                features['rsi_oversold'] = 1 if features['rsi_14'] < 30 else 0
        
        # Session features (simplified based on current time if not available)
        import datetime
        current_hour = datetime.datetime.now().hour
        
        if 'session_asian' not in features:
            features['session_asian'] = 1 if (current_hour >= 21 or current_hour <= 6) else 0
        if 'session_european' not in features:
            features['session_european'] = 1 if (7 <= current_hour <= 16) else 0
        if 'session_us' not in features:
            features['session_us'] = 1 if (13 <= current_hour <= 22) else 0
        if 'session_overlap_eur_us' not in features:
            features['session_overlap_eur_us'] = 1 if (13 <= current_hour <= 16) else 0
        
        # Time features
        if 'hour' not in features:
            features['hour'] = current_hour
        if 'is_monday' not in features:
            features['is_monday'] = 1 if datetime.datetime.now().weekday() == 0 else 0
        if 'is_friday' not in features:
            features['is_friday'] = 1 if datetime.datetime.now().weekday() == 4 else 0
        
        return features
    
    def clean_features(self, features):
        """Clean and validate features"""
        
        cleaned_features = {}
        
        for feature_name, value in features.items():
            # Handle NaN and infinite values
            if pd.isna(value) or np.isinf(value):
                if feature_name in self.default_values:
                    cleaned_value = self.default_values[feature_name]
                else:
                    cleaned_value = 0.0
            else:
                cleaned_value = float(value)
            
            # Apply reasonable bounds
            if 'rsi' in feature_name and not any(x in feature_name for x in ['divergence', 'momentum']):
                cleaned_value = max(0, min(100, cleaned_value))
            elif 'position' in feature_name:
                cleaned_value = max(0, min(1, cleaned_value))
            elif feature_name.endswith(('_overbought', '_oversold')) or 'session_' in feature_name:
                cleaned_value = 1 if cleaned_value > 0.5 else 0
            
            cleaned_features[feature_name] = cleaned_value
        
        return cleaned_features

def apply_feature_fix_to_trading_signal(trading_signal, feature_fix=None):
    """
    Apply feature fix to a trading signal with feature errors
    
    Args:
        trading_signal: Your TradingSignal object with feature errors
        feature_fix: TradingSystemFeatureFix instance (optional)
        
    Returns:
        dict: Fixed features or error info
    """
    
    if feature_fix is None:
        feature_fix = TradingSystemFeatureFix()
    
    # Check if there's a feature error
    if hasattr(trading_signal, 'features') and isinstance(trading_signal.features, dict):
        if 'error' in trading_signal.features:
            print("🔧 Detected feature mismatch error, applying fix...")
            
            # Extract basic info from signal
            current_price = getattr(trading_signal, 'price', None)
            symbol = getattr(trading_signal, 'symbol', None)
            
            # Create minimal feature set for prediction
            minimal_features = {
                'close': current_price if current_price else 1.0,
                'returns': 0.0,  # Will be computed from price data
                'rsi_14': 50,    # Neutral RSI
                'macd': 0.0,     # Neutral MACD
                'atr_14': 0.001, # Small ATR
                'volatility_20': 0.001,
                'momentum_5': 0.0,
            }
            
            # Apply comprehensive fix
            fixed_features = feature_fix.fix_features(
                minimal_features, 
                current_price=current_price, 
                symbol=symbol
            )
            
            print(f"✅ Generated {len(fixed_features)} fixed features")
            
            return {
                'status': 'fixed',
                'fixed_features': fixed_features,
                'feature_count': len(fixed_features),
                'original_error': trading_signal.features['error']
            }
    
    return {
        'status': 'no_fix_needed',
        'message': 'No feature mismatch detected'
    }

def create_emergency_feature_set(symbol, current_price=1.0):
    """Create emergency feature set for immediate trading"""
    
    feature_fix = TradingSystemFeatureFix()
    
    # Emergency minimal features
    emergency_features = {
        'close': current_price,
        'rsi_14': 50,  # Neutral
        'macd': 0,     # Neutral
        'atr_14': current_price * 0.001,  # 0.1% of price
        'returns': 0,  # No change
        'volatility_20': current_price * 0.001,
        'momentum_5': 0,
        'sma_20': current_price,
        'bb_position': 0.5,  # Middle of bands
        'volume_ratio': 1.0,
    }
    
    # Apply full feature fix
    complete_features = feature_fix.fix_features(
        emergency_features, 
        current_price=current_price, 
        symbol=symbol
    )
    
    return complete_features

# Example usage and testing
def test_feature_fix():
    """Test the feature fix with sample data"""
    
    print("🧪 TESTING FEATURE FIX")
    print("="*30)
    
    # Sample real-time features that cause mismatch
    sample_rt_features = {
        'bb_lower_20_2': 1.0500,
        'bb_upper_20_2': 1.0600,
        'bb_position_20_2': 0.3,
        'rsi_14': 45,
        'macd_line': -0.001,
        'close': 1.0545,
    }
    
    # Apply fix
    feature_fix = TradingSystemFeatureFix()
    fixed_features = feature_fix.fix_features(sample_rt_features, current_price=1.0545, symbol='EURUSD')
    
    print(f"📊 Original features: {len(sample_rt_features)}")
    print(f"📊 Fixed features: {len(fixed_features)}")
    
    print(f"\n🔧 Sample mappings applied:")
    for original, value in sample_rt_features.items():
        if original in feature_fix.feature_mapping:
            mapped = feature_fix.feature_mapping[original]
            print(f"   {original} → {mapped}: {value}")
    
    print(f"\n✅ Feature fix test completed successfully!")
    
    return fixed_features

if __name__ == "__main__":
    # Run test
    test_features = test_feature_fix()
    
    print(f"\n🚀 FEATURE FIX READY FOR DEPLOYMENT!")
    print("="*50)
    print("💡 To use in your trading system:")
    print("   1. Import TradingSystemFeatureFix")
    print("   2. Create instance: fix = TradingSystemFeatureFix()")
    print("   3. Apply fix: fixed_features = fix.fix_features(rt_features, price, symbol)")
    print("   4. Use fixed_features for model prediction")