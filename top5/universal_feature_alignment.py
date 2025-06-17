#!/usr/bin/env python3
"""
Universal Feature Alignment Fix for All Trading Symbols
Replaces the OptimizedFeatureEngine to ensure training/inference feature matching
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

class UniversalFeatureAligner:
    """
    Universal feature alignment for all trading symbols
    Loads training metadata and creates matching features
    """
    
    def __init__(self, models_path: str = "exported_models"):
        self.models_path = Path(models_path)
        self.symbol_metadata = {}
        self.feature_selector = None
        self.scaler = None
        self.selected_features = None
        
    def load_symbol_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load training metadata for a specific symbol"""
        
        # Check if already loaded
        if symbol in self.symbol_metadata:
            return self.symbol_metadata[symbol]
        
        # Find metadata files for this symbol
        metadata_files = list(self.models_path.glob(f"{symbol}_training_metadata_*.json"))
        
        if not metadata_files:
            print(f"⚠️  No training metadata found for {symbol}")
            return None
        
        # Get the most recent metadata file
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            
            self.symbol_metadata[symbol] = metadata
            print(f"✅ Loaded metadata for {symbol}: {latest_metadata.name}")
            return metadata
            
        except Exception as e:
            print(f"❌ Failed to load metadata for {symbol}: {e}")
            return None
    
    def create_advanced_features(self, df: pd.DataFrame, symbol: str, hyperparameters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Create features aligned with training for any symbol
        This is the universal replacement for OptimizedFeatureEngine.create_advanced_features
        """
        
        # Load symbol metadata
        metadata = self.load_symbol_metadata(symbol)
        
        # Create comprehensive feature set
        features = self._create_comprehensive_features(df, hyperparameters)
        
        # If we have training metadata, align features exactly
        if metadata and 'selected_features' in metadata:
            training_features = metadata['selected_features']
            aligned_features = self._align_with_training_features(features, training_features, symbol)
            print(f"✅ {symbol}: Aligned {len(aligned_features.columns)} features with training")
            return aligned_features
        
        # Fallback: use all features if no metadata
        print(f"⚠️  {symbol}: Using all {len(features.columns)} features (no training metadata)")
        return features
    
    def _create_comprehensive_features(self, df: pd.DataFrame, hyperparameters: Dict[str, Any] = None) -> pd.DataFrame:
        """Create comprehensive feature set (superset of all possible features)"""
        
        features = pd.DataFrame(index=df.index)
        
        # Get price data
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        open_price = df.get('open', close)
        volume = df.get('tick_volume', df.get('volume', pd.Series(100, index=df.index)))
        
        # Hyperparameter controls
        use_rcs = hyperparameters.get('use_rcs_features', True) if hyperparameters else True
        use_cross_pair = hyperparameters.get('use_cross_pair_features', True) if hyperparameters else True
        
        # === BASIC PRICE FEATURES ===
        features['close'] = close
        features['high'] = high
        features['low'] = low
        features['open'] = open_price
        features['volume'] = volume
        
        # Price relationships
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        features['high_low_pct'] = (high - low) / close
        features['close_position'] = (close - low) / (high - low + 1e-10)
        
        # === MOVING AVERAGES ===
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = close.rolling(period).mean()
            features[f'ema_{period}'] = close.ewm(span=period).mean()
        
        # === RSI FAMILY ===
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
        
        # === MACD ===
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd - macd_signal
        
        # === BOLLINGER BANDS ===
        sma_20 = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_upper'] = sma_20 + (bb_std * 2)
        features['bb_lower'] = sma_20 - (bb_std * 2)
        features['bb_middle'] = sma_20
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        features['bbw'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        
        # === TECHNICAL INDICATORS ===
        # CCI
        tp = (high + low + close) / 3
        features['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean()))))
        
        # ATR and ADX
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
        
        # === TIME FEATURES ===
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['is_monday'] = (features.index.dayofweek == 0).astype(int)
        features['is_friday'] = (features.index.dayofweek == 4).astype(int)
        features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
        
        # === SESSION FEATURES ===
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
        
        # === VOLUME FEATURES ===
        for period in [5, 10, 20]:
            vol_sma = volume.rolling(period).mean()
            features[f'volume_sma_{period}'] = vol_sma
            features[f'volume_ratio'] = volume / (vol_sma + 1)
        
        features['price_volume'] = close * volume
        
        # === VOLATILITY FEATURES ===
        vol_5 = close.pct_change().rolling(5).std()
        vol_20 = close.pct_change().rolling(20).std()
        features['volatility_5'] = vol_5
        features['volatility_20'] = vol_20
        features['volatility_regime'] = (vol_5 > vol_20).astype(int)
        features['volatility_ratio'] = vol_5 / (vol_20 + 1e-10)
        
        # === PRICE POSITION FEATURES ===
        for period in [10, 20]:
            rolling_min = close.rolling(period).min()
            rolling_max = close.rolling(period).max()
            features[f'price_position_{period}'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-10)
        
        # === CANDLESTICK PATTERNS ===
        features['doji'] = (np.abs(close - open_price) < (high - low) * 0.1).astype(int)
        features['hammer'] = ((low < close * 0.99) & (high < close * 1.01)).astype(int)
        features['engulfing'] = features['doji']  # Simplified
        
        # === RCS FEATURES (if enabled) ===
        if use_rcs:
            # Relative Currency Strength features
            features['rcs_momentum'] = features['returns'].rolling(5).mean()
            features['rcs_strength'] = features['rsi_14'] - 50
        
        # === CROSS-PAIR FEATURES (if enabled) ===
        if use_cross_pair:
            # Simplified cross-pair correlation
            features['cross_correlation'] = features['returns'].rolling(20).corr(features['returns'].shift(1))
        
        # === CLEAN FEATURES ===
        # Forward fill and backward fill
        features = features.ffill().bfill()
        
        # Fill remaining NaNs with appropriate defaults
        for col in features.columns:
            if features[col].isnull().any():
                if 'ratio' in col or 'position' in col:
                    features[col] = features[col].fillna(1.0)
                elif 'rsi' in col or 'stoch' in col:
                    features[col] = features[col].fillna(50.0)
                elif any(x in col for x in ['session_', '_overbought', '_oversold', 'is_']):
                    features[col] = features[col].fillna(0)
                else:
                    features[col] = features[col].fillna(0.0)
        
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)
        
        return features
    
    def _align_with_training_features(self, features: pd.DataFrame, training_features: List[str], symbol: str) -> pd.DataFrame:
        """Align features with training feature set in exact order"""
        
        aligned_features = pd.DataFrame(index=features.index)
        
        missing_features = []
        
        for feature_name in training_features:  # Preserve exact order
            if feature_name in features.columns:
                aligned_features[feature_name] = features[feature_name]
            else:
                # Provide default for missing features
                missing_features.append(feature_name)
                if 'rsi' in feature_name and feature_name not in ['rsi_divergence', 'rsi_momentum']:
                    aligned_features[feature_name] = 50.0
                elif 'position' in feature_name:
                    aligned_features[feature_name] = 0.5
                elif any(x in feature_name for x in ['session_', '_overbought', '_oversold', 'is_']):
                    aligned_features[feature_name] = 0
                else:
                    aligned_features[feature_name] = 0.0
        
        if missing_features:
            print(f"⚠️  {symbol}: Added defaults for {len(missing_features)} missing features")
        
        return aligned_features
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with training metadata"""
        
        symbols = set()
        for file in self.models_path.glob("*_training_metadata_*.json"):
            # Extract symbol from filename
            symbol = file.name.split('_')[0]
            symbols.add(symbol)
        
        return sorted(list(symbols))

# Create the universal aligner instance
universal_aligner = UniversalFeatureAligner()

def create_aligned_features_for_symbol(df: pd.DataFrame, symbol: str, hyperparameters: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Universal function to create aligned features for any symbol
    This replaces OptimizedFeatureEngine.create_advanced_features
    """
    return universal_aligner.create_advanced_features(df, symbol, hyperparameters)

if __name__ == "__main__":
    # Test the universal aligner
    print("🔄 Testing Universal Feature Aligner")
    print("=" * 40)
    
    aligner = UniversalFeatureAligner()
    available_symbols = aligner.get_available_symbols()
    
    print(f"✅ Found metadata for {len(available_symbols)} symbols:")
    for symbol in available_symbols:
        metadata = aligner.load_symbol_metadata(symbol)
        if metadata and 'selected_features' in metadata:
            print(f"   {symbol}: {len(metadata['selected_features'])} training features")
    
    print(f"\n🎯 Universal Feature Aligner Ready!")
    print(f"   Use: create_aligned_features_for_symbol(df, symbol, hyperparameters)")