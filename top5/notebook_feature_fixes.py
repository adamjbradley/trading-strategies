"""
Notebook Feature Engineering Fixes
==================================

This module provides enhanced feature engineering functions to bring the notebook
implementation up to par with the legacy code performance.

Usage:
    from notebook_feature_fixes import enhance_notebook_features
    
    # After basic feature engineering in notebook
    enhanced_features = enhance_notebook_features(data, symbol, ohlc, prices)
"""

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator


def calculate_enhanced_bollinger_features(close):
    """Calculate enhanced Bollinger Band features including bb_position"""
    bb_period = 20
    bb_sma = close.rolling(bb_period, min_periods=1).mean()
    bb_std = close.rolling(bb_period, min_periods=1).std()
    
    bb_upper = bb_sma + (bb_std * 2)
    bb_lower = bb_sma - (bb_std * 2)
    
    # Bollinger Band Width (normalized)
    bbw = (bb_upper - bb_lower) / bb_sma
    
    # Bollinger Band Position (0-1 normalized)
    bb_range = bb_upper - bb_lower + 1e-10  # Avoid division by zero
    bb_position = ((close - bb_lower) / bb_range).clip(0, 1)
    
    return bbw, bb_position


def calculate_multi_timeframe_rsi(close):
    """Calculate RSI across multiple timeframes with derived features"""
    rsi_features = {}
    
    # Multi-timeframe RSI
    for period in [7, 14, 21, 50]:
        rsi_features[f'rsi_{period}'] = RSIIndicator(close=close, window=period).rsi()
    
    # RSI divergence (difference between fast and slow)
    rsi_features['rsi_divergence'] = rsi_features['rsi_14'] - rsi_features['rsi_21']
    
    # RSI momentum (3-period change)
    rsi_features['rsi_momentum'] = rsi_features['rsi_14'].diff(3)
    
    return pd.DataFrame(rsi_features, index=close.index)


def calculate_volatility_features(high, low, close):
    """Calculate advanced volatility features"""
    volatility_features = {}
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Multiple ATR periods
    for period in [14, 21]:
        volatility_features[f'atr_{period}'] = true_range.rolling(period, min_periods=1).mean()
    
    # ATR percentage of price
    volatility_features['atr_pct_14'] = volatility_features['atr_14'] / close
    
    # Normalized ATR
    atr_ma_50 = volatility_features['atr_14'].rolling(50, min_periods=1).mean()
    volatility_features['atr_normalized_14'] = volatility_features['atr_14'] / atr_ma_50
    
    # Volatility regime (binary: high/low volatility)
    volatility_features['volatility_regime'] = (
        volatility_features['atr_14'] > atr_ma_50
    ).astype(int)
    
    # Volatility persistence (correlation with lagged values)
    volatility_features['volatility_persistence'] = (
        volatility_features['atr_14'].rolling(10, min_periods=1)
        .corr(volatility_features['atr_14'].shift(1))
    )
    
    return pd.DataFrame(volatility_features, index=close.index)


def calculate_session_features(data_index, symbol):
    """Calculate forex session features with weekend filtering"""
    session_features = pd.DataFrame(index=data_index)
    
    hours = data_index.hour
    weekday = data_index.weekday
    
    # Weekend detection
    is_weekend = (weekday >= 5).astype(int)
    market_open = (1 - is_weekend)
    
    # Trading sessions (UTC times)
    # Asian: 21:00-06:00 UTC (Tokyo)
    # European: 07:00-16:00 UTC (London)
    # US: 13:00-22:00 UTC (New York)
    
    session_asian_raw = ((hours >= 21) | (hours <= 6)).astype(int)
    session_european_raw = ((hours >= 7) & (hours <= 16)).astype(int)
    session_us_raw = ((hours >= 13) & (hours <= 22)).astype(int)
    
    # Apply weekend filtering
    session_features['session_asian'] = session_asian_raw * market_open
    session_features['session_european'] = session_european_raw * market_open
    session_features['session_us'] = session_us_raw * market_open
    
    # Session overlaps
    session_features['session_overlap_eur_us'] = (
        ((hours >= 13) & (hours <= 16)).astype(int) * market_open
    )
    
    # Enhanced time features
    session_features['hour'] = hours
    session_features['is_monday'] = (weekday == 0).astype(int)
    session_features['is_friday'] = (weekday == 4).astype(int)
    
    # Gap trading features
    session_features['friday_close'] = ((weekday == 4) & (hours >= 21)).astype(int)
    session_features['sunday_gap'] = ((weekday == 0) & (hours <= 6)).astype(int)
    
    return session_features


def calculate_currency_strength_features(returns, symbol):
    """Calculate currency-specific strength indicators"""
    strength_features = pd.DataFrame(index=returns.index)
    
    # USD strength proxy
    if 'USD' in symbol:
        if symbol.startswith('USD'):
            # USD is base currency
            strength_features['usd_strength_proxy'] = returns.rolling(10, min_periods=1).mean()
        elif symbol.endswith('USD'):
            # USD is quote currency (inverse relationship)
            strength_features['usd_strength_proxy'] = (-returns).rolling(10, min_periods=1).mean()
        else:
            strength_features['usd_strength_proxy'] = 0
    else:
        strength_features['usd_strength_proxy'] = 0
    
    # EUR specific features
    if symbol == "EURUSD":
        eur_momentum = returns
        strength_features['eur_strength_proxy'] = eur_momentum.rolling(5, min_periods=1).mean()
        strength_features['eur_strength_trend'] = strength_features['eur_strength_proxy'].diff(3)
    else:
        strength_features['eur_strength_proxy'] = 0
        strength_features['eur_strength_trend'] = 0
    
    return strength_features


def calculate_market_structure_features(high, low, close):
    """Calculate market microstructure features"""
    structure_features = pd.DataFrame(index=close.index)
    
    # Range ratio (normalized by close price)
    structure_features['range_ratio'] = (high - low) / close
    
    # Close position within the day's range (0 = low, 1 = high)
    range_size = high - low + 1e-10  # Avoid division by zero
    structure_features['close_position'] = ((close - low) / range_size).clip(0, 1)
    
    # High-low spread
    structure_features['high_low_pct'] = (high - low) / close
    
    return structure_features


def enhance_notebook_features(data, symbol, ohlc, prices=None):
    """
    Enhance notebook features to match legacy implementation quality
    
    Parameters:
    -----------
    data : pd.DataFrame
        Existing features from notebook
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    ohlc : pd.DataFrame
        OHLC data with MultiIndex columns
    prices : pd.DataFrame, optional
        Full prices DataFrame for cross-asset features
        
    Returns:
    --------
    pd.DataFrame
        Enhanced features DataFrame
    """
    # Copy existing features
    enhanced_data = data.copy()
    
    # Extract price series
    close = ohlc[(symbol, "close")]
    high = ohlc[(symbol, "high")]
    low = ohlc[(symbol, "low")]
    
    print("🔧 Enhancing notebook features to match legacy implementation...")
    
    # 1. Enhanced Bollinger Band features
    print("  📊 Adding enhanced Bollinger Band features...")
    bbw, bb_position = calculate_enhanced_bollinger_features(close)
    enhanced_data['bbw'] = bbw
    enhanced_data['bb_position'] = bb_position
    
    # 2. Multi-timeframe RSI
    print("  📊 Adding multi-timeframe RSI features...")
    rsi_features = calculate_multi_timeframe_rsi(close)
    for col in rsi_features.columns:
        enhanced_data[col] = rsi_features[col]
    
    # 3. Advanced volatility features
    print("  📊 Adding advanced volatility features...")
    volatility_features = calculate_volatility_features(high, low, close)
    for col in volatility_features.columns:
        enhanced_data[col] = volatility_features[col]
    
    # 4. Session-based features
    print("  📊 Adding session-based features...")
    session_features = calculate_session_features(enhanced_data.index, symbol)
    for col in session_features.columns:
        enhanced_data[col] = session_features[col]
    
    # 5. Currency strength features
    print("  📊 Adding currency strength features...")
    returns = close.pct_change()
    strength_features = calculate_currency_strength_features(returns, symbol)
    for col in strength_features.columns:
        enhanced_data[col] = strength_features[col]
    
    # 6. Market structure features
    print("  📊 Adding market structure features...")
    structure_features = calculate_market_structure_features(high, low, close)
    for col in structure_features.columns:
        enhanced_data[col] = structure_features[col]
    
    # 7. Additional technical indicators
    print("  📊 Adding additional technical indicators...")
    
    # Stochastic Oscillator enhancements
    low_14 = low.rolling(14, min_periods=1).min()
    high_14 = high.rolling(14, min_periods=1).max()
    enhanced_data['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    enhanced_data['stoch_d'] = enhanced_data['stoch_k'].rolling(3, min_periods=1).mean()
    
    # Rate of Change
    enhanced_data['roc'] = close.pct_change(10) * 100
    
    # 8. Clean data
    print("  🧹 Cleaning enhanced features...")
    
    # Replace infinities
    enhanced_data = enhanced_data.replace([np.inf, -np.inf], np.nan)
    
    # Feature-specific NaN handling
    for col in enhanced_data.columns:
        if 'rsi' in col or 'stoch' in col:
            # Oscillators default to 50 (neutral)
            enhanced_data[col] = enhanced_data[col].fillna(50)
        elif 'position' in col or 'bb_position' in col:
            # Position indicators default to 0.5 (middle)
            enhanced_data[col] = enhanced_data[col].fillna(0.5)
        elif 'session' in col or 'is_' in col or 'regime' in col:
            # Binary features default to 0
            enhanced_data[col] = enhanced_data[col].fillna(0)
        else:
            # Others: forward fill, backward fill, then 0
            enhanced_data[col] = enhanced_data[col].ffill().bfill().fillna(0)
    
    print(f"✅ Feature enhancement complete! Added {len(enhanced_data.columns) - len(data.columns)} new features")
    print(f"   Total features: {len(enhanced_data.columns)}")
    
    return enhanced_data


def validate_enhanced_features(features):
    """Validate that enhanced features meet quality standards"""
    validation_results = {}
    
    # Check for required features
    required_features = [
        'bb_position', 'volatility_persistence', 'session_asian',
        'session_european', 'session_us', 'rsi_divergence',
        'rsi_momentum', 'volatility_regime'
    ]
    
    missing_features = [f for f in required_features if f not in features.columns]
    validation_results['missing_features'] = missing_features
    
    # Check for NaN values
    nan_counts = features.isna().sum()
    validation_results['features_with_nans'] = nan_counts[nan_counts > 0].to_dict()
    
    # Check for infinite values
    inf_counts = np.isinf(features.select_dtypes(include=[np.number])).sum()
    validation_results['features_with_infs'] = inf_counts[inf_counts > 0].to_dict()
    
    # Check value ranges
    range_issues = []
    
    # Oscillators should be 0-100
    for col in features.columns:
        if 'rsi' in col or 'stoch' in col:
            if not features[col].between(0, 100).all():
                range_issues.append(f"{col}: values outside 0-100")
    
    # Position indicators should be 0-1
    for col in ['bb_position', 'close_position']:
        if col in features.columns:
            if not features[col].between(0, 1).all():
                range_issues.append(f"{col}: values outside 0-1")
    
    # Binary features should be 0 or 1
    binary_features = [col for col in features.columns if 'session' in col or 'is_' in col or 'regime' in col]
    for col in binary_features:
        if col in features.columns:
            unique_values = features[col].unique()
            if not set(unique_values).issubset({0, 1}):
                range_issues.append(f"{col}: non-binary values {unique_values}")
    
    validation_results['range_issues'] = range_issues
    
    # Summary
    validation_results['is_valid'] = (
        len(missing_features) == 0 and
        len(validation_results['features_with_nans']) == 0 and
        len(validation_results['features_with_infs']) == 0 and
        len(range_issues) == 0
    )
    
    return validation_results


if __name__ == "__main__":
    print("Notebook Feature Enhancement Module")
    print("=" * 50)
    print("Available functions:")
    print("  - enhance_notebook_features(): Add all missing legacy features")
    print("  - validate_enhanced_features(): Validate feature quality")
    print("\nImport this module in your notebook to enhance feature engineering.")