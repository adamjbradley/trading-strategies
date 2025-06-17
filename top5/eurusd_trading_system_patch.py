
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
EURUSD_TRAINING_FEATURES = ['volume', 'cci', 'stoch_k', 'stoch_d', 'rsi_7', 'rsi_momentum', 'rsi_14', 'rsi_divergence', 'adx', 'hour', 'rsi_21', 'volatility_regime', 'session_european', 'session_us', 'session_asian', 'engulfing', 'volume_ratio', 'is_monday', 'is_friday', 'session_overlap_eur_us', 'rsi_overbought', 'rsi_oversold', 'roc', 'doji', 'volatility_ratio', 'bb_position', 'price_position_10', 'price_position_20', 'roc_momentum']
