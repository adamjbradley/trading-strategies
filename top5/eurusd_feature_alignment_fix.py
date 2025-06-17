
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
        features[f'ema_{period}'] = close.ewm(span=period).mean()
    
    # 3. Basic price features
    features['close'] = close
    features['high'] = high
    features['low'] = low
    features['open'] = open_price
    features['volume'] = volume
    
    # === REST OF FEATURES ===
    # (Add all other features as before...)
    
    # 4. Select only the features that were used in training
    training_feature_list = ['volume', 'cci', 'stoch_k', 'stoch_d', 'rsi_7', 'rsi_momentum', 'rsi_14', 'rsi_divergence', 'adx', 'hour', 'rsi_21', 'volatility_regime', 'session_european', 'session_us', 'session_asian', 'engulfing', 'volume_ratio', 'is_monday', 'is_friday', 'session_overlap_eur_us', 'rsi_overbought', 'rsi_oversold', 'roc', 'doji', 'volatility_ratio', 'bb_position', 'price_position_10', 'price_position_20', 'roc_momentum']
    
    # Keep only training features
    aligned_features = {}
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
