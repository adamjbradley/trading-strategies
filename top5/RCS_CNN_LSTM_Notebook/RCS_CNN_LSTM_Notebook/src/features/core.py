"""
Define Core Features

This script defines the core_features variable for use in the RCS_CNN_LSTM notebook.
It can be executed directly in the notebook using the %run magic command.
"""

# Define the core_features variable
core_features = [
    # Technical Indicators
    'rsi', 
    'macd', 
    'momentum', 
    'cci', 
    'atr', 
    'adx', 
    'stoch_k', 
    'stoch_d', 
    'roc', 
    'bbw',
    
    # Price Features
    'return_1d', 
    'return_3d', 
    'rolling_mean_5', 
    'rolling_std_5', 
    'momentum_slope',
    
    # Forex-Specific Trading Session Features
    'asian_session',
    'london_session', 
    'ny_session',
    'session_overlap',
    'session_volatility_ratio',
    
    # Currency Strength & Correlation Features
    'eur_strength_proxy',
    'eur_strength_trend',
    'usd_strength_impact',
    
    # Advanced Volatility Features
    'volatility_regime',
    'volatility_persistence',
    
    # Market Structure Features  
    'range_ratio',
    'close_position',
    'risk_sentiment',
    
    # Time-based Features
    'hour',
    'is_friday',
    'is_monday'
]

print("✅ core_features variable defined with", len(core_features), "features")
