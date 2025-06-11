"""
Define Core Features

This script defines the core_features variable for use in the RCS_CNN_LSTM notebook.
It can be executed directly in the notebook using the %run magic command.
"""

# Define the core_features variable
core_features = [
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
    'return_1d', 
    'return_3d', 
    'rolling_mean_5', 
    'rolling_std_5', 
    'momentum_slope'
]

print("✅ core_features variable defined with", len(core_features), "features")
