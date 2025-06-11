"""
Core Features Module

This module defines the core features used in the RCS_CNN_LSTM model.
"""

# Define the core features used in the model
def get_core_features():
    """
    Returns a list of core features used in the model.
    
    Returns:
    --------
    list
        List of core feature names
    """
    return [
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
