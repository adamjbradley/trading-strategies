"""
Model Export Example

This script demonstrates how to use the model_export_utils.py module
to export a trained model with backtest metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_export_utils import run_backtest_and_export

# Example usage in a notebook cell:
"""
# --- Step 1: Import the model export utilities ---
from model_export_utils import run_backtest_and_export

# --- Step 2: Define core_features if not already defined ---
core_features = [
    'rsi', 'macd', 'momentum', 'cci', 'atr', 'adx', 'stoch_k', 'stoch_d', 
    'roc', 'bbw', 'return_1d', 'return_3d', 'rolling_mean_5', 
    'rolling_std_5', 'momentum_slope'
]

# --- Step 3: After training your model, run backtest and export ---
# Assuming 'model' is your trained model, 'X_test' and 'y_test' are your test data,
# 'prices' is a pandas Series of closing prices, and 'symbol' is the trading symbol

# Get the feature names used in the model
feature_names = features.columns.tolist()

# Create a directory for the exported models
export_dir = "exported_models"

# Run backtest and export the model with metrics
export_data = run_backtest_and_export(
    model=model,
    X_test=X_test,
    y_test=y_test,
    price_data=prices[(symbol, "close")],  # Adjust this to match your price data structure
    symbol=symbol,
    feature_names=feature_names,
    lookback=1,  # Adjust this based on your prediction horizon
    export_dir=export_dir
)

# --- Step 4: Display the export results ---
print("Model exported successfully!")
print(f"Model name: {export_data['model_name']}")
print(f"Accuracy: {export_data['accuracy']:.4f}")
print(f"Sharpe ratio: {export_data['sharpe_ratio']:.4f}")
print(f"Total return: {export_data['total_return']:.4f}")
print(f"Win rate: {export_data['win_rate']:.4f}")
print(f"Profit factor: {export_data['profit_factor']:.4f}")
print(f"Max drawdown: {export_data['max_drawdown']:.4f}")

# --- Step 5: Plot the backtest results ---
# This assumes you've calculated cumulative returns elsewhere
# If not, you can calculate them from the signals and price returns

# Example plot code:
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns, label='Strategy')
plt.title(f"Backtest Results: {symbol}")
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
"""

print("✅ Example code for model export is ready to use")
print("Copy and paste the relevant sections into your notebook")
