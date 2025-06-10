"""
Notebook Import Example

This script demonstrates how to import and use the model_export_utils.py module
directly in the notebook without embedding all the code.
"""

# --- Copy and paste this into your notebook ---

# Import the model export utilities
from model_export_utils import run_backtest_and_export

# After training your model, run backtest and export
export_dir = "exported_models"

# Get the feature names used in the model
feature_names = features.columns.tolist()

# Run backtest and export the model with metrics
export_data = run_backtest_and_export(
    model=model,
    X_test=X_test,
    y_test=y_test,
    price_data=prices[(symbol, "close")],  # Adjust to match your price data structure
    symbol=symbol,
    feature_names=feature_names,
    lookback=1,  # Adjust based on your prediction horizon
    export_dir=export_dir
)

# Display the export results
print("Model exported successfully!")
print(f"Model name: {export_data['model_name']}")
print(f"Accuracy: {export_data['accuracy']:.4f}")
print(f"Sharpe ratio: {export_data['sharpe_ratio']:.4f}")
print(f"Total return: {export_data['total_return']:.4f}")
print(f"Win rate: {export_data['win_rate']:.4f}")
print(f"Profit factor: {export_data['profit_factor']:.4f}")
print(f"Max drawdown: {export_data['max_drawdown']:.4f}")

print("✅ Notebook import example is ready to use")
print("Copy and paste this into your notebook")
