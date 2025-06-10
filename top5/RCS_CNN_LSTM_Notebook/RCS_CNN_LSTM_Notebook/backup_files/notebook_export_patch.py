"""
Notebook Export Patch

This script provides a patch for the RCS_CNN_LSTM notebook to integrate
the model export utilities for proper model versioning and metrics tracking.
"""

# --- Copy and paste this entire cell into your notebook ---

# Import necessary libraries
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define the model export functions
def export_model_with_metrics(model, X_test, y_test, symbol, feature_names, 
                             backtest_returns=None, export_dir="models"):
    """
    Export a trained model with a unique name and save backtest metrics to CSV.
    """
    # Create export directory if it doesn't exist
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # Generate a unique model name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol}_CNN_LSTM_{timestamp}"
    model_path = os.path.join(export_dir, model_name)
    
    # Save Keras H5 model
    h5_path = f"{model_path}.h5"
    model.save(h5_path)
    print(f"✅ Saved Keras model to: {h5_path}")
    
    # Export to ONNX
    onnx_path = f"{model_path}.onnx"
    
    # Handle different input shapes for CNN vs non-CNN models
    if len(model.input_shape) == 4 and len(X_test.shape) == 3:
        # Reshape for CNN models
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        input_sample = X_test_reshaped[:1]  # Take first sample for shape inference
    else:
        input_sample = X_test[:1]  # Take first sample for shape inference
    
    # Convert to ONNX
    try:
        import tf2onnx
        import onnx
        from input_shape_handler import ensure_compatible_input_shape
        
        # Ensure input shape is compatible
        expected_shape = model.input_shape[1:]  # Remove batch dimension
        input_sample_compatible = ensure_compatible_input_shape(input_sample, expected_shape)
        
        # Handle Sequential models differently
        if isinstance(model, tf.keras.Sequential):
            # Convert Sequential model to Functional model first
            inputs = tf.keras.Input(shape=input_sample_compatible.shape[1:], name="input")
            outputs = model(inputs)
            functional_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_model")
            
            # Add explicit output names
            if not hasattr(functional_model, 'output_names') or not functional_model.output_names:
                functional_model.output_names = ['output']
            
            # Use the Functional model for ONNX conversion
            try:
                # Try direct conversion first
                spec = (tf.TensorSpec((None,) + input_sample_compatible.shape[1:], tf.float32, name="input"),)
                onnx_model, _ = tf2onnx.convert.from_keras(functional_model, input_signature=spec, opset=13)
            except Exception as e:
                print(f"⚠️ First ONNX conversion attempt failed: {e}")
                print("Trying alternative conversion method...")
                
                # Try alternative conversion method
                model_proto, _ = tf2onnx.convert.from_function(
                    lambda x: functional_model(x),
                    input_signature=[tf.TensorSpec((None,) + input_sample_compatible.shape[1:], tf.float32, name="input")],
                    opset=13
                )
                onnx_model = model_proto
        else:
            # For Functional models, use the standard approach
            try:
                spec = (tf.TensorSpec((None,) + input_sample_compatible.shape[1:], tf.float32, name="input"),)
                onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            except Exception as e:
                print(f"⚠️ First ONNX conversion attempt failed: {e}")
                print("Trying alternative conversion method...")
                
                # Try alternative conversion method
                model_proto, _ = tf2onnx.convert.from_function(
                    lambda x: model(x),
                    input_signature=[tf.TensorSpec((None,) + input_sample_compatible.shape[1:], tf.float32, name="input")],
                    opset=13
                )
                onnx_model = model_proto
        
        # Save the ONNX model
        onnx.save(onnx_model, onnx_path)
        print(f"✅ Saved ONNX model to: {onnx_path}")
        onnx_export_success = True
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        onnx_export_success = False
    
    # Calculate model performance metrics
    metrics = calculate_model_metrics(model, X_test, y_test)
    
    # Calculate backtest metrics if returns are provided
    if backtest_returns is not None:
        backtest_metrics = calculate_backtest_metrics(backtest_returns)
        metrics.update(backtest_metrics)
    
    # Add metadata
    metadata = {
        "model_name": model_name,
        "symbol": symbol,
        "timestamp": timestamp,
        "features": feature_names,
        "num_features": len(feature_names),
        "test_samples": len(y_test),
        "h5_path": h5_path,
        "onnx_path": onnx_path if onnx_export_success else "export_failed",
        "onnx_export_success": onnx_export_success
    }
    
    # Combine metrics and metadata
    export_data = {**metadata, **metrics}
    
    # Save to CSV
    csv_path = f"{model_path}_metrics.csv"
    pd.DataFrame([export_data]).to_csv(csv_path, index=False)
    print(f"📊 Saved model metrics to: {csv_path}")
    
    return export_data

def calculate_model_metrics(model, X_test, y_test):
    """
    Calculate model performance metrics.
    """
    # Handle different input shapes for CNN vs non-CNN models
    if hasattr(model, 'input_shape') and len(model.input_shape) == 4 and len(X_test.shape) == 3:
        # Reshape for CNN models
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        y_pred_proba = model.predict(X_test_reshaped).flatten()
    else:
        y_pred_proba = model.predict(X_test).flatten()
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Ensure y_test and y_pred have the same length
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]
    y_pred_proba = y_pred_proba[:min_len]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    
    # Add ROC AUC if we have both classes
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
    
    return metrics

def calculate_backtest_metrics(returns):
    """
    Calculate backtest performance metrics.
    """
    # Convert to numpy array if it's not already
    returns = np.array(returns)
    
    # Calculate metrics
    total_return = np.sum(returns)
    annualized_return = total_return * (252 / len(returns))  # Assuming 252 trading days per year
    
    # Calculate drawdown
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = running_max - cumulative_returns
    max_drawdown = np.max(drawdown)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    daily_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    annualized_sharpe = daily_sharpe * np.sqrt(252)  # Assuming 252 trading days per year
    
    # Calculate win rate
    win_rate = np.sum(returns > 0) / len(returns)
    
    # Calculate profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": annualized_sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(returns)
    }

def run_backtest_and_export(model, X_test, y_test, price_data, symbol, feature_names, 
                           lookback=1, export_dir="models"):
    """
    Run a backtest on the model, calculate metrics, and export everything.
    """
    # Handle different input shapes for CNN vs non-CNN models
    if hasattr(model, 'input_shape') and len(model.input_shape) == 4 and len(X_test.shape) == 3:
        # Reshape for CNN models
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        y_pred_proba = model.predict(X_test_reshaped).flatten()
    else:
        y_pred_proba = model.predict(X_test).flatten()
    
    # Generate signals (-1 for sell, 1 for buy)
    signals = np.where(y_pred_proba > 0.5, 1, -1)
    
    # Calculate returns (assuming price_data is aligned with X_test)
    price_returns = np.log(price_data.shift(-lookback) / price_data).iloc[-len(signals):].values
    strategy_returns = price_returns * signals
    
    # Export model with metrics
    export_data = export_model_with_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        symbol=symbol,
        feature_names=feature_names,
        backtest_returns=strategy_returns,
        export_dir=export_dir
    )
    
    # Plot backtest results
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(strategy_returns), label='Strategy')
    plt.plot(np.cumsum(price_returns), label='Buy & Hold', linestyle='--')
    plt.title(f"Backtest Results: {symbol}")
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return export_data

# --- Example usage (replace with your actual code) ---
"""
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
"""

print("✅ Notebook export patch is ready to use")
print("Copy and paste this entire cell into your notebook")
