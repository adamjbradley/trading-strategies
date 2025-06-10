"""
Model Export Utilities

This module provides utilities for exporting trained models.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from input_shape_handler import ensure_compatible_input_shape

def save_model_with_metrics(model_path, symbol, metrics, export_dir="exported_models"):
    """
    Save model metrics to a CSV file.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    metrics : dict
        Dictionary containing model metrics
    export_dir : str, default="exported_models"
        Directory to save the metrics to
        
    Returns:
    --------
    str
        Path to the saved metrics file
    """
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    # Extract model name from path
    model_name = os.path.basename(model_path).replace('.h5', '')
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame([{
        "model_name": model_name,
        "symbol": symbol,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        **metrics
    }])
    
    # Save metrics to CSV
    metrics_path = os.path.join(export_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved metrics to: {metrics_path}")
    
    # Update all models metrics
    all_metrics_path = os.path.join(export_dir, "all_models_metrics.csv")
    if os.path.exists(all_metrics_path):
        all_metrics = pd.read_csv(all_metrics_path)
        all_metrics = pd.concat([all_metrics, metrics_df], ignore_index=True)
    else:
        all_metrics = metrics_df
    
    all_metrics.to_csv(all_metrics_path, index=False)
    print(f"✅ Updated all models metrics in: {all_metrics_path}")
    
    return metrics_path

def run_backtest_and_export(model, X_test, y_test, price_data, symbol, feature_names, lookback=1, export_dir="exported_models"):
    """
    Run a backtest and export the model if it's better than the last 10 models.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test targets
    price_data : pandas.Series
        Price data for backtesting
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    feature_names : list
        List of feature names used in the model
    lookback : int, default=1
        Number of days to look back for prediction
    export_dir : str, default="exported_models"
        Directory to export the model to
        
    Returns:
    --------
    dict
        Dictionary containing export data
    """
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Run backtest
    signals = y_pred * 2 - 1  # Convert 0/1 to -1/1
    
    # Align signals with price data
    if len(signals) < len(price_data):
        price_data = price_data[-len(signals):]
    elif len(signals) > len(price_data):
        signals = signals[-len(price_data):]
    
    # Calculate returns
    returns = price_data.pct_change().fillna(0).values
    strategy_returns = signals * returns
    
    # Calculate metrics
    total_return = np.sum(strategy_returns)
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
    
    # Create model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol}_CNN_LSTM_{timestamp}"
    
    # Check if this model is better than the last 10 models
    metrics_file = os.path.join(export_dir, "all_models_metrics.csv")
    
    if os.path.exists(metrics_file):
        # Load metrics
        metrics_df = pd.read_csv(metrics_file)
        
        # Filter metrics for this symbol
        symbol_metrics = metrics_df[metrics_df["symbol"] == symbol]
        
        # Get the last 10 models
        last_10 = symbol_metrics.sort_values("timestamp", ascending=False).head(10)
        
        # Check if this model is better than the last 10
        better_than_last_10 = False
        
        if len(last_10) == 0:
            # No previous models for this symbol
            better_than_last_10 = True
        else:
            # Check if accuracy is better than the average of the last 10
            avg_accuracy = last_10["accuracy"].mean()
            if accuracy > avg_accuracy:
                better_than_last_10 = True
            
            # Check if Sharpe ratio is better than the average of the last 10
            avg_sharpe = last_10["sharpe_ratio"].mean()
            if sharpe_ratio > avg_sharpe:
                better_than_last_10 = True
    else:
        # No previous models
        better_than_last_10 = True
    
    # Export data
    export_data = {
        "model_name": model_name,
        "symbol": symbol,
        "timestamp": timestamp,
        "accuracy": accuracy,
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return,
        "feature_names": feature_names,
        "better_than_last_10": better_than_last_10
    }
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        "model_name": model_name,
        "symbol": symbol,
        "timestamp": timestamp,
        "accuracy": accuracy,
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return,
        "num_features": len(feature_names)
    }])
    
    # Save metrics to CSV
    metrics_path = os.path.join(export_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Update all models metrics
    if os.path.exists(metrics_file):
        all_metrics = pd.read_csv(metrics_file)
        all_metrics = pd.concat([all_metrics, metrics_df], ignore_index=True)
    else:
        all_metrics = metrics_df
    
    all_metrics.to_csv(metrics_file, index=False)
    
    # Export model if it's better than the last 10
    if better_than_last_10:
        # Save Keras HDF5 model
        h5_path = os.path.join(export_dir, f"{model_name}.h5")
        model.save(h5_path)
        print(f"✅ Saved Keras model to: {h5_path}")
        
        # Save ONNX model
        try:
            import tf2onnx
            import onnx
            
            # Ensure input shape is compatible
            expected_shape = model.input_shape[1:]  # Remove batch dimension
            X_test_compatible = ensure_compatible_input_shape(X_test, expected_shape)
            
            # Handle Sequential models differently
            if isinstance(model, tf.keras.Sequential):
                # Convert Sequential model to Functional model first
                inputs = tf.keras.Input(shape=X_test_compatible.shape[1:], name="input")
                outputs = model(inputs)
                functional_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_model")
                
                # Add explicit output names
                if not hasattr(functional_model, 'output_names') or not functional_model.output_names:
                    functional_model.output_names = ['output']
                
                # Use the Functional model for ONNX conversion
                try:
                    # Try direct conversion first
                    spec = (tf.TensorSpec((None,) + X_test_compatible.shape[1:], tf.float32, name="input"),)
                    onnx_model, _ = tf2onnx.convert.from_keras(functional_model, input_signature=spec, opset=13)
                except Exception as e:
                    print(f"⚠️ First ONNX conversion attempt failed: {e}")
                    print("Trying alternative conversion method...")
                    
                    # Try alternative conversion method
                    model_proto, _ = tf2onnx.convert.from_function(
                        lambda x: functional_model(x),
                        input_signature=[tf.TensorSpec((None,) + X_test_compatible.shape[1:], tf.float32, name="input")],
                        opset=13
                    )
                    onnx_model = model_proto
            else:
                # For Functional models, use the standard approach
                try:
                    spec = (tf.TensorSpec((None,) + X_test_compatible.shape[1:], tf.float32, name="input"),)
                    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
                except Exception as e:
                    print(f"⚠️ First ONNX conversion attempt failed: {e}")
                    print("Trying alternative conversion method...")
                    
                    # Try alternative conversion method
                    model_proto, _ = tf2onnx.convert.from_function(
                        lambda x: model(x),
                        input_signature=[tf.TensorSpec((None,) + X_test_compatible.shape[1:], tf.float32, name="input")],
                        opset=13
                    )
                    onnx_model = model_proto
            
            # Save the ONNX model
            onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
            onnx.save(onnx_model, onnx_path)
            print(f"✅ Saved ONNX model to: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
            print("This error is non-critical. The model is still saved in HDF5 format.")
    else:
        print(f"⚠️ Model not exported: Not better than the last 10 models")
    
    return export_data

print("✅ Model export utilities loaded")
