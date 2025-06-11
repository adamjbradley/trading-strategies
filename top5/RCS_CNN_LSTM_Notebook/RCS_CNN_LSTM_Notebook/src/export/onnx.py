import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx
import onnx
from datetime import datetime
from sklearn.metrics import r2_score
from ..data.loader import load_or_fetch
# Import from model_training_utils instead of models.training  
try:
    # Try importing from the main model_training_utils module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from model_training_utils import train_model_with_best_features, evaluate_model
except ImportError:
    try:
        # Fallback to relative import
        from ..models.training import train_model_with_best_features, evaluate_model
    except ImportError:
        # Final fallback - direct import from models.training
        from src.models.training import train_model_with_best_features, evaluate_model
import src.features.selection
from ..utils.shape_handler import ensure_compatible_input_shape
from ..models.evaluation import save_model_with_metrics
try:
    from ..models.training import train_model_with_random_features
except ImportError:
    from model_training_utils import train_model_with_random_features

# Configuration
DATA_DIR = "data"
EXPORT_DIR = "exported_models"
TOP_CSV = "top10_models.csv"
TOP_N = 10

os.makedirs(EXPORT_DIR, exist_ok=True)

# Default features to use if no best feature set is found
DEFAULT_FEATURES = [
    "open", "high", "low", "close", "tick_volume", "spread", "real_volume", 
    "hl_range", "oc_range", "rsi_14", "ema_9", "sma_20", "atr_14"
]

# Load or initialize the top 10 CSV
if os.path.exists(TOP_CSV):
    top_df = pd.read_csv(TOP_CSV)
else:
    top_df = pd.DataFrame(columns=[
        "model_id", "symbol", "timeframe", "broker", "model_type", 
        "features", "metric", "cumulative_return", "buy_hold_return", 
        "onnx_path", "h5_path", "timestamp"
    ])

def build_cnn_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, **kwargs):
    """
    Build and train a CNN-LSTM model with improved regularization.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation targets
    epochs : int, default=50
        Number of epochs to train
    batch_size : int, default=32
        Batch size for training
    **kwargs : dict
        Additional keyword arguments for regularization
        
    Returns:
    --------
    keras.Model
        Trained model
    """
    # Import the improved model builder
    from ..models.cnn_lstm import build_cnn_lstm_model as improved_model_builder
    
    # Use the improved model builder with regularization
    return improved_model_builder(X_train, y_train, X_val, y_val, epochs, batch_size, **kwargs)

def process_symbol(symbol, timeframe="H1", broker="metatrader"):
    """
    Process a symbol by training a model using the best feature set and exporting it to ONNX.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    timeframe : str, default="H1"
        Timeframe for the data
    broker : str, default="metatrader"
        Broker for the data
        
    Returns:
    --------
    dict
        Dictionary containing model metadata
    """
    print(f"\nProcessing {symbol}_{timeframe}_{broker}...")
    
    # Load data
    df = load_or_fetch(
        symbol=symbol,
        provider="metatrader",
        loader_func=None,
        api_key="",
        interval=timeframe,
        broker=broker,
    )
    
    # Check if data is valid
    if df is None or len(df) < 100:
        print(f"  Skipping: not enough data for {symbol}.")
        return None
    
    # Prepare target
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])
    
    # Train model using best features
    model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test = train_model_with_best_features(
        symbol=symbol,
        data=df,
        model_builder_func=build_cnn_lstm_model,
        n_features=15,
        use_saved_features=True,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Calculate strategy returns
    test_df = pd.DataFrame(index=range(len(X_test)))
    test_df["close"] = df["close"].values[-len(X_test):]
    test_df["target"] = y_test
    test_df["pred"] = (model.predict(X_test) > 0.5).astype(int).flatten()
    test_df["signal"] = test_df["pred"]
    test_df["returns"] = test_df["close"].pct_change().fillna(0)
    test_df["strategy_returns"] = test_df["signal"].shift(1).fillna(0) * test_df["returns"]
    test_df["equity"] = (1 + test_df["strategy_returns"]).cumprod()
    cumulative_return = test_df["equity"].iloc[-1] - 1 if len(test_df) > 0 else np.nan
    
    # Buy & hold returns
    if len(test_df) > 0:
        buy_hold_return = (test_df["close"].iloc[-1] / test_df["close"].iloc[0]) - 1
    else:
        buy_hold_return = np.nan
    
    # Generate model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{symbol}_{timeframe}_{broker}_{timestamp}"
    
    # Save model to H5
    h5_path = os.path.join("models", f"{model_id}.h5")
    os.makedirs("models", exist_ok=True)
    model.save(h5_path)
    print(f"  Saved H5 model to {h5_path}")
    
    # Export to ONNX with NumPy 2.0 compatibility
    onnx_path = os.path.join(EXPORT_DIR, f"{model_id}.onnx")
    
    try:
        # Import the NumPy 2.0 compatible ONNX export function
        from .onnx_numpy2_fix import export_model_to_onnx_numpy2_safe, ensure_numpy_compatibility
        
        # Ensure input shape is compatible
        expected_shape = model.input_shape[1:]  # Remove batch dimension
        X_train_compatible = ensure_compatible_input_shape(X_train, expected_shape)
        
        # Create test input for verification (small sample)
        test_input = ensure_numpy_compatibility(X_train_compatible[:min(5, len(X_train_compatible))])
        
        # Use the NumPy 2.0 compatible export function
        export_success = export_model_to_onnx_numpy2_safe(
            model=model,
            output_path=onnx_path,
            input_shape=X_train_compatible.shape[1:],
            test_input=test_input,
            model_name=f"{symbol}_CNN_LSTM"
        )
        
        if export_success:
            print(f"  ✅ ONNX model exported successfully to {onnx_path}")
        else:
            print(f"  ⚠️ ONNX export failed, but H5 model is available")
            onnx_path = None  # Set to None to indicate ONNX export failed
            
    except ImportError as e:
        print(f"⚠️ NumPy 2.0 compatible ONNX export not available: {e}")
        print("  Falling back to standard ONNX export...")
        
        # Fallback to original ONNX export method
        try:
            # Ensure input shape is compatible
            expected_shape = model.input_shape[1:]  # Remove batch dimension
            X_train_compatible = ensure_compatible_input_shape(X_train, expected_shape)
            
            # Convert to ONNX using standard method
            spec = (tf.TensorSpec((None,) + X_train_compatible.shape[1:], tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            
            # Save the ONNX model
            onnx.save(onnx_model, onnx_path)
            print(f"  Saved ONNX model to {onnx_path}")
        except Exception as fallback_e:
            print(f"⚠️ Fallback ONNX export also failed: {fallback_e}")
            print("  This error is non-critical. The model is still saved in HDF5 format.")
            onnx_path = None
            
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        print("  This error is non-critical. The model is still saved in HDF5 format.")
        onnx_path = None
    
    # Save metrics
    save_model_with_metrics(
        model_path=h5_path,
        symbol=symbol,
        metrics={
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "cumulative_return": cumulative_return,
            "buy_hold_return": buy_hold_return
        }
    )
    
    # Prepare metadata (handle case where ONNX export failed)
    new_row = {
        "model_id": model_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "broker": broker,
        "model_type": "CNN-LSTM",
        "features": ",".join(feature_names),
        "metric": metrics["f1_score"],
        "cumulative_return": cumulative_return,
        "buy_hold_return": buy_hold_return,
        "onnx_path": onnx_path if onnx_path else "",  # Empty string if ONNX export failed
        "h5_path": h5_path,
        "timestamp": timestamp
    }
    
    return new_row

def main():
    # List of symbols to process
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY"]
    
    for symbol in symbols:
        # Process symbol using best features
        new_row = process_symbol(symbol)
        
        if new_row is not None:
            # Combine and sort
            global top_df
            combined = pd.concat([top_df, pd.DataFrame([new_row])], ignore_index=True)
            combined = combined.sort_values(by="metric", ascending=False).reset_index(drop=True)
            
            # If more than TOP_N, remove the lowest
            if len(combined) > TOP_N:
                to_remove = combined.iloc[TOP_N:]
                for _, row in to_remove.iterrows():
                    # Clean up ONNX files if they exist
                    if row["onnx_path"] and os.path.exists(row["onnx_path"]):
                        try:
                            os.remove(row["onnx_path"])
                            print(f"  Removed old ONNX file: {row['onnx_path']}")
                        except Exception as e:
                            print(f"  ⚠️ Could not remove {row['onnx_path']}: {e}")
                    # Clean up H5 files if they exist
                    if row["h5_path"] and os.path.exists(row["h5_path"]):
                        try:
                            os.remove(row["h5_path"])
                            print(f"  Removed old H5 file: {row['h5_path']}")
                        except Exception as e:
                            print(f"  ⚠️ Could not remove {row['h5_path']}: {e}")
                combined = combined.iloc[:TOP_N]
            
            # Update CSV
            combined.to_csv(TOP_CSV, index=False)
            print(f"  Updated {TOP_CSV} with top {TOP_N} models.")
            
            # Update in-memory top_df for next iteration
            top_df = combined
            
        # Also train a model with random features for comparison
        print(f"\nTraining model with random features for {symbol}...")
        
        # Load data
        df = load_or_fetch(
            symbol=symbol,
            provider="metatrader",
            loader_func=None,
            api_key="",
            interval="H1",
            broker="metatrader",
        )
        
        # Check if data is valid
        if df is None or len(df) < 100:
            print(f"  Skipping random features: not enough data for {symbol}.")
            continue
        
        # Prepare target
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna(subset=["target"])
        
        # Train model with random features
        random_model, random_features, X_train, y_train, X_val, y_val, X_test, y_test = train_model_with_random_features(
            symbol=symbol,
            data=df,
            model_builder_func=build_cnn_lstm_model,
            n_features=15,
            random_seed=42,
            epochs=50,
            batch_size=32
        )
        
        # Evaluate model
        metrics = evaluate_model(random_model, X_test, y_test)
        print(f"  Random features model accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
