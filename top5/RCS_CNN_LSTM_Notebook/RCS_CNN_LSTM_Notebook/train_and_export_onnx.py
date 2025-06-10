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
from async_data_loader import load_or_fetch
from model_training_utils import train_model_with_best_features, evaluate_model
import feature_set_utils
from input_shape_handler import ensure_compatible_input_shape
from model_export_utils import save_model_with_metrics

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

def build_cnn_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Build and train a CNN-LSTM model.
    
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
        
    Returns:
    --------
    keras.Model
        Trained model
    """
    # Get input shape
    input_shape = X_train.shape[1:]
    
    # Build model
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(50, return_sequences=False),
        layers.Dense(20, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return model

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
    
    # Export to ONNX
    onnx_path = os.path.join(EXPORT_DIR, f"{model_id}.onnx")
    
    try:
        # Ensure input shape is compatible
        expected_shape = model.input_shape[1:]  # Remove batch dimension
        X_train_compatible = ensure_compatible_input_shape(X_train, expected_shape)
        
        # Handle Sequential models differently
        if isinstance(model, tf.keras.Sequential):
            # Convert Sequential model to Functional model first
            inputs = tf.keras.Input(shape=X_train_compatible.shape[1:], name="input")
            outputs = model(inputs)
            functional_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_model")
            
            # Add explicit output names
            if not hasattr(functional_model, 'output_names') or not functional_model.output_names:
                functional_model.output_names = ['output']
            
            # Use the Functional model for ONNX conversion
            try:
                # Try direct conversion first
                spec = (tf.TensorSpec((None,) + X_train_compatible.shape[1:], tf.float32, name="input"),)
                onnx_model, _ = tf2onnx.convert.from_keras(functional_model, input_signature=spec, opset=13)
            except Exception as e:
                print(f"⚠️ First ONNX conversion attempt failed: {e}")
                print("Trying alternative conversion method...")
                
                # Try alternative conversion method
                model_proto, _ = tf2onnx.convert.from_function(
                    lambda x: functional_model(x),
                    input_signature=[tf.TensorSpec((None,) + X_train_compatible.shape[1:], tf.float32, name="input")],
                    opset=13
                )
                onnx_model = model_proto
        else:
            # For Functional models, use the standard approach
            try:
                spec = (tf.TensorSpec((None,) + X_train_compatible.shape[1:], tf.float32, name="input"),)
                onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            except Exception as e:
                print(f"⚠️ First ONNX conversion attempt failed: {e}")
                print("Trying alternative conversion method...")
                
                # Try alternative conversion method
                model_proto, _ = tf2onnx.convert.from_function(
                    lambda x: model(x),
                    input_signature=[tf.TensorSpec((None,) + X_train_compatible.shape[1:], tf.float32, name="input")],
                    opset=13
                )
                onnx_model = model_proto
        
        # Save the ONNX model
        onnx.save(onnx_model, onnx_path)
        print(f"  Saved ONNX model to {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        print("  This error is non-critical. The model is still saved in HDF5 format.")
    
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
    
    # Prepare metadata
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
        "onnx_path": onnx_path,
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
                    if os.path.exists(row["onnx_path"]):
                        os.remove(row["onnx_path"])
                combined = combined.iloc[:TOP_N]
            
            # Update CSV
            combined.to_csv(TOP_CSV, index=False)
            print(f"  Updated {TOP_CSV} with top {TOP_N} models.")
            
            # Update in-memory top_df for next iteration
            top_df = combined
            
        # Also train a model with random features for comparison
        from model_training_utils import train_model_with_random_features, evaluate_model
        
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
