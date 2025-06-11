import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization
import tf2onnx
import onnx
from async_data_loader import load_or_fetch
from input_shape_handler import ensure_compatible_input_shape

def download_yahoo_data(symbols, start, end):
    data = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end)
        df = df[['Close']].rename(columns={'Close': sym})
        data[sym] = df
    prices = pd.concat(data.values(), axis=1)
    prices.columns = symbols
    prices.dropna(inplace=True)
    return prices

def compute_rcs(logrets):
    pairs = logrets.columns
    currencies = list(set([s[:3] for s in pairs] + [s[3:6] for s in pairs]))
    rcs_data = {c: [] for c in currencies}
    for i in range(len(logrets)):
        row = logrets.iloc[i]
        daily_strength = {c: 0 for c in currencies}
        counts = {c: 0 for c in currencies}
        for pair, ret in row.items():
            base, quote = pair[:3], pair[3:]
            daily_strength[base] += ret
            daily_strength[quote] -= ret
            counts[base] += 1
            counts[quote] += 1
        for c in currencies:
            avg = daily_strength[c] / counts[c] if counts[c] else 0
            rcs_data[c].append(avg)
    return pd.DataFrame(rcs_data, index=logrets.index)

def add_technical_indicators(df, price_col):
    import ta
    indicators = pd.DataFrame(index=df.index)
    indicators['rsi'] = ta.momentum.RSIIndicator(close=df[price_col]).rsi()
    indicators['macd'] = ta.trend.MACD(df[price_col]).macd()
    indicators['momentum'] = ta.momentum.ROCIndicator(close=df[price_col]).roc()
    indicators['cci'] = ta.trend.CCIIndicator(high=df[price_col], low=df[price_col], close=df[price_col]).cci()
    return indicators

def create_sequences(features, lookback):
    X = np.array([features[i-lookback:i] for i in range(lookback, len(features))])
    return X

def build_cnn_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description="RCS CNN+LSTM FX Prediction Pipeline")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--timeframe", type=str, default="H1")
    parser.add_argument("--broker", type=str, default="amp_global")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--export_onnx", action="store_true")
    args = parser.parse_args()

    # Step 1: Load broker-specific data using main data loader
    print(f"Loading {args.symbol} {args.timeframe} {args.broker} using main data loader...")
    df = load_or_fetch(
        symbol=args.symbol,
        provider="metatrader",
        loader_func=None,
        api_key="",
        interval=args.timeframe,
        broker=args.broker,
    )

    # Step 2: Feature engineering
    features = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume", "hl_range", "oc_range"]
    df = df.dropna(subset=features)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=features + ["target"])
    X = df[features].values
    y = df["target"].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Rolling window sequences
    lookback = args.lookback
    X_seq = []
    y_seq = []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i-lookback:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    print("X_seq shape:", X_seq.shape, "y_seq shape:", y_seq.shape)

    # Step 4: Train/test split
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Step 5: Build and train model
    model = build_cnn_lstm_model(X_seq.shape[1:])
    model.summary()
    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, verbose=2)

    # Step 6: Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Step 7: Export
    if args.export_onnx:
        # Create export directory if it doesn't exist
        export_dir = "exported_models"
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            # Ensure input shape is compatible
            expected_shape = model.input_shape[1:]  # Remove batch dimension
            X_seq_compatible = ensure_compatible_input_shape(X_seq, expected_shape)
            
            # Handle Sequential models differently
            if isinstance(model, tf.keras.Sequential):
                # Convert Sequential model to Functional model first
                inputs = tf.keras.Input(shape=X_seq_compatible.shape[1:], name="input")
                outputs = model(inputs)
                functional_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_model")
                
                # Add explicit output names
                if not hasattr(functional_model, 'output_names') or not functional_model.output_names:
                    functional_model.output_names = ['output']
                
                # Use the Functional model for ONNX conversion
                try:
                    # Try direct conversion first
                    spec = (tf.TensorSpec((None,) + X_seq_compatible.shape[1:], tf.float32, name="input"),)
                    onnx_model, _ = tf2onnx.convert.from_keras(functional_model, input_signature=spec, opset=13)
                except Exception as e:
                    print(f"⚠️ First ONNX conversion attempt failed: {e}")
                    print("Trying alternative conversion method...")
                    
                    # Try alternative conversion method
                    model_proto, _ = tf2onnx.convert.from_function(
                        lambda x: functional_model(x),
                        input_signature=[tf.TensorSpec((None,) + X_seq_compatible.shape[1:], tf.float32, name="input")],
                        opset=13
                    )
                    onnx_model = model_proto
            else:
                # For Functional models, use the standard approach
                try:
                    spec = (tf.TensorSpec((None,) + X_seq_compatible.shape[1:], tf.float32, name="input"),)
                    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
                except Exception as e:
                    print(f"⚠️ First ONNX conversion attempt failed: {e}")
                    print("Trying alternative conversion method...")
                    
                    # Try alternative conversion method
                    model_proto, _ = tf2onnx.convert.from_function(
                        lambda x: model(x),
                        input_signature=[tf.TensorSpec((None,) + X_seq_compatible.shape[1:], tf.float32, name="input")],
                        opset=13
                    )
                    onnx_model = model_proto
            
            # Save the ONNX model
            onnx_path = os.path.join(export_dir, f"{args.symbol}_{args.timeframe}_2year_{args.broker}_cnn_lstm.onnx")
            onnx.save(onnx_model, onnx_path)
            print(f"✅ Saved ONNX model to: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
            print("This error is non-critical. The model is still available in memory.")

if __name__ == "__main__":
    main()
