import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from async_data_loader import load_or_fetch

DATA_DIR = "data"
ONNX_DIR = "onnx_models"
os.makedirs(ONNX_DIR, exist_ok=True)

features = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume", "hl_range", "oc_range"]

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".parquet") and "_2year_" in fname:
        pair_info = fname.replace(".parquet", "")
        print(f"\nProcessing {pair_info}...")
        # Parse symbol, timeframe, broker from filename
        parts = pair_info.split("_2year_")
        if len(parts) != 2:
            print("  Skipping: filename format not recognized.")
            continue
        prefix, broker = parts
        try:
            symbol, timeframe = prefix.split("_")
        except Exception:
            print("  Skipping: could not parse symbol/timeframe.")
            continue
        # Use main data loader
        df = load_or_fetch(
            symbol=symbol,
            provider="metatrader",
            loader_func=None,
            api_key="",
            interval=timeframe,
            broker=broker,
        )
        if not all(f in df.columns for f in features):
            print("  Skipping: missing required features.")
            continue
        df = df.dropna(subset=features)
        df["target"] = df["close"].shift(-1)
        df = df.dropna(subset=["target"])
        X = df[features]
        y = df["target"]
        if len(df) < 100:
            print("  Skipping: not enough data.")
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f"  Trained RandomForestRegressor on {len(X_train)} samples.")

        # Export to ONNX
        initial_type = [("float_input", FloatTensorType([None, len(features)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx_path = os.path.join(ONNX_DIR, f"{pair_info}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  Saved ONNX model to {onnx_path}")
