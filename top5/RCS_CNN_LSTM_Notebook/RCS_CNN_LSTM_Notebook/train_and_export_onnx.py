import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from async_data_loader import load_or_fetch

DATA_DIR = "data"
ONNX_DIR = "onnx_models"
TOP_CSV = "top10_models.csv"
TOP_N = 10

os.makedirs(ONNX_DIR, exist_ok=True)

features = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume", "hl_range", "oc_range"]

# Load or initialize the top 10 CSV
if os.path.exists(TOP_CSV):
    top_df = pd.read_csv(TOP_CSV)
else:
    top_df = pd.DataFrame(columns=[
        "model_id", "symbol", "timeframe", "broker", "model_type", "n_estimators", "random_state",
        "features", "metric", "cumulative_return", "buy_hold_return", "onnx_path"
    ])

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

        # Evaluate model
        y_pred = model.predict(X_test)
        metric = r2_score(y_test, y_pred)
        print(f"  Model R^2 on test set: {metric:.4f}")

        # Calculate strategy returns (simple long/flat: long if prediction > current price, else flat)
        test_df = X_test.copy()
        test_df["close"] = df.loc[X_test.index, "close"]
        test_df["target"] = y_test
        test_df["pred"] = y_pred
        test_df["signal"] = (test_df["pred"] > test_df["close"]).astype(int)
        test_df["returns"] = test_df["close"].pct_change().fillna(0)
        test_df["strategy_returns"] = test_df["signal"].shift(1).fillna(0) * test_df["returns"]
        test_df["equity"] = (1 + test_df["strategy_returns"]).cumprod()
        cumulative_return = test_df["equity"].iloc[-1] - 1 if len(test_df) > 0 else np.nan

        # Buy & hold returns
        if len(test_df) > 0:
            buy_hold_return = (test_df["close"].iloc[-1] / test_df["close"].iloc[0]) - 1
        else:
            buy_hold_return = np.nan

        # Prepare metadata
        new_row = {
            "model_id": pair_info,
            "symbol": symbol,
            "timeframe": timeframe,
            "broker": broker,
            "model_type": "RandomForestRegressor",
            "n_estimators": 100,
            "random_state": 42,
            "features": ",".join(features),
            "metric": metric,
            "cumulative_return": cumulative_return,
            "buy_hold_return": buy_hold_return,
            "onnx_path": os.path.join(ONNX_DIR, f"{pair_info}.onnx")
        }

        # Combine and sort
        combined = pd.concat([top_df, pd.DataFrame([new_row])], ignore_index=True)
        combined = combined.sort_values(by="metric", ascending=False).reset_index(drop=True)
        # If more than TOP_N, remove the lowest
        if len(combined) > TOP_N:
            to_remove = combined.iloc[TOP_N:]
            for _, row in to_remove.iterrows():
                if os.path.exists(row["onnx_path"]):
                    os.remove(row["onnx_path"])
            combined = combined.iloc[:TOP_N]

        # If this model is in the top N, save ONNX and update CSV
        if pair_info in combined["model_id"].values:
            # Export to ONNX
            initial_type = [("float_input", FloatTensorType([None, len(features)]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            onnx_path = os.path.join(ONNX_DIR, f"{pair_info}.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"  Saved ONNX model to {onnx_path}")
            # Update CSV
            combined.to_csv(TOP_CSV, index=False)
            print(f"  Updated {TOP_CSV} with top {TOP_N} models.")
        else:
            print("  Model did not make it into the top 10. ONNX not saved.")

        # Update in-memory top_df for next iteration
        top_df = combined
