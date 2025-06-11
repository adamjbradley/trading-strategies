import pandas as pd

def generate_labels_rolling(df, target_col="close", horizon=10, method="binary"):
    # Shifted future returns (target) over horizon
    df = df.copy()
    df["future_return"] = df[target_col].shift(-horizon) / df[target_col] - 1

    if method == "binary":
        df["label"] = (df["future_return"] > 0).astype(int)
    elif method == "ternary":
        df["label"] = df["future_return"].apply(lambda x: 1 if x > 0.001 else (-1 if x < -0.001 else 0))
    elif method == "regression":
        df["label"] = df["future_return"]
    else:
        raise ValueError(f"Unsupported labeling method: {method}")

    return df.dropna(subset=["label"])
