import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# Normalize symbols like "EUR/USD" -> "EURUSD" for providers such as Polygon
def normalize_symbol(symbol: str) -> str:
    """Return an uppercase symbol without separator characters."""
    return symbol.replace('/', '').upper()

# Utility: parse date and convert to datetime
def parse_date(ts, fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.strptime(ts, fmt)

def _resolve_key(key: str, env_var: str) -> str:
    """Return the given key or fall back to an environment variable."""
    return key if key and not key.startswith("YOUR_") else os.getenv(env_var, "")


def load_polygon_data(symbol, api_key, interval="minute", limit=500):
    api_key = _resolve_key(api_key, "POLYGON_API_KEY")
    symbol_clean = normalize_symbol(symbol)
    url = (
        "https://api.polygon.io/v2/aggs/ticker/C:"
        f"{symbol_clean}/range/1/{interval}/2023-01-01/2023-12-31"
        f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
    )
    resp = requests.get(url)
    data = resp.json().get("results", [])
    return pd.DataFrame([{
        "timestamp": datetime.fromtimestamp(d["t"] / 1000),
        "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"], "volume": d["v"]
    } for d in data])

def load_twelve_data(symbol, api_key, interval="1min", outputsize=500):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    resp = requests.get(url)
    data = resp.json().get("values", [])
    return pd.DataFrame([{
        "timestamp": parse_date(d["datetime"]),
        "open": float(d["open"]), "high": float(d["high"]),
        "low": float(d["low"]), "close": float(d["close"]), "volume": float(d.get("volume", 0))
    } for d in reversed(data)])

def load_alpha_vantage(symbol, api_key, interval="1min"):
    url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={symbol[:3]}&to_symbol={symbol[3:]}&interval={interval}&apikey={api_key}&outputsize=compact"
    resp = requests.get(url)
    time_series = resp.json().get(f"Time Series FX ({interval})", {})
    return pd.DataFrame([{
        "timestamp": parse_date(ts),
        "open": float(d["1. open"]), "high": float(d["2. high"]),
        "low": float(d["3. low"]), "close": float(d["4. close"]), "volume": 0
    } for ts, d in sorted(time_series.items())])

def load_currencystack(symbol, api_key):
    url = f"https://api.currencystack.io/forex?base={symbol[:3]}&target={symbol[3:]}&apikey={api_key}"
    resp = requests.get(url)
    data = resp.json()
    return pd.DataFrame([{
        "timestamp": datetime.now(),
        "open": data["rate"], "high": data["rate"], "low": data["rate"],
        "close": data["rate"], "volume": 0
    }])

def load_tiingo(symbol, api_key, limit=500):
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}
    url = f"https://api.tiingo.com/tiingo/fx/{symbol}/prices?resampleFreq=1min&limit={limit}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    return pd.DataFrame([{
        "timestamp": parse_date(d["date"]),
        "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"], "volume": d.get("volume", 0)
    } for d in data])


def load_yfinance(symbol, interval="1m", period="1y", **_):
    """Load data from Yahoo Finance."""
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
    )
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.rename(
        columns={
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

def save_optimized(df, symbol, provider):
    symbol_clean = symbol.replace('/', '').upper()
    filename = f"data/{provider}_{symbol_clean}"
    df.to_parquet(f"{filename}.parquet", index=False)
    df.to_hdf(f"{filename}.h5", key='df', mode='w')
    print(f"📦 Saved to {filename}.parquet and {filename}.h5")

def load_or_fetch(symbol, provider, loader_func, api_key, force_refresh=False, **kwargs):
    import os
    symbol_clean = symbol.replace('/', '').upper()
    base_path = f"data/{provider}_{symbol_clean}"
    parquet_path = f"{base_path}.parquet"
    h5_path = f"{base_path}.h5"

    if not force_refresh:
        if os.path.exists(parquet_path):
            print(f"✅ Loaded cached data from {parquet_path}")
            return pd.read_parquet(parquet_path)
        elif os.path.exists(h5_path):
            print(f"✅ Loaded cached data from {h5_path}")
            return pd.read_hdf(h5_path)

    print(f"🔄 Fetching fresh data from {provider} API")
    df = loader_func(symbol, api_key, **kwargs)
    save_optimized(df, symbol, provider)
    return df

def assess_quality(df):
    """Print basic quality metrics for a DataFrame."""
    if df.empty:
        print("⚠️ DataFrame is empty")
        return {}
    stats = {
        "rows": len(df),
        "missing": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }
    if "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp")
        stats["start"] = df_sorted["timestamp"].iloc[0]
        stats["end"] = df_sorted["timestamp"].iloc[-1]
    print("=== Data Quality ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    return stats


def compare_dataframes(df_a, df_b, name_a="A", name_b="B"):
    """Compare two dataframes on timestamp overlap and close price similarity."""
    if df_a.empty or df_b.empty:
        print("One of the dataframes is empty; cannot compare.")
        return
    merged = pd.merge(df_a, df_b, on="timestamp", suffixes=(f"_{name_a}", f"_{name_b}"))
    if merged.empty:
        print("No overlapping timestamps for comparison")
        return
    corr = merged[f"close_{name_a}"].corr(merged[f"close_{name_b}"])
    diff = (merged[f"close_{name_a}"] - merged[f"close_{name_b}"]).abs().mean()
    print(f"Correlation ({name_a} vs {name_b}): {corr:.4f}")
    print(f"Mean absolute close difference: {diff:.6f}")
    return {"correlation": corr, "mean_abs_diff": diff}

